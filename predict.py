import os.path
import torch.utils.data
from utils import *


def parser_args():
    parser = argparse.ArgumentParser(description="Eval multiple datasets ")
    parser.add_argument('--config',
                        default='/content/drive/MyDrive/PolypModel/configs/eval_model.yaml',
                        help='config file path')

    args = parser.parse_args()

    return args


class Eval_Pipeline:

    def __init__(self, args):
        self.args = args
        self.args['print_log'] = True
        self.output_device = self.args['device']

    def load_data(self, phase):

        data_loader = dict()
        if phase == 'eval':
            val1_dataset = EvalDataset(image_root=self.args['data']['test']['img1_root'],
                                       gt_root=self.args['data']['test']['gt1_root'],
                                       eval_size=self.args['data']['test']['img_size'])  # CVC-300 dataset

            val2_dataset = EvalDataset(image_root=self.args['data']['test']['img2_root'],
                                       gt_root=self.args['data']['test']['gt2_root'],
                                       eval_size=self.args['data']['test']['img_size'])  # CVC-ClinicDB dataset

            val3_dataset = EvalDataset(image_root=self.args['data']['test']['img3_root'],
                                       gt_root=self.args['data']['test']['gt3_root'],
                                       eval_size=self.args['data']['test']['img_size'])  # CVC-ColonDB dataset

            val4_dataset = EvalDataset(image_root=self.args['data']['test']['img4_root'],
                                       gt_root=self.args['data']['test']['gt4_root'],
                                       eval_size=self.args['data']['test']['img_size'])  # ETIS-LaribPolypDB

            val5_dataset = EvalDataset(image_root=self.args['data']['test']['img5_root'],
                                       gt_root=self.args['data']['test']['gt5_root'],
                                       eval_size=self.args['data']['test']['img_size'])  # Kvsair-SEG

            data_loader['cvc-300'] = torch.utils.data.DataLoader(dataset=val1_dataset,
                                                                 batch_size=len(val1_dataset),
                                                                 shuffle=True,
                                                                 num_workers=self.args['data']['workers_per_gpu'],
                                                                 drop_last=False, worker_init_fn=init_seed)

            data_loader['cvc-clinicDB'] = torch.utils.data.DataLoader(dataset=val2_dataset,
                                                                      batch_size=len(val2_dataset),
                                                                      shuffle=True,
                                                                      num_workers=self.args['data']['workers_per_gpu'],
                                                                      drop_last=False, worker_init_fn=init_seed)

            data_loader['cvc-colondb'] = torch.utils.data.DataLoader(dataset=val3_dataset,
                                                                     batch_size=len(val3_dataset),
                                                                     shuffle=True,
                                                                     num_workers=self.args['data']['workers_per_gpu'],
                                                                     drop_last=False, worker_init_fn=init_seed)

            data_loader['etis'] = torch.utils.data.DataLoader(dataset=val4_dataset,
                                                              batch_size=len(val4_dataset),
                                                              shuffle=True,
                                                              num_workers=self.args['data']['workers_per_gpu'],
                                                              drop_last=False, worker_init_fn=init_seed)

            data_loader['kvasir'] = torch.utils.data.DataLoader(dataset=val5_dataset,
                                                                batch_size=len(val5_dataset),
                                                                shuffle=True,
                                                                num_workers=self.args['data']['workers_per_gpu'],
                                                                drop_last=False, worker_init_fn=init_seed)

        return data_loader

    def make_model(self):
        model = Meta_Polypv2(decode_channels=self.args['model']['decode_channels'],
                             pretrained=self.args['model']['pretrained'],
                             num_classes=self.args['model']['num_classes'],
                             window_size=self.args['model']['window_size'])

        model.load_state_dict(torch.load(f=self.args['weight_path']))

        model = model.cuda(self.output_device)

        return model

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.args['print_log']:
            with open('{}/test_log.txt'.format(self.args['work_dir']), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def eval_step(self, model: nn.Module, name_dataset):

        model.eval()
        eval_loader = self.load_data(phase='eval')[name_dataset]
        dice_value = []
        iou_value = []
        for batch_idx, (batch_image, batch_mask) in enumerate(tqdm(eval_loader, ncols=40)):
            with torch.no_grad():
                batch_image = batch_image.float().cuda(self.output_device)
                batch_mask = batch_mask.float().cuda(self.output_device)

                output_mask = model(batch_image)
                output_mask = torch.sigmoid(output_mask)

                self.threshold = self.args['threshold']
                low = torch.tensor(0.).to(self.output_device)
                high = torch.tensor(1.).to(self.output_device)
                output_mask = torch.where(output_mask < self.threshold, low, high)

                Dice_score = Dice_Coeff()(output_mask, batch_mask).float().cpu().detach().numpy()
                IoU_score = IoU()(output_mask, batch_mask).float().cpu().detach().numpy()
                dice_value.append(Dice_score)
                iou_value.append(IoU_score)

        dice_value = np.mean(dice_value) * 100
        iou_value = np.mean(iou_value) * 100

        self.print_log(
            '\tMean test {} dice_score: {:.2f}%.   Mean test {} iou_score: {:.2f}%'.format(name_dataset,
                                                                                           dice_value,
                                                                                           name_dataset,
                                                                                           iou_value))

        return dice_value, iou_value

    def evaluating_multi_dataset(self):

        model = self.make_model()
        list_datasets = ['cvc-300', 'cvc-clinicDB', 'cvc-colondb', 'etis', 'kvasir']

        for name_dataset in list_datasets:
            dice_score, iou_score = self.eval_step(model=model, name_dataset=name_dataset)


def main(args):
    config = args.config
    f = open(config, 'r')
    default_args = yaml.safe_load(f)
    evaluate = Eval_Pipeline(default_args)
    evaluate.evaluating_multi_dataset()

if __name__ == '__main__':
    args = parser_args()
    main(args)