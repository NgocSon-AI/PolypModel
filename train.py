import os.path
import torch.utils.data
from torch.utils.data.dataset import random_split
from utils import *

os.environ['CUDA_LAUNCH_BLOCKING'] = "0"
gc.collect()
torch.cuda.empty_cache()

def parser_args():
    parser = argparse.ArgumentParser(description="Train merged datasets ")
    parser.add_argument('--config',
                        default='/content/drive/MyDrive/PolypModel/configs/train_model.yaml',
                        help='config file path')
    args = parser.parse_args()
    return args

class TrainPipeline:
    def __init__(self, args):
        self.args = args
        self.args['print_log'] = True
        self.args['model_saved_name'] = os.path.join(self.args['work_dir'], 'runs')
        self.output_device = self.args['device']
        self.best_dice = 0
        self.best_iou = 0

    def load_data(self, phase):
        data_loader = dict()
        if phase == 'train':
            train_ratio = 0.8
            test_ratio = 0.2
            dataset = TrainDataset(image_root=self.args['data']['train']['img_root'],
                                         gt_root=self.args['data']['train']['gt_root'],
                                         train_size=self.args['data']['train']['img_size'])
            num_data = len(dataset)
            train_size = int(train_ratio * num_data)
            val_size = int(test_ratio * num_data)
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

            data_loader['train'] = torch.utils.data.DataLoader(dataset=train_dataset,
                                                               batch_size=self.args['data']['train']['batch_size'],
                                                               shuffle=True,
                                                               num_workers=self.args['data']['workers_per_gpu'],
                                                               drop_last=True, worker_init_fn=init_seed)

            data_loader['eval'] = torch.utils.data.DataLoader(dataset=val_dataset,
                                                               batch_size=self.args['data']['train']['batch_size'],
                                                               shuffle=True,
                                                               num_workers=self.args['data']['workers_per_gpu'],
                                                               drop_last=False, worker_init_fn=init_seed)
        return data_loader
    
    def make_model(self):
        model = Meta_Polypv2(decode_channels=self.args['model']['decode_channels'],
                             pretrained=self.args['model']['pretrained'],
                             num_classes=self.args['model']['num_classes'],
                             window_size=self.args['model']['window_size'])
        model = model.cuda(self.output_device)
        return model

    def make_loss(self):
        loss = MetaPolypv2_Loss()
        return loss

    def make_optimizer(self, model):
        optimizer = Optimizer(self.args, model).get_optim()
        return optimizer

    def make_scheduler(self, optimizer):
        scheduler = Scheduler(cfgs=self.args, optimizer=optimizer).get_scheduler()
        return scheduler

    def make_work_dirs_path(self):
        if os.path.isdir(self.args['model_saved_name']):
            print('log_dir: ', self.args['model_saved_name'], 'already exist')
            answer = input('delete it? y/n:')
            if answer == 'y':
                shutil.rmtree(self.args['model_saved_name'])
                print('Dir removed: ', self.args['model_saved_name'])
                input('Refresh the website of tensorboard by pressing any keys')
            else:
                print('Dir not removed: ', self.args['model_saved_name'])
        self.train_writer = SummaryWriter(os.path.join(self.args['model_saved_name'], 'train'), 'train')
        self.val_writer = SummaryWriter(os.path.join(self.args['model_saved_name'], 'val'), 'val')
    
    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.args['print_log']:
            with open('{}/log.txt'.format(self.args['work_dir']), 'a') as f:
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

    def train_step(self, model: nn.Module, optimizer: optim.Optimizer, epoch, scheduler, save_model=False):
        model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        train_loader = self.load_data(phase='train')['train']

        self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)

        loss = self.make_loss()
        loss_per_batch = 0
        dice_value = []
        iou_value = []
        for batch_idx, (batch_image, batch_mask) in enumerate(tqdm(train_loader, ncols=40)):
            self.global_step += 1
            with torch.no_grad():
                batch_image = batch_image.float().cuda(self.output_device)
                batch_mask = batch_mask.float().cuda(self.output_device)
            timer['dataloader'] += self.split_time()

            # forward path
            output_mask = model(batch_image)
            output_mask = list(output_mask)
            output_mask[0] = torch.sigmoid(output_mask[0])
            output_mask[1] = torch.sigmoid(output_mask[1])
            loss_batch = loss(output_mask, batch_mask)
            optimizer.zero_grad()
            clip_gradient(optimizer=optimizer, grad_clip=0.5)
            loss_batch.backward()
            optimizer.step()
            loss_per_batch += loss_batch.data.item()

            timer['model'] += self.split_time()

            self.threshold = self.args['threshold']
            low = torch.tensor(0.).to(self.output_device)
            high = torch.tensor(1.).to(self.output_device)
            output_mask[0] = torch.where(output_mask[0] < self.threshold, low, high)

            Dice_score = Dice_Coeff()(output_mask[0], batch_mask).float().cpu().detach().numpy()
            IoU_score = IoU()(output_mask[0], batch_mask).float().cpu().detach().numpy()
            dice_value.append(Dice_score)
            iou_value.append(IoU_score)

            timer['statistics'] += self.split_time()

        loss_per_batch /= len(train_loader.dataset)
        dice_value = np.mean(dice_value) * 100
        iou_value = np.mean(iou_value) * 100

        self.train_writer.add_scalar('dice_score', dice_value, self.global_step)
        self.train_writer.add_scalar('IoU_score', iou_value, self.global_step)
        self.train_writer.add_scalar('loss_train', loss_per_batch, self.global_step)

        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(
            '\tMean training loss: {:.4f}.  Mean dice_score: {:.2f}%.   Mean iou_score: {:.2f}%'.format(loss_per_batch,
                                                                                                        dice_value,
                                                                                                        iou_value))
        self.print_log('\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(**proportion))

        if save_model:
          if dice_value > 92. :
              state_dict = model.state_dict()
              weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
              torch.save(weights, self.args['model_saved_name'] + '-' + str(epoch + 1) + '-' + str(int(self.global_step)) + '.pt')

        return loss_per_batch, dice_value, iou_value
    
    def eval_step(self, model: nn.Module, epoch):
        model.eval()
        self.print_log('Eval epoch: {}'.format(epoch + 1))

        eval_loader = self.load_data(phase='train')['eval']
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

        if dice_value > self.best_dice:
            self.best_dice = dice_value
            self.best_dice_epoch = epoch + 1

        if iou_value > self.best_iou:
            self.best_iou = iou_value
            self.best_iou = epoch + 1

        self.print_log(
            '\tMean test dice_score: {:.2f}%.   Mean test iou_score: {:.2f}%'.format(dice_value, iou_value))
        self.val_writer.add_scalar('dice', dice_value, self.global_step)
        self.val_writer.add_scalar('iou', iou_value, self.global_step)

        return dice_value, iou_value
    
    def epoch_step(self):
        self.make_work_dirs_path()
        model = self.make_model()
        self.args['start_epoch'] = 0
        self.global_step = self.args['start_epoch'] * len(self.load_data(phase='train')['train']) / self.args['data'][
            'train']['batch_size']

        def count_parameters(train_model):
            return sum(p.numel() for p in train_model.parameters() if p.requires_grad)

        optimizer = self.make_optimizer(model=model)
        scheduler = self.make_scheduler(optimizer=optimizer)
        self.print_log(f'# Parameters: {count_parameters(model)}')
        for epoch in range(self.args['start_epoch'], self.args['num_epochs']):
            save_model = True
            train_loss, dice_score, iou_score = self.train_step(model=model, optimizer=optimizer, epoch=epoch,
                                                                scheduler=scheduler, save_model=save_model)
            val_dice_score, val_iou_score = self.eval_step(model=model, epoch=epoch)

            scheduler.step()

            save_metrics = {
                "Train/Train loss": train_loss,
                "Train/dice_score": dice_score,
                "Train/iou_score": iou_score
            }

            val_metric = {
                'Eval/ dice_score' : val_dice_score,
                'Eval/ iou_score' : val_iou_score
            }

def main(args):
    config = args.config
    f = open(config, 'r')
    default_args = yaml.safe_load(f)
    train = TrainPipeline(default_args)
    train.epoch_step()


if __name__ == '__main__':
    args = parser_args()
    main(args)