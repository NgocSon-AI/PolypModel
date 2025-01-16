import torch.nn as nn
# from encoder import caformer_s18_in21ft1k
# from decoder import Decoder
from torchsummary import summary
from models.encoder import caformer_s18_in21ft1k
from models.decoder import Decoder


class Meta_Polypv2(nn.Module):
    def __init__(self, decode_channels=64,
                 dropout=0.1,
                 pretrained=True,
                 window_size=8,
                 num_classes=1):
        super().__init__()

        self.backbone = caformer_s18_in21ft1k(pretrained=pretrained)
        encoder_channels = (64, 128, 320, 512)

        self.decoder = Decoder(encoder_channels, decode_channels, dropout, window_size, num_classes)

    def forward(self, x):
        h, w = x.size()[-2:]
        res1, res2, res3, res4 = self.backbone.feature_info(x)
        if self.training:
            x, ah = self.decoder(res1, res2, res3, res4, h, w)
            return x, ah
        else:
            x = self.decoder(res1, res2, res3, res4, h, w)
            return x

# model_segment = Meta_Polypv2()

# encoder = caformer_s18_in21ft1k(pretrained=True)
# from PIL import Image
# from timm.data import create_transform

# transform = create_transform(input_size=256, crop_pct=encoder.default_cfg['crop_pct'])
# image = Image.open('models/dog.jpg')
# input_image = transform(image).unsqueeze(0)
# pred = model_segment(input_image)
# print(pred[0].shape)
