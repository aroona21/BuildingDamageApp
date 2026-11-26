import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )

class SiamUNet(nn.Module):
    def __init__(self, encoder_name="resnet50", num_classes=5, encoder_weights=None):
        super().__init__()
        self.encoder = smp.encoders.get_encoder(encoder_name, in_channels=3, depth=5, weights=encoder_weights)
        channels = self.encoder.out_channels
        self.fuse = nn.ModuleList([nn.Conv2d(3 * channels[i], channels[i], kernel_size=1) for i in range(1, 6)])
        decoder_channels = [256, 128, 64, 32]
        self.decoder = nn.ModuleList()
        previous_channels = channels[5]
        for skip_channels, out_channels in zip([channels[4], channels[3], channels[2], channels[1]], decoder_channels):
            self.decoder.append(conv_block(previous_channels + skip_channels, out_channels))
            previous_channels = out_channels
        self.segmentation_head = nn.Sequential(conv_block(previous_channels, 32), nn.Conv2d(32, num_classes, kernel_size=1))

    def forward(self, pre, post):
        pre_features = self.encoder(pre)
        post_features = self.encoder(post)
        fused = []
        for i in range(1, 6):
            difference = torch.abs(post_features[i] - pre_features[i])
            combined = torch.cat([pre_features[i], post_features[i], difference], dim=1)
            fused.append(self.fuse[i - 1](combined))
        x = fused[4]
        for j, decoder_block in enumerate(self.decoder):
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = decoder_block(torch.cat([x, fused[3 - j]], dim=1))
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return self.segmentation_head(x)
