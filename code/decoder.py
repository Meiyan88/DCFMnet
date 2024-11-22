import torch.nn as nn

class ConvDowm(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1):
        super(ConvDowm, self).__init__()
        self.ConvDowm = nn.Sequential(
            nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm3d(out_planes),
            nn.LeakyReLU(negative_slope=0.2)
        )
    def forward(self, x):
        return self.ConvDowm(x)


class ResBlockDown(nn.Module):
    def __init__(self, filters_in, filters_out, stride):
        super(ResBlockDown, self).__init__()

        self.conv1_block = nn.Sequential(
            conv3x3x3(filters_in, filters_out,stride=stride),
            nn.BatchNorm3d(filters_out),
            nn.ReLU(inplace=True)
        )

        self.conv2_block = nn.Sequential(
            conv3x3x3(filters_out, filters_out),
            nn.BatchNorm3d(filters_out),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1_block(x)
        out = self.conv2_block(out)
        out += residual
        out = self.relu(out)

        return out

class Rec_Decoder(nn.Module):
    def __init__(self, output_nc, n_vae_dis, conf=None):
        super(Rec_Decoder, self).__init__()

        self.args = conf
        self.decode_MLP = nn.Sequential(
            nn.Linear(n_vae_dis, 512, False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(512, 256 * 5 * 5 * 6, False),
        )
        self.up_conv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            ResBlockDown(256, 256, 1),
            ConvDowm(256, 128, 1),
        )
        self.up_conv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            ResBlockDown(128, 128, 1),
            ConvDowm(128, 64, 1),
        )

        self.up_conv3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            ResBlockDown(64, 64, 1),
            ConvDowm(64, 32, 1)
        )

        self.up_conv4 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            ResBlockDown(32, 32, 1),
            ConvDowm(32, 16),
        )
        self.finally_layer = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True),
            nn.Conv3d(16, output_nc, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1), bias=False),
            nn.Tanh()
        ).to(self.args.device[1])

    def forward(self, x):
        x = self.decode_MLP(x)
        x = x.view(-1, 256, 5, 5, 6)
        up1 = self.up_conv1(x)
        up2 = self.up_conv2(up1)
        up3 = self.up_conv3(up2)
        up4 = self.up_conv4(up3)
        out = self.finally_layer(up4)
        return out
