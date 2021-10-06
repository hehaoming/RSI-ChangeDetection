from typing import List
import torch
import torch.nn as nn
from torch import Tensor
from utils import kaiming_init, constant_init, xavier_init



class ConvBlock(nn.Module):
    """将Conv2d、BatchNorm2d、ReLU打包成一个模块"""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, mode: str = "None") -> None:
        super(ConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.mode = mode
        self.conv1 = nn.Conv2d(self.in_channels,
                               self.out_channels,
                               kernel_size=self.kernel_size,
                               padding=1,
                               stride=1,
                               bias=True)

        self.selu1 = nn.SELU(inplace=True)
        self.bn1 = nn.BatchNorm2d(self.out_channels)

        self.conv2 = nn.Conv2d(self.out_channels,
                               self.out_channels,
                               kernel_size=self.kernel_size,
                               padding=1,
                               stride=1,
                               bias=True)
        self.selu2 = nn.SELU(inplace=True)
        self.bn2 = nn.BatchNorm2d(self.out_channels)

    def forward(self, x: Tensor) -> Tensor:
        # Conv => SELU => BN
        x = self.selu1(self.conv1(x))
        x0 = x
        x = self.bn1(x)
        x = self.selu2(self.conv2(x))
        x = self.bn2(x)
        if self.mode == 'residual':
            x = x + x0
        return x


class NestedUNet(nn.Module):
    """UNET++的变种版本模型，使用了MSOF策略优化了输出"""

    def __init__(self,
                 in_channels: int = 6,
                 out_channels: int = 3,
                 nb_filter: List = None,
                 deep_supervision: bool = False,
                 msof: bool = True,
                 activation: str = 'sigmoid',
                 **params) -> None:
        super(NestedUNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.deep_supervision = deep_supervision
        self.nb_filter = nb_filter
        self.mode = 'residual'
        self.activation = activation
        self.msof = msof
        if self.nb_filter is None:
            self.nb_filter = [32, 64, 128, 256, 512]
        self.conv1_1 = ConvBlock(self.in_channels, self.nb_filter[0])
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)

        self.conv2_1 = ConvBlock(self.nb_filter[0], self.nb_filter[1], mode=self.mode)
        self.pool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=0)

        self.up1_2 = nn.ConvTranspose2d(self.nb_filter[1], self.nb_filter[0], kernel_size=(2, 2), stride=(2, 2))
        self.conv1_2 = ConvBlock(2 * self.nb_filter[0], self.nb_filter[0], mode=self.mode)

        self.conv3_1 = ConvBlock(self.nb_filter[1], self.nb_filter[2], mode=self.mode)
        self.pool3 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.up2_2 = nn.ConvTranspose2d(self.nb_filter[2], self.nb_filter[1], kernel_size=(2, 2), stride=(2, 2))
        self.conv2_2 = ConvBlock(self.nb_filter[1] * 2, self.nb_filter[1], mode=self.mode)

        self.up1_3 = nn.ConvTranspose2d(self.nb_filter[1], self.nb_filter[0], kernel_size=(2, 2), stride=(2, 2))
        self.conv1_3 = ConvBlock(self.nb_filter[0] * 3, self.nb_filter[0], mode=self.mode)

        self.conv4_1 = ConvBlock(self.nb_filter[2], self.nb_filter[3], mode=self.mode)
        self.pool4 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.up3_2 = nn.ConvTranspose2d(self.nb_filter[3], self.nb_filter[2], kernel_size=(2, 2), stride=(2, 2))
        self.conv3_2 = ConvBlock(self.nb_filter[2] * 2, self.nb_filter[2], mode=self.mode)

        self.up2_3 = nn.ConvTranspose2d(self.nb_filter[2], self.nb_filter[1], kernel_size=(2, 2), stride=(2, 2))
        self.conv2_3 = ConvBlock(self.nb_filter[1] * 3, self.nb_filter[1], mode=self.mode)

        self.up1_4 = nn.ConvTranspose2d(self.nb_filter[1], self.nb_filter[0], kernel_size=(2, 2), stride=(2, 2))
        self.conv1_4 = ConvBlock(self.nb_filter[0] * 4, self.nb_filter[0], mode=self.mode)

        self.conv5_1 = ConvBlock(self.nb_filter[3], self.nb_filter[4], mode=self.mode)

        self.up4_2 = nn.ConvTranspose2d(self.nb_filter[4], self.nb_filter[3], kernel_size=(2, 2), stride=(2, 2))
        self.conv4_2 = ConvBlock(self.nb_filter[3] * 2, self.nb_filter[3], mode=self.mode)

        self.up3_3 = nn.ConvTranspose2d(self.nb_filter[3], self.nb_filter[2], kernel_size=(2, 2), stride=(2, 2))
        self.conv3_3 = ConvBlock(self.nb_filter[2] * 3, self.nb_filter[2], mode=self.mode)

        self.up2_4 = nn.ConvTranspose2d(self.nb_filter[2], self.nb_filter[1], kernel_size=(2, 2), stride=(2, 2))
        self.conv2_4 = ConvBlock(self.nb_filter[1] * 4, self.nb_filter[1], mode=self.mode)

        self.up1_5 = nn.ConvTranspose2d(self.nb_filter[1], self.nb_filter[0], kernel_size=(2, 2), stride=(2, 2))
        self.conv1_5 = ConvBlock(self.nb_filter[0] * 5, self.nb_filter[0], mode=self.mode)

        self.nestnet_output_1 = nn.Conv2d(self.nb_filter[0], self.out_channels, kernel_size=1, stride=1, padding=0)
        self.nestnet_output_2 = nn.Conv2d(self.nb_filter[0], self.out_channels, kernel_size=1, stride=1, padding=0)
        self.nestnet_output_3 = nn.Conv2d(self.nb_filter[0], self.out_channels, kernel_size=1, stride=1, padding=0)
        self.nestnet_output_4 = nn.Conv2d(self.nb_filter[0], self.out_channels, kernel_size=1, stride=1, padding=0)
        if self.msof:
            self.nestnet_output_5 = nn.Conv2d(self.nb_filter[0] * 4, self.out_channels, kernel_size=1, stride=1, padding=0)

        # 初始化网络参数
        self.init_weights()

    def forward(self, x1: Tensor, x2: Tensor) -> List[Tensor]: # self, x1: Tensor, x2: Tensor
        x = torch.cat((x1, x2), dim=1)                                                # 1, 6, 512, 512
        conv1_1 = self.conv1_1(x)                                                     # 1, 32, 512, 512
        pool1 = self.pool1(conv1_1)                                                   # 1, 32, 256, 256

        conv2_1 = self.conv2_1(pool1)                                                 # 1, 64, 256, 256
        pool2 = self.pool2(conv2_1)                                                   # 1, 64, 128, 128

        up1_2 = self.up1_2(conv2_1)                                                   # 1, 32, 512, 512
        conv1_2 = torch.cat((up1_2, conv1_1), dim=1)                                  # 1, 64, 512, 512
        conv1_2 = self.conv1_2(conv1_2)                                               # 1, 32, 512, 512

        conv3_1 = self.conv3_1(pool2)                                                 # 1, 128, 128, 128
        pool3 = self.pool3(conv3_1)                                                   # 1, 128, 64, 64

        up2_2 = self.up2_2(conv3_1)                                                   # 1, 64, 256, 256
        conv2_2 = torch.cat((up2_2, conv2_1), dim=1)                                  # 1, 128, 256, 256
        conv2_2 = self.conv2_2(conv2_2)                                               # 1, 64, 256, 256

        up1_3 = self.up1_3(conv2_2)  # 1, 32, 512, 512
        conv1_3 = torch.cat((up1_3, conv1_1, conv1_2), dim=1)  # 1, 96, 512, 512
        conv1_3 = self.conv1_3(conv1_3)  # 1, 32, 512, 512

        conv4_1 = self.conv4_1(pool3)  # 1, 256, 64, 64
        pool4 = self.pool4(conv4_1)  # 1, 256, 32, 32

        up3_2 = self.up3_2(conv4_1)  # 1, 128, 128, 128
        conv3_2 = torch.cat((up3_2, conv3_1), dim=1)  # 1, 256, 128, 128
        conv3_2 = self.conv3_2(conv3_2)  # 1, 128, 128, 128

        up2_3 = self.up2_3(conv3_2)  # 1, 64, 256, 256
        conv2_3 = torch.cat((up2_3, conv2_1, conv2_2), dim=1)  # 1, 192, 256, 256
        conv2_3 = self.conv2_3(conv2_3)  # 1, 64, 256, 256

        up1_4 = self.up1_4(conv2_3)  # 1, 32, 512, 512
        conv1_4 = torch.cat((up1_4, conv1_1, conv1_2, conv1_3), dim=1)  # 1, 128, 512, 512
        conv1_4 = self.conv1_4(conv1_4)  # 1, 32, 512, 512

        conv5_1 = self.conv5_1(pool4)  # 1, 512, 32, 32

        up4_2 = self.up4_2(conv5_1)  # 1, 256, 64, 64
        conv4_2 = torch.cat((up4_2, conv4_1), dim=1)  # 1, 512, 32, 32
        conv4_2 = self.conv4_2(conv4_2)  # 1, 512, 64, 64

        up3_3 = self.up3_3(conv4_2)  # 1, 128, 128, 128
        conv3_3 = torch.cat((up3_3, conv3_1, conv3_2), dim=1)  # 1, 384, 128, 128
        conv3_3 = self.conv3_3(conv3_3)  # 1, 128, 128, 128

        up2_4 = self.up2_4(conv3_3)  # 1, 64, 256, 256
        conv2_4 = torch.cat((up2_4, conv2_1, conv2_2, conv2_3), dim=1)  # 1, 256, 128, 128
        conv2_4 = self.conv2_4(conv2_4)  # 1, 256, 256, 256

        up1_5 = self.up1_5(conv2_4)  # 1, 32, 512, 512
        conv1_5 = torch.cat((up1_5, conv1_1, conv1_2, conv1_3, conv1_4), dim=1)  # 1, 160, 512, 512
        conv1_5 = self.conv1_5(conv1_5)  # 1, 32, 512, 512

        nestnet_output_1 = self.nestnet_output_1(conv1_2)
        nestnet_output_2 = self.nestnet_output_2(conv1_3)
        nestnet_output_3 = self.nestnet_output_3(conv1_4)
        nestnet_output_4 = self.nestnet_output_4(conv1_5)

        output = [nestnet_output_1, nestnet_output_2, nestnet_output_3, nestnet_output_4]
        if self.msof:
            conv_fuse = torch.cat((conv1_2, conv1_3, conv1_4, conv1_5), dim=1)
            nestnet_output_5 = self.nestnet_output_5(conv_fuse)
            output.append(nestnet_output_5)
        if self.activation is not None:
            if self.activation == 'sigmoid':
                output = list(map(lambda t: torch.squeeze(torch.sigmoid(t)), output))

        return output

    def init_weights(self) -> None:
        """Initialize the weights in Network."""

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                constant_init(m, 1)
            elif isinstance(m, nn.ConvTranspose2d):
                xavier_init(m, distribution='uniform')


# if __name__ == "__main__":
#     from torchvision.io import read_image
#     model = NestedUNet()
#     image1 = read_image(r"D:\datasets\trainData\images\2_1.png")
#     image1 = image1.float()
#     image1 /= 255.
#     image1 = torch.unsqueeze(image1, dim=0)
#     image2 = read_image(r"D:\datasets\trainData\images\2_2.png")
#     image2 = image2.float()
#     image2 /= 255.
#     image2 = torch.unsqueeze(image2, dim=0)
#     output = model(image1, image2)
#     print(type(output))
#     print(len(output))
#     print(output[4].shape)