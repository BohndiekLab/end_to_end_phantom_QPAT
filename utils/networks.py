import torch.nn as nn
from torch import cat


def convolve(in_channels, out_channels, kernel_size=(3, 3)):
    layer = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
        nn.LeakyReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size, padding=1),
        nn.LeakyReLU(inplace=True)
    )
    return layer


def contract(in_channels, out_channels, kernel_size=(3, 3)):
    layer = nn.Sequential(
        nn.Conv2d(out_channels, out_channels, kernel_size, padding=1, stride=(2, 2)),
        nn.LeakyReLU(inplace=True)
    )
    return layer


def expand(in_channels, out_channels, kernel_size=(3, 3)):
    layer = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
        nn.LeakyReLU(inplace=True),
        )
    return layer


class RegressionUNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=1, initial_filter_size=64, kernel_size=3):
        super().__init__()

        self.contr_1_1 = convolve(in_channels, initial_filter_size, kernel_size)
        self.contr_1_2 = contract(initial_filter_size, initial_filter_size, kernel_size)
        self.contr_2_1 = convolve(initial_filter_size, initial_filter_size * 2, kernel_size)
        self.contr_2_2 = contract(initial_filter_size * 2, initial_filter_size * 2, kernel_size)
        self.contr_3_1 = convolve(initial_filter_size * 2, initial_filter_size * 2 ** 2, kernel_size)
        self.contr_3_2 = contract(initial_filter_size * 2 ** 2, initial_filter_size * 2 ** 2, kernel_size)
        self.contr_4_1 = convolve(initial_filter_size * 2 ** 2, initial_filter_size * 2 ** 3, kernel_size)
        self.contr_4_2 = contract(initial_filter_size * 2 ** 3, initial_filter_size * 2 ** 3, kernel_size)

        self.center = nn.Sequential(
            nn.Conv2d(initial_filter_size*2**3, initial_filter_size*2**4, (3, 3), padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(initial_filter_size*2**4, initial_filter_size*2**4, (3, 3), padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(initial_filter_size*2**4, initial_filter_size*2**3, (2, 2), stride=(2, 2)),
            nn.LeakyReLU(inplace=True),
        )

        self.expand_4_1 = expand(initial_filter_size*2**4, initial_filter_size*2**3)
        self.expand_4_2 = expand(initial_filter_size*2**3, initial_filter_size*2**3)
        self.upscale4 = nn.ConvTranspose2d(initial_filter_size*2**3, initial_filter_size*2**2, (2, 2), stride=(2, 2))

        self.expand_3_1 = expand(initial_filter_size*2**3, initial_filter_size*2**2)
        self.expand_3_2 = expand(initial_filter_size*2**2, initial_filter_size*2**2)
        self.upscale3 = nn.ConvTranspose2d(initial_filter_size*2**2, initial_filter_size*2, (2, 2), stride=(2, 2))

        self.expand_2_1 = expand(initial_filter_size*2**2, initial_filter_size*2)
        self.expand_2_2 = expand(initial_filter_size*2, initial_filter_size*2)
        self.upscale2 = nn.ConvTranspose2d(initial_filter_size*2, initial_filter_size, (2, 2), stride=(2, 2))

        self.expand_1_1 = expand(initial_filter_size*2, initial_filter_size)
        self.expand_1_2 = expand(initial_filter_size, initial_filter_size)

        self.final = nn.Conv2d(initial_filter_size, out_channels, kernel_size=(1, 1))

    def forward(self, x):
        contr_1_1 = self.contr_1_1(x)
        contr_1_2 = self.contr_1_2(contr_1_1)
        contr_2_1 = self.contr_2_1(contr_1_2)
        contr_2_2 = self.contr_2_2(contr_2_1)
        contr_3_1 = self.contr_3_1(contr_2_2)
        contr_3_2 = self.contr_3_2(contr_3_1)
        contr_4_1 = self.contr_4_1(contr_3_2)
        contr_4_2 = self.contr_4_2(contr_4_1)

        center = self.center(contr_4_2)

        concat = cat([center, contr_4_1], 1)

        expand = self.expand_4_2(self.expand_4_1(concat))
        upscale = self.upscale4(expand)

        concat = cat([upscale, contr_3_1], 1)
        expand = self.expand_3_2(self.expand_3_1(concat))
        upscale = self.upscale3(expand)

        concat = cat([upscale, contr_2_1], 1)
        expand = self.expand_2_2(self.expand_2_1(concat))
        upscale = self.upscale2(expand)

        concat = cat([upscale, contr_1_1], 1)
        expand = self.expand_1_2(self.expand_1_1(concat))
        output = self.final(expand)

        return output
