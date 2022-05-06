import torch 
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms

class ConvNet(nn.Module):
    def __init__(self, in_channel = 3, activ = nn.functional.relu):
        super(ConvNet, self).__init__()
        self.activ = activ
        self.in_channel = in_channel

        layer_sizes = [64, 64, 128, 128, 128, 256, 256, 256, 512, 512, 512]
        strides = [1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1]
        # here we have to assume for all layers, padding = 1, otherwise not trainable
        self.conv_layers = [
            torch.nn.Conv2d(in_channel, layer_sizes[0], kernel_size=3, stride=strides[0], # specified in the appendix
            padding=1, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None), # not specififed params here
        ]
        for i in range(1, len(layer_sizes)):
            self.conv_layers.append(
                torch.nn.Conv2d(layer_sizes[i-1], layer_sizes[i], kernel_size=3, stride=strides[i], padding = 1)
            )

        self.conv_layers = nn.Sequential(*self.conv_layers)
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 10),
        )


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

        print("---network architecture---")
        print(self.conv_layers)
        print("--------------------------")
        

    def forward(self, x):
        for layer in self.conv_layers:
            x = layer(x)
            x = self.activ(x)

        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)

        return x