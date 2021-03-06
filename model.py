import torch 
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms

class ConvNet(nn.Module):
    def __init__(self, in_channel = 3, activ = nn.functional.relu, num_classes=10):
        super(ConvNet, self).__init__()
        self.activ = activ
        self.in_channel = in_channel
        self.num_classes = num_classes

        self.layer_sizes = [64, 64, 128, 128, 128, 256, 256, 256, 512, 512, 512]
        self.feature_sizes = [32, 32, 16, 16, 16, 8, 8, 8, 4, 4, 4]
        strides = [1, 1, 2, 1, 1, 2, 1, 1, 2, 1, 1]
        # here we have to assume for all layers, padding = 1, otherwise not trainable
        self.conv_layers = [
            torch.nn.Conv2d(in_channel, self.layer_sizes[0], kernel_size=3, stride=strides[0], # specified in the appendix
            padding=1, dilation=1, groups=1, bias=True), # not specififed params here
            torch.nn.BatchNorm2d(self.layer_sizes[0]),
        ]
        for i in range(1, len(self.layer_sizes)):
            self.conv_layers.append(
                torch.nn.Conv2d(self.layer_sizes[i-1], self.layer_sizes[i], kernel_size=3, stride=strides[i], padding = 1)
            )
            self.conv_layers.append(
                torch.nn.BatchNorm2d(self.layer_sizes[i]),
            )

        self.conv_layers = nn.Sequential(*self.conv_layers)
        self.fc = nn.Sequential(
            nn.Linear(512 * 4 * 4, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 1024),
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

    def init_mus(self, device):
        self.mus = [torch.zeros((self.num_classes, self.layer_sizes[0])).to(device)]  # num_classes * layer_size
        for i in range(1, len(self.layer_sizes)):
            self.mus.append(torch.zeros((self.num_classes, self.layer_sizes[i])).to(device))
        

    def forward(self, x):
        for i, layer in enumerate(self.conv_layers):
            x = layer(x)
            if i % 2: # after BN
                x = self.activ(x)

        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)

        return x

    def clamp_forward(self, x, layer_idx, neuron_idx, device):  # clamp jth neuron on ith layer
        for i, layer in enumerate(self.conv_layers):
            x = layer(x)
            if i % 2: # after BN
                x = self.activ(x)
                if (i // 2) == layer_idx:
                    for single in x:
                        single[neuron_idx] = torch.zeros((self.feature_sizes[layer_idx], self.feature_sizes[layer_idx])).to(device)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)

        return x

    def forward_and_record(self, x, labels,device):
        print("foward_and_record, layer", end=" ")
        for i, layer in enumerate(self.conv_layers):
            print(i, end=" ")
            x = layer(x)
            if i % 2: # after BN
                x = self.activ(x)
                avg_x = torch.mean(x, (2, 3)).to(device)  # bs * layer_size, activity of feature map is averaged across all elemnts
                for j, single in enumerate(avg_x):
                    self.mus[i // 2][labels[j]] += single

        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        print("done")
        return x

    def compute_selectivity(self, testloader, device):
        num_images = 0
        self.init_mus(device)
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                num_images += images.shape[0]
                _ = self.forward_and_record(images, labels, device)
        result = []
        for mu in self.mus:
            t_mu = torch.transpose(mu, 0, 1)
            mu_max, _ = torch.max(t_mu, 1)
            mu_sum = torch.sum(t_mu, 1)
            mu_mmax = (mu_sum - mu_max) / (self.num_classes - 1)
            mu_max /= num_images
            mu_mmax /= num_images
            result.append(torch.div(mu_max-mu_mmax, mu_max+mu_mmax).cpu().numpy())
        return result