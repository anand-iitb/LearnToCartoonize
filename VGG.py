import torch
import torch.nn as nn

# Defining the architecture of different VGG models
# "M" is the maxpooling layer and Constatnt is the convolution layer
# For maxpooling layer, the kernel size is 2 and stride is 2
# For convolution layer, the kernel size is 3, stride is 1 and padding is 1

VGG11 = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]
VGG13 = [64, 64, "M", 128, 128, "M", 256,
         256, "M", 512, 512, "M", 512, 512, "M"]
VGG16 = [64, 64, "M", 128, 128, "M", 256, 256, 256,
         "M", 512, 512, 512, "M", 512, 512, 512, "M"]
VGG19 = [64, 64, "M", 128, 128, "M", 256, 256, 256, 256,
         "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"]


class VGG_Network(nn.Module):
    def __init__(self, architecture, input_channels=3, input_size=224, num_hidden_units=4096, num_classes=1000):
        super(VGG_Network, self).__init__()

        self.input_channels = input_channels

        # The input size is 224 meaning that the input image is 224x224
        self.input_size = input_size

        self.num_hidden_units = num_hidden_units

        self.num_classes = num_classes

        self.conv_layers = self.create_conv_layers(architecture)

        last_output_size = self.input_size // (2 ** architecture.count("M"))

        for x in reversed(architecture):
            if x != "M":
                last_output_channels = x
                break

        self.fcs = nn.Sequential(
            nn.Linear(last_output_channels * last_output_size *
                      last_output_size, num_hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(num_hidden_units, num_hidden_units),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(num_hidden_units, num_classes)
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.reshape(x.size(0), -1)
        x = self.fcs(x)
        return x

    def create_conv_layers(self, vgg_architecture):
        conv_layers = []
        inpt = self.input_channels
        for item in vgg_architecture:
            if item == 'M':
                conv_layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = nn.Conv2d(inpt, item,
                                   kernel_size=3, stride=1, padding=1)
                conv_layers += [conv2d, nn.BatchNorm2d(item), nn.ReLU()]
                inpt = item
        return nn.Sequential(*conv_layers)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Working on {device}")
model = VGG_Network(VGG16, 3, 224, 1000).to(device)
X = torch.randn(1, 3, 224, 224).to(device)
print(model(X))
