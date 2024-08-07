import torch
import torch.nn as nn
import torchsummary

class ConvLayering(nn.Module):
  def __init__(self, in_channels, num_filters, kernel_size, stride=1, padding=0):
    super(ConvLayering, self).__init__()

    self.conv = nn.Conv2d(in_channels, num_filters, kernel_size = kernel_size, stride = stride, padding = padding)
    self.batchnorm = nn.BatchNorm2d(num_filters)
    self.leaky_relu = nn.LeakyReLU(0.01)

  def forward(self, x):
    x = self.conv(x)
    x = self.batchnorm(x)
    x = self.leaky_relu(x)
    return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        #first layer
        self.convLayer1 = ConvLayering(3, 64, 7, 2, 3)
        #maxPool1

        #second layer
        self.convLayer2 = ConvLayering(64, 192, 3, 1, 1)
        #maxPool2

        #third layer
        self.convLayer3 = ConvLayering(192, 128, 1)
        self.convLayer4 = ConvLayering(128, 256, 3, 1, 1)
        self.convLayer5 = ConvLayering(256, 256, 1)
        self.convLayer6 = ConvLayering(256, 512, 3, 1, 1)
        #maxPool3

        #fourth layer
        #4x
        self.convLayer7 = ConvLayering(512, 256, 1)
        self.convLayer8 = ConvLayering(256, 512, 3, 1, 1)
        self.convLayer9 = ConvLayering(512, 256, 1)
        self.convLayer10 = ConvLayering(256, 512, 3, 1, 1)
        self.convLayer11 = ConvLayering(512, 256, 1)
        self.convLayer12 = ConvLayering(256, 512, 3, 1, 1)
        self.convLayer13 = ConvLayering(512, 256, 1)
        self.convLayer14 = ConvLayering(256, 512, 3, 1, 1)

        self.convLayer15 = ConvLayering(512, 512, 1)
        self.convLayer16 = ConvLayering(512, 1024, 3, 1, 1)
        #maxPool4

        #fifth layer
        #2x
        self.convLayer17 = ConvLayering(1024, 512, 1)
        self.convLayer18 = ConvLayering(512, 1024, 3, 1, 1)
        self.convLayer19 = ConvLayering(1024, 512, 1)
        self.convLayer20 = ConvLayering(512, 1024, 3, 1, 1)

        self.convLayer21 = ConvLayering(1024, 1024, 3, 1, 1)
        self.convLayer22 = ConvLayering(1024, 1024, 3, 2, 1)
        #no maxPool

        #sixth layer
        self.convLayer23 = ConvLayering(1024, 1024, 3, 1, 1)
        self.convLayer24 = ConvLayering(1024, 1024, 3, 1, 1)
        #no maxPool

        #seventh layer
        self.fc1 = nn.Linear(50176, 4096)

        #eighth layer
        self.fc2 = nn.Linear(4096, 1470)


        #max pooling
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #activation functions
        self.leaky_relu = nn.LeakyReLU(0.01)

        #flatten
        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.convLayer1(x)
        x = self.pool(x)
        #print("layer1 complete")

        x = self.convLayer2(x)
        x = self.pool(x)
        #print("layer2 complete")

        x = self.convLayer3(x)
        x = self.convLayer4(x)
        x = self.convLayer5(x)
        x = self.convLayer6(x)
        x = self.pool(x)
        #print("layer3 complete")

        x = self.convLayer7(x)
        x = self.convLayer8(x)
        x = self.convLayer9(x)
        x = self.convLayer10(x)
        x = self.convLayer11(x)
        x = self.convLayer12(x)
        x = self.convLayer13(x)
        x = self.convLayer14(x)
        x = self.convLayer15(x)
        x = self.convLayer16(x)
        x = self.pool(x)
        #print("layer4 complete")

        x = self.convLayer17(x)
        x = self.convLayer18(x)
        x = self.convLayer19(x)
        x = self.convLayer20(x)
        x = self.convLayer21(x)
        x = self.convLayer22(x)
        #print("layer5 complete")

        x = self.convLayer23(x)
        x = self.convLayer24(x)
        x = self.flatten(x)
        #print(x.shape)
        #print("layer6 complete")

        x = self.leaky_relu(self.fc1(x))
        #print("layer7 complete")

        x = self.fc2(x)
        #print("layer8 complete")

        return x
    

# Initialize the YoloV1 network
model = CNN()

sample_input = torch.randint(0, 2, (4, 3, 448, 448), dtype=torch.float32) # (batches, channels, height, width)
sample_output = model(sample_input).detach()

# Print the output
print("Model input:")
print(sample_input.shape, "\n", sample_input)
print("Model output:")
print(sample_output.shape, "\n", sample_output)

# Print the summary
torchsummary.summary(model, input_size=(3, 448, 448))  # (channels, height, width)