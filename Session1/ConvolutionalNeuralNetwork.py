import torch
import torch.nn as nn
import torchsummary

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)

        self.flatten = nn.Flatten()

    def forward(self, x):
        x = self.tanh(self.conv1(x))
        x = self.pool(x)
        x = self.tanh(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        x = self.tanh(self.fc1(x))
        x = self.tanh(self.fc2(x))
        x = self.fc3(x)
        return self.softmax(x)


# Initialize the network
model = CNN()

sample_input = torch.randint(0, 10, (4, 1, 32, 32), dtype=torch.float32) # (batches, channels, height, width)
sample_output = model(sample_input).detach()

# Print the output
print("Model input:")
print(sample_input)
print("Model output:")
print(sample_output)

# Print the summary
torchsummary.summary(model, input_size=(1, 32, 32))  # (channels, height, width)