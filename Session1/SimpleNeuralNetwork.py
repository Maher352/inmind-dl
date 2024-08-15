import torch
import torch.nn as nn
import torchsummary


class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(4, 6)
        self.fc2 = nn.Linear(6, 8)
        self.fc3 = nn.Linear(8, 4)
        self.fc4 = nn.Linear(4, 4)
        self.fc5 = nn.Linear(4, 2)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.01)  
        self.elu = nn.ELU(alpha=1.0)  
        self.softmax = nn.Softmax(dim=1)  

    def forward(self, x):
        x = self.leaky_relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        x = self.elu(self.fc4(x))
        x = self.softmax(self.fc5(x))
        return x
    
# Initialize the network
model = SimpleNN()

sample_input = torch.randn(16, 4)  # Batch size of 16, 4 features
sample_output = model(sample_input).detach()

# Print the output
print("Model input:")
print(sample_input)
print("Model output:")
print(sample_output)

# Print the summary
torchsummary.summary(model, input_size=(4,))