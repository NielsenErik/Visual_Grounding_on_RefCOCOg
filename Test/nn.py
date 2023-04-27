import torch


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(100, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 10)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return self.linear(x)


tinyModel = TinyModel()

print("The model is: ")
print(tinyModel)

# print("1st layer weights: ")
# print(tinyModel.linear.weight)

# print("\n 2nd layer weights: ")
# print(tinyModel.linear2.weight)

# print("\n Model parameters: ")
# for params in tinyModel.parameters():
#     print(params)

# print("\n Layer parameters: ")
# for params in tinyModel.linear2.parameters():
#     print(params)
