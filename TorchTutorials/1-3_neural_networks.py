import torch
import torch.nn as nn   #
import torch.nn.functional as F
import torch.optim as optim


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 5)  # 畳み込み層
        self.conv2 = nn.Conv2d(6, 16, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
# print(net)

# You just have to define the forward function,
# and the backward function (where gradients are computed) is automatically defined for you using autograd.
# You can use any of the Tensor operations in the forward function.
#
# The learnable parameters of a model are returned by net.parameters()

params = list(net.parameters())
# print("len(list(net.parameters)) = " + str(len(params)))
# print("list(net.parameters())[0].size is conv1's weight")
# print(params[0].size())  # conv1's .weight

# Let try a random 32x32 input.
# Note: expected input size of this net (LeNet) is 32x32.
# To use this net on MNIST dataset, please resize the images from the dataset to 32x32.

input0 = torch.randn(1, 1, 32, 32)
out = net(input0)
# print("out is " + str(out))

# Zero the gradient buffers of all parameters and backprops with random gradients:
net.zero_grad()
out.backward(torch.randn(1, 10))


# --------- loss function ----------
output = net(input0)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss()

loss = criterion(output, target)
# print("loss is "+str(loss))

# input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
#       -> view -> linear -> relu -> linear -> relu -> linear
#       -> MSELoss
#       -> loss
# print(loss.grad_fn)  # MSELoss
# print(loss.grad_fn.next_functions[0][0])  # Linear
# print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

# --------- back prop ----------

# need to clear the existing gradients though, else gradients will be accumulated to existing gradients.
net.zero_grad()     # zeroes the gradient buffers of all parameters.

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)


# ---------- Update the weights ----------

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
optimizer.step()    # Does the update
