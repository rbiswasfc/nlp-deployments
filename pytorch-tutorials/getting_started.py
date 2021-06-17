#%%
import torch
import numpy as np

#%%
x = [[1, 2, 5], [3, 4, 0]]
x = torch.Tensor(x)
print(x)
print(type(x))
# tensors in pytorch are objects created from
# torch.Tensor
# %%
print(x.device)
print(x.layout)
print(x.dtype)
# tensor contain uniform numerical data
# each data type has cpu and gpu implementation
# tensor operations between tensors must happen
# between tensors that exist in the same device
# tensor computations between tensors depend on both type and device

# %%
# creating tensors from existing data
t = np.array([1, 2, 3])
t1 = torch.Tensor(t)  # class constructor
print(t1)
print(t1.dtype)
print(type(t1))
# %%
t2 = torch.tensor(t)  # factory function
# allows for more dynamic object creation
print(t2)
print(t2.dtype)
print(type(t2))
# %%
t3 = torch.as_tensor(t)
print(t3)
print(t3.dtype)
print(type(t3))
# %%
t4 = torch.from_numpy(t)
print(t4)
print(t4.dtype)
print(type(t4))


# %%
n1 = torch.eye(2)
n1
# %%
n2 = torch.rand((2, 2))
n2




# %%
n3 = torch.zeros(2)
n3

# %%
n4 = torch.ones(2)
n4
# %%
torch.get_default_dtype()




# %%
t = np.array([3, 6, 9])
t1 = torch.Tensor(t)  # copy data
t2 = torch.tensor(t)  # copy data
t3 = torch.as_tensor(t, dtype=torch.float64)  # copy data
t4 = torch.from_numpy(t)  # share data
print(t1)
print(t2)
print(t3)
print(t4)
# %%
t[0] = -1





# %%
print(t1)
print(t2)
print(t3)
print(t4)
# effect of memory mapping







# %%
t = torch.tensor([[1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]], dtype=torch.float32)
print(t.numel())
print((torch.tensor(t.shape)).prod().item())
# tensor operations
# reshaping
t1 = t.reshape((6, 2))
print(t1)
# %%
t1[0][0] = -1
print(t)
print(t1)
# reshaping is memory sharing operation
# %%
t2 = t.reshape((1,1,1,12))
print(t2.shape)
# %%
t2 = t2.squeeze()
print(t2.shape)
# %%
t2 = t2.unsqueeze(axis=0).unsqueeze(axis=-1)
t2.shape
# %%
t1 = torch.rand((2,4,3))
t2 = torch.rand((2,5,3))
t3 = torch.rand((2,6,3))
t = torch.cat((t1, t2, t3), dim=1)
print(t.shape)
# %%
# cat vs stack in pytorch
# stack concatenates tensors using a new dimension
# cat concatenates tensors along a specified dimension
t1 = torch.rand((2,4,3))
t2 = torch.rand((2,4,3))
t3 = torch.rand((2,4,3))
t_cat = torch.cat((t1, t2, t3), dim=1)
t_stack = torch.stack((t1, t2, t3), dim=-1)

print(t_cat.shape)
print(t_stack.shape)

# %%
# flaten
t1 = torch.ones((4,4))
t2 = torch.ones((4,4))*2
t3 = torch.ones((4,4))*3
t_batch = torch.stack((t1, t2, t3), dim=0)
print(t_batch.shape)
t_batch = t_batch.unsqueeze(dim=1)
print(t_batch.shape)
t_batch = t_batch.flatten(start_dim=1)
print(t_batch.shape)
# %%
# element wise operations and broadcasting
t1 = torch.tensor([[1, 2], [4, 3]])
t2 = torch.tensor([[3, 4], [9, 0]])
print(t1-t2)
# %%
t1 = torch.rand((3,4,3))
t2 = torch.rand((3,3))
t = t1* t2
print(t.shape)
# %%
import torchvision
import torchvision.transforms as transforms

# %%
training_set = torchvision.datasets.FashionMNIST(
    root="./data/FashionMNIST",
    download=True,
    train=True,
    transform=transforms.Compose([
        transforms.ToTensor()
        ]
    )
)
# %%
sample = next(iter(training_set))
len(sample)
# %%
sample[0].shape

# %%
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=5, out_channels=12, kernel_size=5)
        
        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, x):
        return x



# %%
nn.Conv2d??
# %%

# %%
class A:
    magic = 5
    def __init__(self, name):
        self.name = name
    @classmethod
    def set_magic(cls, val):
        cls.magic = val

a = A("khan")
print(a.magic)
a.set_magic(77)
print(a.magic)

b = A("term")
print(b.magic)
b.set_magic(88)
print(A.magic)
print(a.magic)
a.magic = 70
print(b.magic)
# %%
import queue
q = queue.Queue()
q.put(1)
q.put('x')
# %%
q.deq()
# %%
class Rectangle:
    description = "I am a rectangle"
    def __init__(self, length, width):
        self.length = length
        self.width = width

    def area(self):
        return self.length * self.width

    def perimeter(self):
        return 2 * self.length + 2 * self.width

# Here we declare that the Square class inherits from the Rectangle class
class Square(Rectangle):
    def __init__(self, length):
        # super().__init__(length, length)
        self.length = length
# %%
s = Square(4)
print(s.length)
# print(s.area())
s.description = "I am a square"
print(s.description)
s2 = Square(5)
print(s2.description)
# %%

# %%

# %%
t = torch.tensor(np.array([1,2,3]))

def sigmoid(t):
    return t.exp()/(1+t.exp())

print(sigmoid(t))
# %%
