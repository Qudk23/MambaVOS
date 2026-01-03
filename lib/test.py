import torch

from builder import VOSNet
from config import Parameters
opt = Parameters().parse()
model = VOSNet(opt).to('cuda')
print(model)
x = torch.randn(4, 3, 512, 512).to('cuda')
y = torch.randn(4, 3, 512, 512).to('cuda')
z = model(x ,y)
for i in z:
    print(i.shape)
