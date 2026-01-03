# from vmunet import VMUNet
from VSwin import vswin_tiny
import torch
# model = VMUNet().to("cuda")
model = vswin_tiny().to('cuda')
input1 = torch.randn(4, 96, 128, 128).to('cuda')
input2 = torch.randn(4, 192, 64, 64).to('cuda')
input3 = torch.randn(4, 384, 32, 32).to('cuda')
input4 = torch.randn(4, 768, 16, 16).to('cuda')
input = [input1, input2, input3, input4]
# input = torch.randn(4, 3, 512, 512).to("cuda")
output = model(input,)
for out in output:
    print(out.shape)

# import torch
# from mamba2 import VMAMBA2Block
# model = VMAMBA2Block(96, (128, 128), 2).to('cuda')
# input = torch.randn(4, 4096, 96).to('cuda')
# H, W = (64, 64)
# output = model(input, H, W)
# print(output.shape)