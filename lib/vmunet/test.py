import torch
from vmunet import VMUNet
from vmamba import VSSBlock
from mamba2 import Mamba2
hidden_dim = 512
num_heads = 16
ssd_expansion=2
ssd_ngroups=1
ssd_chunk_size=256
linear_attn_duality=True
d_state=16
model = VMUNet().to('cuda')
model2 = VSSBlock(hidden_dim=512, num_heads=16).to('cuda')
model3 = Mamba2(d_model=hidden_dim, expand=ssd_expansion, headdim= hidden_dim*ssd_expansion // num_heads,
                                ngroups=ssd_ngroups, chunk_size=ssd_chunk_size,
                                linear_attn_duality=linear_attn_duality, d_state=d_state).to('cuda')
# x1 = torch.randn(4, 64, 128, 128).to('cuda')
# x2 = torch.randn(4, 128, 64, 64).to('cuda')
# x3 = torch.randn(4, 320, 32, 32).to('cuda')
x4 = torch.randn(4, 256, 512).to('cuda')
# x = [x1, x2, x3, x4]
# x = torch.randn(4, 3, 512, 512).to('cuda')
y = model3(x4, 16, 16)
print(y.shape)
# for i in y:
#     print(i.shape)
# print(2**(3-0)*96)
