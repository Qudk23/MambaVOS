import torch
from efficientnet_pytorch import EfficientNet
class EfficientNetBackbone(EfficientNet):
    def __init__(self, blocks_args=None, global_params=None):
        super(EfficientNetBackbone, self).__init__(blocks_args, global_params)

    def forward(self, x):
        outlist = self.extract_endpoints(x)
        results = []
        for i, out in enumerate(outlist.values()):
            if i == 0 or i == 5: pass
            else: results.append(out)
        return results
# model = EfficientNetBackbone.from_pretrained('efficientnet-b3').to('cuda')
# input = torch.randn(1, 3, 512, 512).to('cuda')
# outlist = model(input)
# for out in outlist:
#     print(out.shape)
