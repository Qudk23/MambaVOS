import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

# from lib.vmunet.test import ssd_ngroups
from .mamba2 import Mamba2
from timm.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined
from mamba_ssm.ops.triton.ssd_combined import mamba_split_conv1d_scan_combined
from mamba_ssm.ops.triton.layernorm_gated import RMSNorm as RMSNormGated
from mamba_ssm.ops.triton.selective_state_update import selective_state_update
from einops import rearrange, repeat

import math
import copy
try:
    from mamba_util import PatchMerging,SimplePatchMerging, Stem, SimpleStem, Mlp
except:
    from .mamba_util import PatchMerging, SimplePatchMerging, Stem, SimpleStem, Mlp
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count

class tTensor(torch.Tensor):
    @property
    def shape(self):
        shape = super().shape
        return tuple([int(s) for s in shape])


to_ttensor = lambda *args: tuple([tTensor(x) for x in args]) if len(args) > 1 else tTensor(args[0])


class PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim * 2
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // self.dim_scale)
        x = self.norm(x)

        return x


class VMAMBA2Block(nn.Module):
    r""" MLLA Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True, drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, ssd_expansion=2, ssd_ngroups=1, ssd_chunk_size=256,
                 linear_attn_duality=False, d_state = 64, **kwargs):
        super().__init__()
        self.ssd_expansion = ssd_expansion
        self.ssd_ngroups = ssd_ngroups
        self.ssd_chunk_size = ssd_chunk_size
        self.linear_attn_duality = linear_attn_duality
        self.d_state = d_state
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.cpe1 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm1 = norm_layer(dim)
        self.attn = Mamba2(d_model=dim, expand=ssd_expansion, headdim= dim*ssd_expansion // num_heads,
                                ngroups=ssd_ngroups, chunk_size=ssd_chunk_size,
                                linear_attn_duality=linear_attn_duality, d_state=d_state, **kwargs)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.cpe2 = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x):
        B, H, W, C = x.shape
        cpe1 = self.cpe1(x.permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)
        x = x.permute(0, 3, 1, 2).flatten(2).permute(0, 2, 1)
        x = x + cpe1
        shortcut = x
        print(self.ssd_expansion, self.ssd_ngroups, self.ssd_chunk_size, self.d_state, self.linear_attn_duality)
        x = self.norm1(x)
        # SSD or Standard Attention
        x = self.attn(x, H, W)
        x = shortcut + self.drop_path(x)
        x = x + self.cpe2(x.reshape(B, H, W, C).permute(0, 3, 1, 2)).flatten(2).permute(0, 2, 1)

        # FFN
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = x.reshape(B, H, W, C)
        return x

class BasicLayer(nn.Module):
    """ A basic MLLA layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        drop (float, optional): Dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, depth, num_heads, mlp_ratio=4., qkv_bias=True, drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False,
                 ssd_expansion=2, ssd_ngroups=1, ssd_chunk_size=256, linear_attn_duality=False, d_state=64, **kwargs):

        super().__init__()
        self.dim = dim
        # self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            VMAMBA2Block(dim=dim, num_heads=num_heads,
                      mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop,
                      drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path, norm_layer=norm_layer,
                      ssd_expansion=ssd_expansion, ssd_ngroups=ssd_ngroups, ssd_chunk_size=ssd_chunk_size,
                      linear_attn_duality=linear_attn_duality, d_state=d_state, **kwargs)
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            self.upsample = upsample(dim=dim)
        else:
            self.upsample = None

    def forward(self, x):
        if self.upsample is not None:
            x = self.upsample(x)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        return x

class VMAMBA2(nn.Module):
    def __init__(self, patch_size=4,
                 embed_dim=[768, 384, 192, 96], depths=[2, 9, 2, 2], num_heads=[16, 8, 4, 2],
                 mlp_ratio=4., qkv_bias=True, drop_rate=0., drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm, use_checkpoint=False,
                 ssd_expansion=2, ssd_ngroups=1, ssd_chunk_size=256,
                 linear_attn_duality= False, d_state=16, **kwargs):
        super().__init__()
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        # self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # self.simple_downsample = kwargs.get('simple_downsample', False)
        # self.simple_patch_embed = kwargs.get('simple_patch_embed', False)
        #self.attn_types = kwargs.get('attn_types', ['mamba2', 'mamba2', 'mamba2', 'standard'])
        # if self.simple_patch_embed:
        #     self.patch_embed = SimpleStem(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim[0])
        # else:
        #     self.patch_embed = Stem(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim[0])
        # if self.simple_downsample:
        #     PatchMergingBlock = SimplePatchMerging
        # else:
        #     PatchMergingBlock = PatchMerging
        # num_patches = self.patch_embed.num_patches
        # patches_resolution = self.patch_embed.patches_resolution
        # self.patches_resolution = patches_resolution


        # self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers_up = nn.ModuleList()
        for i_layer in range(self.num_layers):
            # kwargs['attn_type'] = self.attn_types[i_layer]
            layer = BasicLayer(dim=embed_dim[i_layer],
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, drop=drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               upsample=PatchExpand2D if (i_layer > 0) else None,
                               use_checkpoint=use_checkpoint,
                               ssd_expansion=ssd_expansion,
                               ssd_ngroups=ssd_ngroups,
                               ssd_chunk_size=ssd_chunk_size,
                               linear_attn_duality = linear_attn_duality,
                               d_state = d_state,
                               **kwargs)
            self.layers_up.append(layer)

        # self.norm = norm_layer(self.num_features)
        # self.avgpool = nn.AdaptiveAvgPool1d(1)
        # self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.no_grad()
    def flops(self, shape=(3, 224, 224), verbose=True):
        # shape = self.__input_shape__[1:]
        supported_ops = {
            "aten::silu": None,  # as relu is in _IGNORED_OPS
            "aten::neg": None,  # as relu is in _IGNORED_OPS
            "aten::exp": None,  # as relu is in _IGNORED_OPS
            "aten::flip": None,  # as permute is in _IGNORED_OPS
        }

        model = copy.deepcopy(self)
        model.cuda().eval()

        input = torch.randn((1, *shape), device=next(model.parameters()).device)
        params = parameter_count(model)[""]
        try:
            Gflops, unsupported = flop_count(model=model, inputs=(input,), supported_ops=supported_ops)
        except Exception as e:
            print('get exception', e)
            print('Error in flop_count, set to default value 1e9')
            return 1e9
        del model, input

        return sum(Gflops.values()) * 1e9
    def forward(self, input):
        for i in range(len(input)):
            input[i] = input[i].permute(0, 2, 3, 1)
        x, skip_list = input[3], input
        outlist = []
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
                outlist.append(x.permute(0, 3, 1, 2))
            else:
                x = layer_up(x + skip_list[-inx])
                outlist.append(x.permute(0, 3, 1, 2))
        outlist.reverse()
        return outlist

if __name__ == '__main__':
    model = VMAMBA2().to('cuda')
    x1 = torch.randn(4, 96, 128, 128).to('cuda')
    x2 = torch.randn(4, 192, 64, 64).to('cuda')
    x3 = torch.randn(4, 384, 32, 32).to('cuda')
    x4 = torch.randn(4, 768, 16, 16).to('cuda')
    x = [x1, x2, x3, x4]
    # x = torch.randn(4, 3, 512, 512).to('cuda')
    y = model(x)
    for i in y:
        print(i.shape)
    # print(2**(3-0)*96)
