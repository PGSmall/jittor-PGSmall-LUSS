# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
from functools import partial
import math
import warnings

import jittor as jt
from jittor import init
from jittor import nn

from .decoder_head import EAHead


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):

    def norm_cdf(x):
        return ((1.0 + math.erf((x / math.sqrt(2.0)))) / 2.0)
    if ((mean < (a - (2 * std))) or (mean > (b + (2 * std)))):
        warnings.warn('mean is more than 2 std from [a, b] in nn.init.trunc_normal_. The distribution of values may be incorrect.', stacklevel=2)
    with jt.no_grad():
        l = norm_cdf(((a - mean) / std))
        u = norm_cdf(((b - mean) / std))
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor = tensor.erfinv()
        tensor = tensor.multiply((std * math.sqrt(2.0)))
        tensor = tensor.add(mean)
        tensor = tensor.clamp(min_v=a, max_v=b)
        return tensor

def drop_path(x, drop_prob: float=0.0, training: bool=False):
    if ((drop_prob == 0.0) or (not training)):
        return x
    keep_prob = (1 - drop_prob)
    shape = ((x.shape[0],) + ((1,) * (x.ndim - 1)))
    random_tensor = (keep_prob + jt.rand(shape, dtype=x.dtype, device=x.device))
    random_tensor.floor_()
    output = (x.div(keep_prob) * random_tensor)
    return output

class DropPath(nn.Module):

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def execute(self, x):
        return drop_path(x, self.drop_prob, self.training)


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """
    def __init__(self, dim):
        super().__init__()
        self.gamma = jt.array(jt.zeros(1, 1, 1, dim))
        self.beta = jt.array(jt.zeros(1, 1, 1, dim))

    def execute(self, x):
        Gx = jt.norm(x, p=2, dim=(1,2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdims=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-06, data_format='channels_last'):
        super().__init__()
        self.weight = jt.array(jt.ones(normalized_shape))
        self.bias = jt.array(jt.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if (self.data_format not in ['channels_last', 'channels_first']):
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def execute(self, x):
        if (self.data_format == 'channels_last'):
            return nn.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif (self.data_format == 'channels_first'):
            u = x.mean(1, keepdims=True)
            s = (x - u).pow(2).mean(1, keepdims=True)
            x = ((x - u) / jt.sqrt((s + self.eps)))
            x = ((self.weight[:, None, None] * x) + self.bias[:, None, None])
            return x


class Block(nn.Module):
    """ ConvNeXtV2 Block.
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def execute(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class Mulpixelattn(nn.Module):
    def __init__(self, channels):
        super().__init__()
        hidden_mlp = channels
        self.atten = nn.Sequential(
            nn.Conv2d(channels, hidden_mlp, 1),
            nn.BatchNorm2d(hidden_mlp),
            nn.ReLU(),
            nn.Conv2d(hidden_mlp, channels, 1),
            nn.BatchNorm2d(channels, affine=True),
        )
        self.threshold = jt.zeros((1, channels, 1, 1))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def execute(self, x):
        x = self.atten(x)
        x = x + self.threshold
        att = jt.sigmoid(x)
        return att


class ConvNeXtV2(nn.Module):
    """ ConvNeXt V2
        
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=919, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], 
                 drop_path_rate=0., head_init_scale=1.,
                 normalize=False, output_dim=0, hidden_mlp=0, nmb_prototypes=0,
                 eval_mode=False, train_mode='finetune', shallow=None
                 ):
        super().__init__()
        assert train_mode in ['pretrain', 'pixelattn', 'finetune'], train_mode

        self.eval_mode = eval_mode
        self.train_mode = train_mode
        self.shallow = shallow
        if isinstance(self.shallow, int):
            self.shallow = [self.shallow]
        
        self.depths = depths
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates = [x.item() for x in jt.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j]) for j in range(depths[i])],
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first")
            )
            self.stages.append(stage)
            cur += depths[i]

        # norm_layer = partial(LayerNorm, eps=1e-6, data_format="channels_first")
        # for i_layer in range(4):
        #     layer = norm_layer(dims[i_layer])
        #     layer_name = f'norm{i_layer}'
        #     self.add_module(layer_name, layer)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # normalize output features
        self.l2norm = normalize

        # self.norm = LayerNorm(dims[-1], eps=1e-6) # final norm layer
        # self.head = nn.Linear(dims[-1], num_classes)

        # projection head
        if output_dim == 0:
            self.projection_head = None
            if self.train_mode == 'pretrain':
                self.projection_head_shallow = None
        elif hidden_mlp == 0:
            self.projection_head = nn.Linear(dims[3], output_dim)
            if self.train_mode == 'pretrain':
                self.projection_head_shallow = nn.ModuleList()
                if self.shallow is not None:
                    for stage in shallow:
                        assert stage < 4
                        self.projection_head_shallow.add_module(
                            f'projection_head_shallow{stage}',
                            nn.Linear(dims[3], output_dim))

        else:
            mlps = [
                nn.Linear(dims[3], hidden_mlp),
                nn.BatchNorm1d(hidden_mlp),
                nn.ReLU(),
                nn.Linear(hidden_mlp, output_dim),
                nn.BatchNorm1d(output_dim, affine=False)
            ]
            self.projection_head = nn.Sequential(*mlps)
            if self.train_mode == 'pretrain':
                self.projection_head_shallow = nn.ModuleList()
                if self.shallow is not None:
                    for stage in shallow:
                        assert stage < 4
                        self.projection_head_shallow.add_module(
                            f'projection_head_shallow{stage}',
                            nn.Sequential(
                                nn.Linear(dims[3] // (2 * (4 - stage)),hidden_mlp),
                                nn.BatchNorm1d(hidden_mlp),
                                nn.ReLU(),
                                nn.Linear(hidden_mlp, output_dim),
                                nn.BatchNorm1d(output_dim, affine=False)))
        if self.train_mode == 'pretrain':
            self.projection_head_pixel_shallow = nn.ModuleList()
            if self.shallow is not None:
                for stage in shallow:
                    assert stage < 4
                    self.projection_head_pixel_shallow.add_module(
                        f'projection_head_pixel{stage}',
                        nn.Sequential(
                            nn.Conv2d(
                                dims[3] // (2 * (4 - stage)),
                                hidden_mlp,
                                kernel_size=1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(hidden_mlp),
                            nn.ReLU(),
                            nn.Conv2d(hidden_mlp,
                                      hidden_mlp,
                                      kernel_size=1,
                                      bias=False),
                            nn.BatchNorm2d(hidden_mlp),
                            nn.ReLU(),
                            nn.Conv2d(
                                hidden_mlp,
                                dims[3] // (2 * (4 - stage)),
                                kernel_size=1,
                                bias=False,
                            ),
                            nn.BatchNorm2d(dims[3] //
                                           (2 * (4 - stage)),
                                           affine=False),
                        ))

            # projection for pixel-to-pixel
            self.projection_head_pixel = nn.Sequential(
                nn.Conv2d(dims[3],
                          hidden_mlp,
                          kernel_size=1,
                          bias=False),
                nn.BatchNorm2d(hidden_mlp),
                nn.ReLU(),
                nn.Conv2d(hidden_mlp, hidden_mlp, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_mlp),
                nn.ReLU(),
                nn.Conv2d(hidden_mlp, output_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(output_dim, affine=False),
            )
            self.predictor_head_pixel = nn.Sequential(
                nn.Conv2d(output_dim, output_dim, 1, bias=False),
                nn.BatchNorm2d(output_dim),
                nn.ReLU(),
                nn.Conv2d(output_dim, output_dim, 1),
            )

        # prototype layer
        self.prototypes = None
        if isinstance(nmb_prototypes, list):
            self.prototypes = MultiPrototypes(output_dim, nmb_prototypes)
        elif nmb_prototypes > 0:
            self.prototypes = nn.Linear(output_dim, nmb_prototypes, bias=False)

        if train_mode == 'pixelattn':
            self.fbg = Mulpixelattn(dims[3])
        elif train_mode == 'finetune':
            # self.last_layer = nn.Conv2d(dims[3], num_classes + 1, 1, 1)
            self.last_layer = EAHead(in_channels=dims[3], 
                                     channels=512, num_classes=num_classes + 1)

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d, nn.Linear):


        self.apply(self._init_weights)
        # self.head.weight.data.mul_(head_init_scale)
        # self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            # init.constant_(m.bias, value=0)

    def execute_backbone(self, x, avgpool=True):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            outs.append(x)
        
        if self.eval_mode or self.train_mode != 'pretrain':
            return x
        
        if avgpool:
            x = self.avgpool(x)
            x = jt.flatten(x, 1)
            return x
        
        return outs[3], outs[2], outs[1], outs[0]

    def execute_head(self, x):
        if self.projection_head is not None:
            x = self.projection_head(x)

        if self.l2norm:
            x = jt.normalize(x, dim=1, p=2)

        if self.prototypes is not None:
            return x, self.prototypes(x)
        return x

    def execute_head_shallow(self, x, stage):
        if (self.projection_head_shallow is not None
                and f'projection_head_shallow{stage}'
                in self.projection_head_shallow.keys()):
            x = self.projection_head_shallow.layers[
                f'projection_head_shallow{stage}'](x)

        if self.l2norm:
            x = jt.normalize(x, dim=1, p=2)

        if self.prototypes is not None:
            return x, nn.matmul_transpose(x, self.prototypes.weight.detach()) 
        return x

    def execute_head_pixel(self, x, gridq, gridk):
        if self.projection_head_pixel is not None:
            x = self.projection_head_pixel(x)

        # grid sample 28 x 28
        grid = jt.concat([gridq, gridk], dim=0)
        x = nn.grid_sample(x, grid, align_corners=False, mode='bilinear')

        return x, self.predictor_head_pixel(x)


    def execute(self, inputs, gridq=None, gridk=None, mode='train'):
        if mode == 'cluster':
            output = self.inference_cluster(inputs)
            return output
        elif mode == 'inference_pixel_attention':
            return self.inference_pixel_attention(inputs)

        if self.train_mode == 'finetune':
            out = self.execute_backbone(inputs)
            return self.last_layer(out)

        if not isinstance(inputs, list):
            inputs = [inputs]
        idx_crops, last_size = [0], inputs[0].shape[-1]
        for sample in [inp.shape[-1] for inp in inputs]:
            if sample == last_size:
                idx_crops[-1] += 1
            else:
                idx_crops.append(idx_crops[-1] + 1)

        start_idx = 0
        for end_idx in idx_crops:
            _out = self.execute_backbone(jt.concat(
                inputs[start_idx:end_idx]), avgpool=self.train_mode != 'pretrain')
            if start_idx == 0:
                if self.train_mode == 'pixelattn':
                    _out = self.execute_pixel_attention(_out)
                elif self.train_mode == 'pretrain':
                    _out, _c3, _c2, _c1 = _out
                    (
                        embedding_deep_pixel,
                        output_deep_pixel,
                    ) = self.execute_head_pixel(_out, gridq, gridk)
                    _stages = [_c1, _c2, _c3]
                    if self.shallow is not None:
                        output_c = []
                        for i, stage in enumerate(self.shallow):
                            _c = _stages[stage - 1]
                            _out_c = self.projection_head_pixel_shallow.layers[
                                f'projection_head_pixel{stage}'](_c)
                            _out_c = self.avgpool(_out_c)
                            _out_c = jt.flatten(_out_c, 1)
                            output_c.append(_out_c)
                    _out = self.avgpool(_out)
                    _out = jt.flatten(_out, 1)
                output = _out
            else:
                if self.train_mode == 'pixelattn':
                    _out = self.execute_pixel_attention(_out)
                elif self.train_mode == 'pretrain':
                    _out, _, _, _ = _out
                    _out = self.avgpool(_out)
                    _out = jt.flatten(_out, 1)
                output = jt.concat((output, _out))
            start_idx = end_idx

        embedding, output = self.execute_head(output)
        if self.shallow is not None:
            for i, stage in enumerate(self.shallow):
                embedding_c_, output_c_ = self.execute_head_shallow(output_c[i],
                                                                  stage=stage)
                embedding = jt.concat((embedding, embedding_c_))
                output = jt.concat((output, output_c_))
        if self.train_mode == 'pixelattn':
            return embedding, output
        elif self.train_mode == 'pretrain':
            return embedding, output, embedding_deep_pixel, output_deep_pixel
        return embedding, output

    def execute_pixel_attention(self, out, threshold=None):
        out = nn.interpolate(
            out, 
            size=(out.shape[2] * 4, out.shape[3] * 4), 
            mode='bilinear')
        out = jt.normalize(out, dim=1, p=2)
        fg = self.fbg(out)
        if threshold is not None:
            fg[fg < threshold] = 0

        out = out * fg
        out = self.avgpool(out)
        out = jt.flatten(out, 1)

        return out

    def inference_cluster(self, x, threshold=None):
        out = self.execute_backbone(x)
        out = nn.interpolate(
            out, 
            size=(out.shape[2] * 4, out.shape[3] * 4), 
            mode='bilinear')
        nout = jt.normalize(out, dim=1, p=2)
        fg = self.fbg(nout)
        if threshold is not None:
            fg[fg < threshold] = 0

        out = out * fg
        out = self.avgpool(out)
        out = jt.flatten(out, 1)

        return out

    def inference_pixel_attention(self, x):
        out = self.execute_backbone(x)

        out = nn.interpolate(
            out, 
            size=(out.shape[2] * 4, out.shape[3] * 4), 
            mode='bilinear')
        out_ = jt.normalize(out, dim=1, p=2)
        fg = self.fbg(out_)
        fg = fg.mean(dim=1, keepdims=True)

        return out, fg


class MultiPrototypes(nn.Module):
    def __init__(self, output_dim, nmb_prototypes):
        super().__init__()
        self.nmb_heads = len(nmb_prototypes)
        for i, k in enumerate(nmb_prototypes):
            setattr(self, 'prototypes' + str(i),
                            nn.Linear(output_dim, k, bias=False))

    def execute(self, x):
        out = []
        for i in range(self.nmb_heads):
            out.append(getattr(self, 'prototypes' + str(i))(x))
        return out


def convnextv2_atto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[40, 80, 160, 320], **kwargs)
    return model


def convnextv2_femto(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[48, 96, 192, 384], **kwargs)
    return model


def convnextv2_pico(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 6, 2], dims=[64, 128, 256, 512], **kwargs)
    return model


def convnextv2_nano(**kwargs):
    model = ConvNeXtV2(depths=[2, 2, 8, 2], dims=[80, 160, 320, 640], **kwargs)
    return model


def convnextv2_tiny(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], **kwargs)
    return model


def convnextv2_base(**kwargs):
    model = ConvNeXtV2(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024], **kwargs)
    return model