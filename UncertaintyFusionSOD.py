'''
加入边界

'''

import torch
import torch.nn as nn
from timm.models.layers import DropPath
import numpy as np
import torch.nn.functional as F
from models.SwinTransformers import SwinTransformer

class ConvMlp(nn.Module):
    def __init__(self, in_channels, hidden_channels=None):
        super().__init__()
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, 1)
        self.act = nn.GELU()
        self.fc2 = nn.Conv2d(hidden_channels, in_channels, 1)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))
def _stable_kl_mean(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    KL(N(mu, sigma^2) || N(0, 1)) 的稳定形式并做均值
    """
    # 为避免混合精度下溢出, 强制用 float32 计算 KL
    with torch.cuda.amp.autocast(enabled=False):
        mu32 = mu.float()
        lv32 = logvar.float()
        kl = 0.5 * (torch.exp(lv32) + mu32.pow(2) - 1.0 - lv32)
        return kl.mean()

class Trans_rgb(nn.Module):
    """
    单模态变换器. 输出 mu 和 logvar 以及 std
    使用 logvar 表征并裁剪以避免数值溢出
    """
    def __init__(self, dim: int):
        super().__init__()
        self.conv_attn = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.norm3 = nn.BatchNorm2d(dim)
        self.mlp_mu = ConvMlp(dim)
        self.mlp_lv = ConvMlp(dim)  # 输出 logvar

    def forward(self, x: torch.Tensor):
        x = x + self.conv_attn(self.norm1(x))
        mu = x + self.mlp_mu(self.norm2(x))
        logvar = x + self.mlp_lv(self.norm3(x))
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        std = torch.exp(0.5 * logvar)
        return mu, logvar, std


class Trans_rgbe(nn.Module):
    """
    跨模态交互. 输入 x_rgb 与 x_d
    输出 mu 和 logvar 以及 std
    """
    def __init__(self, dim: int):
        super().__init__()
        self.cross_attn = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, 1, 0),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
        )
        self.norm2 = nn.BatchNorm2d(dim)
        self.norm3 = nn.BatchNorm2d(dim)
        self.mlp_mu = ConvMlp(dim)
        self.mlp_lv = ConvMlp(dim)  # 输出 logvar

    def forward(self, x_rgb: torch.Tensor, x_d: torch.Tensor):
        x = torch.cat([x_rgb, x_d], dim=1)   # [B, 2C, H, W]
        x = x_rgb + self.cross_attn(x)       # residual 到 RGB 分支
        mu = x + self.mlp_mu(self.norm2(x))
        logvar = x + self.mlp_lv(self.norm3(x))
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        std = torch.exp(0.5 * logvar)
        return mu, logvar, std


class Trans_fusion(nn.Module):
    """
    RGB 与 RGBE 的融合
    """
    def __init__(self, dim: int):
        super().__init__()
        self.cross_attn = nn.Sequential(
            nn.Conv2d(dim * 2, dim, 1, 1, 0),
            nn.BatchNorm2d(dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
        )
        self.mlp = ConvMlp(dim)
        self.norm = nn.BatchNorm2d(dim)

    def forward(self, x_rgb: torch.Tensor, x_rgbe: torch.Tensor):
        x = torch.cat([x_rgb, x_rgbe], dim=1)  # [B, 2C, H, W]
        x = x_rgb + self.cross_attn(x)
        x = x + self.mlp(self.norm(x))
        return x



import torch
import matplotlib.pyplot as plt
import numpy as np
import os



class UncertaintyHead(nn.Module):
    """
    输入两个模态的 logvar，输出 conf_map ∈ [0,1] 的逐像素置信度图
    可选温度系数 T 控制门控平滑度
    """
    def __init__(self, T: float = 1.0):
        super().__init__()
        self.T = T
        self.refine = nn.Sequential(
            nn.Conv2d(2, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1)
        )

    def _inv_uncert_from_logvar(self, logvar: torch.Tensor) -> torch.Tensor:
        # var = exp(softplus(logvar)) 保障正定与数值稳定
        var = torch.exp(F.softplus(logvar))
        inv_u = 1.0 / (var.mean(dim=1, keepdim=True) + 1e-6)  # 通道均值聚合
        return inv_u

    def forward(self, logvar_a: torch.Tensor, logvar_b: torch.Tensor) -> torch.Tensor:
        inv_u_a = self._inv_uncert_from_logvar(logvar_a)
        inv_u_b = self._inv_uncert_from_logvar(logvar_b)
        x = torch.cat([inv_u_a, inv_u_b], dim=1)           # [B,2,H,W]
        x = self.refine(x)                                  # [B,1,H,W]
        conf = torch.sigmoid(x / max(self.T, 1e-6))         # 温度平滑
        # print(conf.min(), conf.max())
        # print(conf)
        # print(conf.shape, logvar_a.shape, logvar_b.shape,inv_u_a.shape,inv_u_b.shape)
        # visualize_conf_map(x, save_path='test_maps/boundary_20251007_vis/conf_map.png', cmap='plasma')
        # exit()
        return conf

class UncertaintyFusionSOD(nn.Module):
    """
    稳定不确定性融合
    返回 fused_feat 与逐像素 conf_map 以及 KL 项
    """
    def __init__(self, inp: int, out: int, T_conf: float = 1.0):
        super().__init__()
        dim = max(32, inp // 32)

        self.transformer_rgb = Trans_rgb(dim)
        self.transformer_depth = Trans_rgb(dim)
        self.transformer_rgbe = Trans_rgbe(dim)
        self.transformer_fusion = Trans_fusion(dim)

        self.conv1 = nn.Conv2d(inp // 2, dim, 1, 1, 0)
        self.conv2 = nn.Conv2d(inp // 2, dim, 1, 1, 0)
        self.conv3 = nn.Conv2d(dim, out // 2, 1, 1, 0)

        self.unc_head = UncertaintyHead(T=T_conf)

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor):
        '''
        if not self.training:
            x_r, x_d, x_rgbe = mu_r, mu_d, mu_rgbe
        else:
            x_r = mu_r + torch.randn_like(std_r) * std_r
            x_d = mu_d + torch.randn_like(std_d) * std_d
            x_rgbe = mu_rgbe + torch.randn_like(std_rgbe) * std_rgbe
        '''
        rgb = self.conv1(rgb)
        depth = self.conv2(depth)

        mu_r, logvar_r, std_r = self.transformer_rgb(rgb)
        # x_r = mu_r + torch.randn_like(std_r) * std_r
        if not self.training:
            x_r = mu_r
        else:
            x_r = mu_r + torch.randn_like(std_r) * std_r
        kl_r = _stable_kl_mean(mu_r, logvar_r)

        mu_d, logvar_d, std_d = self.transformer_depth(depth)
        # x_d = mu_d + torch.randn_like(std_d) * std_d
        if not self.training:
            x_d = mu_d
        else:
            x_d = mu_d + torch.randn_like(std_d) * std_d
        
        kl_d = _stable_kl_mean(mu_d, logvar_d)

        mu_rgbe, logvar_rgbe, std_rgbe = self.transformer_rgbe(x_r, x_d)
        # x_rgbe = mu_rgbe + torch.randn_like(std_rgbe) * std_rgbe
        if not self.training:
            x_rgbe = mu_rgbe
        else:
            x_rgbe = mu_rgbe + torch.randn_like(std_rgbe) * std_rgbe

        # print(mu_r.shape, mu_d.shape, mu_rgbe.shape)
        # print(x_r.shape, x_d.shape, x_rgbe.shape)
        # visualize_conf_map(depth, save_path='test_maps/boundary_20251007_vis/vis.png', cmap='plasma')

        fused = self.transformer_fusion(x_r, x_rgbe)
        fused = self.conv3(fused)

        conf_map = self.unc_head(logvar_r, logvar_d)       # [B,1,H,W]
        return fused, conf_map, kl_r, kl_d


def conv3x3_bn_relu(in_planes, out_planes, k=3, s=1, p=1, b=False):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=k, stride=s, padding=p, bias=b),
            nn.BatchNorm2d(out_planes),
            nn.GELU(),
            )



class UncertaintyFusionNet(nn.Module):
    def __init__(self):
        super(UncertaintyFusionNet, self).__init__()
        print("UncertaintyFusionNet, UncertaintyFusionSOD_PAPER_20251011")
        self.rgb_swin = SwinTransformer(embed_dim=128, depths=[2,2,18,2], num_heads=[4,8,16,32])
        self.depth_swin = SwinTransformer(embed_dim=128, depths=[2,2,18,2], num_heads=[4,8,16,32])
        self.up2 = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor = 4)



        self.CA_SA_Enhance_1 = CoordAttNotX(2048, 2048)
        self.CA_SA_Enhance_2 = CoordAttNotX(1024, 1024)
        self.CA_SA_Enhance_3 = CoordAtt(512, 512)
        self.CA_SA_Enhance_4 = CoordAtt(256, 256)

        self.FA_Block2 = Block(dim=256)
        self.FA_Block3 = Block(dim=128)
        self.FA_Block4 = Block(dim=64)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.deconv_layer_1 =  nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.GELU(),
            self.upsample2
        )
        self.deconv_layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=1024, out_channels=256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.GELU(),
            self.upsample2
        )
        self.deconv_layer_3 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            self.upsample2
        )
        self.deconv_layer_4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            self.upsample2
        )
        self.predict_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            self.upsample2,
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, bias=True),
            )
        self.predtrans2 = nn.Conv2d(128, 1, kernel_size=3, padding=1)
        self.predtrans3 = nn.Conv2d(256, 1, kernel_size=3, padding=1)
        self.predtrans4 = nn.Conv2d(512, 1, kernel_size=3, padding=1)
        self.dwc3 = conv3x3_bn_relu(256, 128)
        self.dwc2 = conv3x3_bn_relu(512, 256)
        self.dwc1 = conv3x3_bn_relu(1024, 512)
        self.dwcon_1 = conv3x3_bn_relu(2048, 1024)
        self.dwcon_2 = conv3x3_bn_relu(1024, 512)
        self.dwcon_3 = conv3x3_bn_relu(512, 256)
        self.dwcon_4 = conv3x3_bn_relu(256, 128)
        self.conv43 = conv3x3_bn_relu(128, 256, s=2)
        self.conv32 = conv3x3_bn_relu(256, 512, s=2)
        self.conv21 = conv3x3_bn_relu(512, 1024, s=2)

        #####
        self.boundary_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            self.upsample2,
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, bias=True),
            )
        self.boundary_layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, bias=True),
            )
        self.boundary_layer_3 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, bias=True),
            )
        self.boundary_layer_4 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, padding=1, bias=True),
            )

    def forward(self,x ,d):
        rgb_list = self.rgb_swin(x)
        depth_list = self.depth_swin(d)

        r4 = rgb_list[0]
        r3 = rgb_list[1]
        r2 = rgb_list[2]
        r1 = rgb_list[3]
        d4 = depth_list[0]
        d3 = depth_list[1]
        d2 = depth_list[2]
        d1 = depth_list[3]
        # visualize_conf_map(d1, save_path='test_maps/boundary_20251007_vis/depth1.png', cmap='plasma')
        # visualize_conf_map(d2, save_path='test_maps/boundary_20251007_vis/depth2.png', cmap='plasma')
        # visualize_conf_map(d3, save_path='test_maps/boundary_20251007_vis/depth3.png', cmap='plasma')
        # visualize_conf_map(d4, save_path='test_maps/boundary_20251007_vis/depth4.png', cmap='plasma')
        # exit()

        r3_up = F.interpolate(self.dwc3(r3), size=96, mode='bilinear')
        r2_up = F.interpolate(self.dwc2(r2), size=48, mode='bilinear')
        r1_up = F.interpolate(self.dwc1(r1), size=24, mode='bilinear')
        d3_up = F.interpolate(self.dwc3(d3), size=96, mode='bilinear')
        d2_up = F.interpolate(self.dwc2(d2), size=48, mode='bilinear')
        d1_up = F.interpolate(self.dwc1(d1), size=24, mode='bilinear')

        # visualize_conf_map(r3_up, save_path='test_maps/boundary_20251007_vis/r3_up.png', cmap='plasma')
        # visualize_conf_map(r2_up, save_path='test_maps/boundary_20251007_vis/r2_up.png', cmap='plasma')
        # visualize_conf_map(r1_up, save_path='test_maps/boundary_20251007_vis/r1_up.png', cmap='plasma')
        # visualize_conf_map(d3_up, save_path='test_maps/boundary_20251007_vis/d3_up.png', cmap='plasma')
        # visualize_conf_map(d2_up, save_path='test_maps/boundary_20251007_vis/d2_up.png', cmap='plasma')
        # visualize_conf_map(d1_up, save_path='test_maps/boundary_20251007_vis/d1_up.png', cmap='plasma')


        r1_con = torch.cat((r1, r1), 1)
        r1_con = self.dwcon_1(r1_con)
        # d1_con = torch.cat((d1, d1), 1)
        # d1_con = self.dwcon_1(d1_con)

        r2_con = torch.cat((r2, r1_up), 1)
        r2_con = self.dwcon_2(r2_con)
        # d2_con = torch.cat((d2, d1_up), 1)
        # d2_con = self.dwcon_2(d2_con)

        r3_con = torch.cat((r3, r2_up), 1)
        r3_con = self.dwcon_3(r3_con)
        d3_con = torch.cat((d3, d2_up), 1)
        d3_con = self.dwcon_3(d3_con)

        r4_con = torch.cat((r4, r3_up), 1)
        r4_con = self.dwcon_4(r4_con)
        d4_con = torch.cat((d4, d3_up), 1)
        d4_con = self.dwcon_4(d4_con)

        
        # exit()

        #
        xf_1 = self.CA_SA_Enhance_1(r1_con)  # 1024,12,12
        xf_2 = self.CA_SA_Enhance_2(r2_con)  # 512,24,24
        #
        xf_3, kl_rgb_3, kl_depth_3 = self.CA_SA_Enhance_3(r3_con, d3_con)  # 256,48,48
        xf_4, kl_rgb_4, kl_depth_4 = self.CA_SA_Enhance_4(r4_con, d4_con)  # 128,96,96


        df_f_1 = self.deconv_layer_1(xf_1)

        xc_1_2 = torch.cat((df_f_1, xf_2), 1)
        df_f_2 = self.deconv_layer_2(xc_1_2)
        df_f_2 = self.FA_Block2(df_f_2)

        xc_1_3 = torch.cat((df_f_2, xf_3), 1)
        df_f_3 = self.deconv_layer_3(xc_1_3)
        df_f_3 = self.FA_Block3(df_f_3)

        xc_1_4 = torch.cat((df_f_3, xf_4), 1)
        df_f_4 = self.deconv_layer_4(xc_1_4)
        df_f_4 = self.FA_Block4(df_f_4)
        y1 = self.predict_layer_1(df_f_4)
        y2 = F.interpolate(self.predtrans2(df_f_3), size=384, mode='bilinear')
        y3 = F.interpolate(self.predtrans3(df_f_2), size=384, mode='bilinear')
        y4 = F.interpolate(self.predtrans4(df_f_1), size=384, mode='bilinear')

        b1 = self.boundary_layer_1(df_f_4)
        b2 = F.interpolate(self.boundary_layer_2(df_f_3), size=384, mode='bilinear')
        b3 = F.interpolate(self.boundary_layer_3(df_f_2), size=384, mode='bilinear')
        b4 = F.interpolate(self.boundary_layer_4(df_f_1), size=384, mode='bilinear')

        kl_rgb = kl_rgb_3 + kl_rgb_4
        kl_depth = kl_depth_3 + kl_depth_4
        return y1,y2,y3,y4, kl_rgb, kl_depth, b1, b2, b3, b4

    def load_pre(self, pre_model):
        self.rgb_swin.load_state_dict(torch.load(pre_model)['model'],strict=False)
        print(f"RGB SwinTransformer loading pre_model ${pre_model}")
        self.depth_swin.load_state_dict(torch.load(pre_model)['model'], strict=False)
        print(f"Depth SwinTransformer loading pre_model ${pre_model}")


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class SA_Enhance(nn.Module):
    def __init__(self, kernel_size=7):
        super(SA_Enhance, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)



# ========= 全局指导调制：FiLM 通道调制 + 空间先验门控 =========
class GlobalGuidedModulation(nn.Module):
    """
    用 align_fea 作为全局先验指导像素级特征 out
    conf_map 控制指导强度
    out_ch 为 out 的通道数
    guide_ch 为 align_fea 的通道数
    """
    def __init__(self, out_ch: int, guide_ch: int):
        super().__init__()
        hid_film = max(out_ch, 16)
        hid_prior = max(guide_ch // 4, 16)

        self.guide_pool = nn.AdaptiveAvgPool2d(1)
        self.film_head = nn.Sequential(
            nn.Conv2d(guide_ch, hid_film, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid_film, out_ch * 2, 1, bias=True)  # 生成 [gamma | beta]
        )

        self.prior_head = nn.Sequential(
            nn.Conv2d(guide_ch, hid_prior, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid_prior, 1, 1, bias=True),
            nn.Sigmoid()
        )

        

    def forward(self, out: torch.Tensor, align_fea: torch.Tensor, conf_map: torch.Tensor) -> torch.Tensor:
        # 通道调制参数
        g = self.guide_pool(align_fea)                     # [B, Cg,1,1]
        gb = self.film_head(g)                             # [B, 2*Co,1,1]
        gamma, beta = torch.chunk(gb, 2, dim=1)            # [B, Co,1,1]

        # 广播 conf_map 到通道维
        while conf_map.dim() < out.dim():
            conf_map = conf_map.unsqueeze(1)

        out_mod = (out * gamma + beta) * conf_map

        # 空间先验掩膜
        P = self.prior_head(align_fea)                     # [B,1,H,W]
        P = P * conf_map

        # 残差式融合，数值稳定
        out = out + P * ( out_mod - out)
        return out
    
# ========= 在 CoordAtt 中调用封装模块的示例 =========
class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, 1, 1, 0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, 1, 1, 0)
        self.conv_w = nn.Conv2d(mip, oup, 1, 1, 0)
        self.conv_end = nn.Conv2d(oup, oup // 2, 1, 1, 0)
        self.self_SA_Enhance = SA_Enhance()

        self.alignblock = UncertaintyFusionSOD(inp, oup)
        self.guided_mod = GlobalGuidedModulation(out_ch=oup // 2, guide_ch=oup//2)

    def forward(self, rgb, depth):
       


        x = torch.cat((rgb, rgb), dim=1)
        n, c, h, w = x.size()

        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out_ca = x * a_w * a_h
        out_sa = self.self_SA_Enhance(out_ca)
        out = x.mul(out_sa)
        out = self.conv_end(out)                           # [B, Co, H, W], Co = oup//2

        
        align_fea, conf_map, kl1, kl2 = self.alignblock(rgb, depth)  # align_fea: [B, oup, H, W]

        out = self.guided_mod(out, align_fea, conf_map)
        return out, kl1, kl2






class CoordAttNotX(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, 1, 1, 0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, 1, 1, 0)
        self.conv_w = nn.Conv2d(mip, oup, 1, 1, 0)
        self.conv_end = nn.Conv2d(oup, oup // 2, 1, 1, 0)
        self.self_SA_Enhance = SA_Enhance()



    def forward(self, rgb):
        x = torch.cat((rgb, rgb), dim=1)
        n, c, h, w = x.size()

        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out_ca = x * a_w * a_h
        out_sa = self.self_SA_Enhance(out_ca)
        out = x.mul(out_sa)
        out = self.conv_end(out)                           # [B, Co, H, W], Co = oup//2
        return out





def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_first"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Block(nn.Module):
    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim,)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C]
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W]

        x = shortcut + self.drop_path(x)
        return x
