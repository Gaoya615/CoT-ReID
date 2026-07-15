import torch.nn as nn
import einops
import torch.nn.functional as F
import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import cv2
from modeling.backbones.vit_pytorch import trunc_normal_
import textwrap


class DAttentionBaseline(nn.Module):

    def __init__(
            self, q_size, n_heads, n_head_channels, n_groups,
            attn_drop, proj_drop, stride,
            offset_range_factor, ksize, share
    ):

        super().__init__()
        self.n_head_channels = n_head_channels
        self.scale = self.n_head_channels ** -0.5
        self.n_heads = n_heads
        self.q_h, self.q_w = q_size
        self.kv_h, self.kv_w = self.q_h // stride, self.q_w // stride
        self.nc = n_head_channels * n_heads
        self.n_groups = n_groups
        self.n_group_channels = self.nc // self.n_groups
        self.n_group_heads = self.n_heads // self.n_groups
        self.offset_range_factor = offset_range_factor
        self.ksize = ksize
        self.stride = stride
        kk = self.ksize
        pad_size = 0
        self.share_offset = share
        if self.share_offset:
            self.conv_offset = nn.Sequential(
                nn.Conv2d(3 * self.n_group_channels, self.n_group_channels, 1, 1, 0),
                nn.GELU(),
                nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, pad_size,
                          groups=self.n_group_channels),
                nn.GELU(),
                nn.Conv2d(self.n_group_channels, 2, 1, 1, 0, bias=False),
            )
        else:
            self.conv_offset_r = nn.Sequential(
                nn.Conv2d(self.n_group_channels, self.n_group_channels, 1, 1, 0),
                nn.GELU(),
                nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, pad_size,
                          groups=self.n_group_channels),
                nn.GELU(),
                nn.Conv2d(self.n_group_channels, 1, 1, 1, 0, bias=False)
            )
            self.conv_offset_n = nn.Sequential(
                nn.Conv2d(self.n_group_channels, self.n_group_channels, 1, 1, 0),
                nn.GELU(),
                nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, pad_size,
                          groups=self.n_group_channels),
                nn.GELU(),
                nn.Conv2d(self.n_group_channels, 1, 1, 1, 0, bias=False)
            )
            self.conv_offset_t = nn.Sequential(
                nn.Conv2d(self.n_group_channels, self.n_group_channels, 1, 1, 0),
                nn.GELU(),
                nn.Conv2d(self.n_group_channels, self.n_group_channels, kk, stride, pad_size,
                          groups=self.n_group_channels),
                nn.GELU(),
                nn.Conv2d(self.n_group_channels, 1, 1, 1, 0, bias=False)
            )

        self.proj_q = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_k = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_v = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_out = nn.Conv2d(
            self.nc, self.nc,
            kernel_size=1, stride=1, padding=0
        )

        self.proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = nn.Dropout(attn_drop)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.no_grad()
    def _get_ref_points(self, H_in, W_in, B, kernel_size, stride, dtype, device):
        """
        生成参考点在每个卷积块的中心位置。

        :param H_in: 输入特征图的高度 (如 16)
        :param W_in: 输入特征图的宽度 (如 8)
        :param B: 批次大小
        :param kernel_size: 卷积核大小
        :param stride: 卷积步幅
        :param dtype: 数据类型
        :param device: 设备类型
        :return: 参考点张量，形状为 (B * n_groups, H_out, W_out, 2)
        """

        # 计算输出特征图的高度和宽度
        H_out = (H_in - kernel_size) // stride + 1
        W_out = (W_in - kernel_size) // stride + 1

        # 计算每个卷积位置的中心点在原图坐标上的位置
        center_y = torch.arange(H_out, dtype=dtype, device=device) * stride + (kernel_size // 2)
        center_x = torch.arange(W_out, dtype=dtype, device=device) * stride + (kernel_size // 2)

        # 生成网格
        ref_y, ref_x = torch.meshgrid(center_y, center_x, indexing='ij')
        ref = torch.stack((ref_y, ref_x), dim=-1)  # Shape: (H_out, W_out, 2)

        # 归一化到 [-1, 1]
        ref[..., 1].div_(W_in - 1.0).mul_(2.0).sub_(1.0)  # x 坐标归一化
        ref[..., 0].div_(H_in - 1.0).mul_(2.0).sub_(1.0)  # y 坐标归一化

        # 扩展批次和组维度
        ref = ref[None, ...].expand(B * self.n_groups, -1, -1, -1)  # Shape: (B * n_groups, H_out, W_out, 2)

        return ref

    @torch.no_grad()
    def visualize_sampling_with_offset(self, feature_maps, sampled_pointss, img_paths, reference_pointss, writer=None,
                                       epoch=0, title='Sampling Points with Offset', pattern=0, patch_size=(16, 16)):
        """
        在原始图像上标记采样点的偏移效果并保存
        :param feature_map: 原始特征图，形状为 (C, H, W)
        :param sampled_points: 目标采样点，形状为 (N, 2)，其中 N 是采样点的数量，2 对应于 (y, x)
        :param img_path: 原始图像路径
        :param reference_points: 参考采样点，用于标记偏移起始位置，形状为 (N, 2)
        :param writer: 用于保存图像的writer (比如Tensorboard的SummaryWriter)
        :param epoch: 当前epoch，用于保存不同epoch的结果
        :param title: 图形标题
        :param patch_size: 每个 patch 的尺寸 (height, width)
        """
        # 根据模式设置图像路径前缀
        modality = ['RGB', 'NI', 'TI']
        if pattern == 0:
            prefix = '../RGBNT201/test/RGB/'
        elif pattern == 1:
            prefix = '../RGBNT201/test/NI/'
        elif pattern == 2:
            prefix = '../RGBNT201/test/TI/'
        for i in range(len(img_paths)):
            img_path = prefix + img_paths[i]

            # 加载并调整图像大小
            original_image = Image.open(img_path).resize((128, 256))
            original_image = np.array(original_image)

            # 处理特征图
            feature_map = torch.mean(feature_maps[i], dim=0, keepdim=True)
            feature_map = feature_map.detach().cpu().numpy()
            feature_map = (feature_map - np.min(feature_map)) / np.ptp(feature_map)

            # 转换采样点和参考点为 numpy 格式
            sampled_points = sampled_pointss[i].detach().cpu().numpy()
            reference_points = reference_pointss[i].detach().cpu().numpy()

            # 获取特征图和原始图像的尺寸
            H_feat, W_feat = feature_map.shape[1:]
            H_orig, W_orig = original_image.shape[:2]

            # 计算下采样比例
            scale_x = W_orig / W_feat
            scale_y = H_orig / H_feat

            # 转换坐标到原图系
            sampled_points[:, 1] = (sampled_points[:, 1] + 1) / 2 * (W_feat - 1) * scale_x
            sampled_points[:, 0] = (sampled_points[:, 0] + 1) / 2 * (H_feat - 1) * scale_y
            reference_points[:, 1] = (reference_points[:, 1] + 1) / 2 * (W_feat - 1) * scale_x
            reference_points[:, 0] = (reference_points[:, 0] + 1) / 2 * (H_feat - 1) * scale_y

            # 绘制原图像
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(original_image, aspect='auto')
            ax.set_title(title)

            # 更改采样点样式和箭头样式
            for ref, samp in zip(reference_points, sampled_points):
                ref_y, ref_x = ref
                samp_y, samp_x = samp

                # 参考点用淡蓝色
                ax.scatter(ref_x, ref_y, c='skyblue', s=70, marker='o', edgecolor='black', linewidth=2)  # 较大、黑色边框
                # 目标点用橙色
                ax.scatter(samp_x, samp_y, c='orange', s=70, marker='x', linewidth=12)  # 橙色 "x" 标记

                # 箭头颜色为半透明绿色
                ax.arrow(ref_x, ref_y, samp_x - ref_x, samp_y - ref_y, color='limegreen', alpha=0.7,
                         head_width=4, head_length=6, linewidth=6, length_includes_head=True)

            # 绘制 patch 分隔线
            patch_height, patch_width = patch_size
            for y in range(0, H_orig, patch_height):
                ax.plot([0, W_orig], [y, y], color='white', linewidth=1.5, linestyle='--')  # 水平线
            for x in range(0, W_orig, patch_width):
                ax.plot([x, x], [0, H_orig], color='white', linewidth=1.5, linestyle='--')  # 垂直线

            ax.set_xlim(-1, W_orig)
            ax.set_ylim(H_orig, -1)  # y 轴反转

            # 保存到 writer
            if writer is not None:
                writer.add_figure(f"{title}", fig, global_step=epoch)
            plt.savefig(
                f'../off_vis/{modality[pattern]}/{img_path.split("/")[-1].split(".")[0]}.png')
            # plt.show()
            plt.close(fig)

    def show_cam_on_image(self, img: np.ndarray,
                          mask: np.ndarray,
                          use_rgb: bool = False,
                          colormap: int = cv2.COLORMAP_HOT,
                          image_weight: float = 0.3) -> np.ndarray:
        """ This function overlays the cam mask on the image as an heatmap.
        By default the heatmap is in BGR format.

        :param img: The base image in RGB or BGR format.
        :param mask: The cam mask.
        :param use_rgb: Whether to use an RGB or BGR heatmap, this should be set to True if 'img' is in RGB format.
        :param colormap: The OpenCV colormap to be used.
        :param image_weight: The final result is image_weight * img + (1-image_weight) * mask.
        :returns: The default image with the cam overlay.
        """
        heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
        if use_rgb:
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmap = np.float32(heatmap) / 255

        if np.max(img) > 1:
            raise Exception(
                "The input image should np.float32 in the range [0, 1]")

        if image_weight < 0 or image_weight > 1:
            raise Exception(
                f"image_weight should be in the range [0, 1].\
                    Got: {image_weight}")

        cam = (1 - image_weight) * heatmap + image_weight * img
        cam = cam / np.max(cam)
        return np.uint8(255 * cam)

    @torch.no_grad()
    def visualize_attention_with_distribution(self, attn_map, img_paths, index, reference_pointss, sampled_pointss,
                                              writer=None, epoch=0, title='Attention Heatmap with Distribution',
                                              patch_size=(16, 16), text=''):
        """
        在原始图像上根据 attn_map 选择一个注意力分布，生成热图并覆盖，并显示描述文本
        """
        modality = ['v_RGB', 'v_NIR', 'v_TIR', 't_RGB', 't_NIR', 't_TIR']
        if index == 0 or index == 3:
            prefix = '../RGBNT201/test/RGB/'
            text = text['rgb_text']
        elif index == 1 or index == 4:
            prefix = '../RGBNT201/test/NI/'
            text = text['ni_text']
        elif index == 2 or index == 5:
            prefix = '../RGBNT201/test/TI/'
            text = text['ti_text']

        grid_height = 7  # 高度方向的块数
        grid_width = 3  # 宽度方向的块数

        for i in range(len(img_paths)):
            img_path = prefix + img_paths[i]
            original_image = Image.open(img_path).convert('RGB').resize((128, 256))
            original_image = np.float32(original_image) / 255  # 将图像转换为 [0, 1] 范围的浮点数

            H_orig, W_orig = original_image.shape[:2]  # 获取原图的高度和宽度
            grid_height_size = H_orig // grid_height  # 计算每个网格的高度
            grid_width_size = W_orig // grid_width  # 计算每个网格的宽度

            # 根据 index 选择对应的注意力区域
            if index == 0 or index == 3:
                selected_attn = attn_map[i, index, :21]
            elif index == 1 or index == 4:
                selected_attn = attn_map[i, index, 21:42]
            else:
                selected_attn = attn_map[i, index, 42:]

            selected_attn = selected_attn * 1000 * 2  # 放大权重
            selected_attn = torch.softmax(selected_attn, dim=0)  # 应用 softmax
            selected_attn = selected_attn.detach().cpu().numpy()  # 转换为 NumPy 数组

            # 初始化一个全为零的热图
            heatmap = np.zeros((H_orig, W_orig))

            # 根据网格和注意力权重构建热图
            for row in range(grid_height):
                for col in range(grid_width):
                    # 计算网格区域的起始和结束位置
                    y_start = row * grid_height_size
                    x_start = col * grid_width_size
                    y_end = (row + 1) * grid_height_size if row < grid_height - 1 else H_orig
                    x_end = (col + 1) * grid_width_size if col < grid_width - 1 else W_orig

                    # 获取当前网格的注意力权重
                    weight = selected_attn[row * grid_width + col]

                    # 将该权重值应用到对应的网格区域
                    heatmap[y_start:y_end, x_start:x_end] += weight

            # 将热图与原图叠加
            overlay = self.show_cam_on_image(original_image, heatmap, use_rgb=True, image_weight=0.5)

            # 创建可视化图像
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.imshow(overlay, aspect='auto')
            ax.set_title(f"{title} - Image {i + 1}")

            # 在图像中添加描述文本

            ax.text(
                0.5,  # X 位置 (0到1之间的比例坐标)
                -0.12,  # Y 位置，设置为负值将文本放置在图像下方
                textwrap.fill(text[i], width=80),  # 要显示的文本
                transform=ax.transAxes,  # 使用轴的坐标系统
                color='black',  # 字体颜色
                fontsize=10,  # 字体大小
                ha='center',  # 水平居中
                va='bottom',  # 垂直对齐方式，设置为'底部'对齐
                weight='bold'  # 字体加粗
            )

            # 设置图像的坐标轴
            ax.set_xlim(-1, W_orig)
            ax.set_ylim(H_orig, -1)

            # 如果提供了 writer，则保存到 TensorBoard
            if writer is not None:
                writer.add_figure(f"{title}", fig, global_step=epoch)

            # 保存结果图像
            output_path = f'../attn_vis/{modality[index]}/{img_path.split("/")[-1].split(".")[0]}.png'
            plt.savefig(output_path)
            # plt.show()  # 如果你需要在屏幕上显示图像
            plt.close(fig)

    def off_set_shared(self, data, reference):
        data = einops.rearrange(data, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=3 * self.n_group_channels)
        offset = self.conv_offset(data)
        Hk, Wk = offset.size(2), offset.size(3)
        if self.offset_range_factor > 0:
            offset_range = torch.tensor([1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)], device=data.device).reshape(1, 2, 1, 1)
            offset = offset.tanh().mul(offset_range).mul(self.offset_range_factor)
        offset = einops.rearrange(offset, 'b p h w -> b h w p')
        pos_x = (offset + reference).clamp(-1., +1.)
        pos_y = (offset + reference).clamp(-1., +1.)
        pos_z = (offset + reference).clamp(-1., +1.)
        return pos_x, pos_y, pos_z, Hk, Wk

    def off_set_unshared(self, data, reference):
        x, y, z = data.chunk(3, dim=1)
        x = einops.rearrange(x, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        y = einops.rearrange(y, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        z = einops.rearrange(z, 'b (g c) h w -> (b g) c h w', g=self.n_groups, c=self.n_group_channels)
        offset_r = self.conv_offset_r(x)
        offset_n = self.conv_offset_n(y)
        offset_t = self.conv_offset_t(z)
        Hk, Wk = offset_r.size(2), offset_r.size(3)
        if self.offset_range_factor > 0:
            offset_range = torch.tensor([1.0 / (Hk - 1.0), 1.0 / (Wk - 1.0)], device=data.device).reshape(1, 2, 1, 1)
            offset_r = offset_r.tanh().mul(offset_range).mul(self.offset_range_factor)
            offset_n = offset_n.tanh().mul(offset_range).mul(self.offset_range_factor)
            offset_t = offset_t.tanh().mul(offset_range).mul(self.offset_range_factor)
        offset_r = einops.rearrange(offset_r, 'b p h w -> b h w p')
        offset_n = einops.rearrange(offset_n, 'b p h w -> b h w p')
        offset_t = einops.rearrange(offset_t, 'b p h w -> b h w p')
        pos_x = (offset_r + reference).clamp(-1., +1.)
        pos_y = (offset_n + reference).clamp(-1., +1.)
        pos_z = (offset_t + reference).clamp(-1., +1.)
        return pos_x, pos_y, pos_z, Hk, Wk

    def forward(self, query, x, y, z, writer=None, epoch=None, img_path=None, text=''):
        B, C, H, W = x.size()
        b_, c_, h_, w_ = query.size()
        dtype, device = x.dtype, x.device
        data = torch.cat([x, y, z], dim=1)
        reference = self._get_ref_points(H, W, B, self.ksize, self.stride, dtype, device)
        if self.share_offset:
            pos_x, pos_y, pos_z, Hk, Wk = self.off_set_shared(data, reference)
        else:
            pos_x, pos_y, pos_z, Hk, Wk = self.off_set_unshared(data, reference)
        n_sample = Hk * Wk
        sampled_x = F.grid_sample(
            input=x.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=pos_x[..., (1, 0)],  # y, x -> x, y
            mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg
        sampled_y = F.grid_sample(
            input=y.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=pos_y[..., (1, 0)],  # y, x -> x, y
            mode='bilinear', align_corners=True)
        sampled_z = F.grid_sample(
            input=z.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=pos_z[..., (1, 0)],  # y, x -> x, y
            mode='bilinear', align_corners=True)

        sampled_x = sampled_x.reshape(B, C, 1, n_sample)
        sampled_y = sampled_y.reshape(B, C, 1, n_sample)
        sampled_z = sampled_z.reshape(B, C, 1, n_sample)
        sampled = torch.cat([sampled_x, sampled_y, sampled_z], dim=-1)

        q = self.proj_q(query)
        q = q.reshape(B * self.n_heads, self.n_head_channels, h_ * w_)
        k = self.proj_k(sampled).reshape(B * self.n_heads, self.n_head_channels, 3 * n_sample)
        v = self.proj_v(sampled).reshape(B * self.n_heads, self.n_head_channels, 3 * n_sample)
        attn = torch.einsum('b c m, b c n -> b m n', q, k)  # B * h, HW, Ns
        attn = attn.mul(self.scale)
        attn = F.softmax(attn, dim=2)

        attn = self.attn_drop(attn)
        out = torch.einsum('b m n, b c n -> b c m', attn, v)
        out = out.reshape(B, C, 1, h_ * w_)
        out = self.proj_drop(self.proj_out(out))
        out = query + out
        return out.squeeze(2)

    def forward_woCrossAttn(self, query, x, y, z, writer=None, epoch=None, img_path=None):
        B, C, H, W = x.size()
        dtype, device = x.dtype, x.device
        data = torch.cat([x, y, z], dim=1)
        reference = self._get_ref_points(H, W, B, self.ksize, self.stride, dtype, device)

        if self.share_offset:
            pos_x, pos_y, pos_z, Hk, Wk = self.off_set_shared(data, reference)
        else:
            pos_x, pos_y, pos_z, Hk, Wk = self.off_set_unshared(data, reference)
        n_sample = Hk * Wk
        sampled_x = F.grid_sample(
            input=x.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=pos_x[..., (1, 0)],  # y, x -> x, y
            mode='bilinear', align_corners=True)  # B * g, Cg, Hg, Wg
        sampled_y = F.grid_sample(
            input=y.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=pos_y[..., (1, 0)],  # y, x -> x, y
            mode='bilinear', align_corners=True)
        sampled_z = F.grid_sample(
            input=z.reshape(B * self.n_groups, self.n_group_channels, H, W),
            grid=pos_z[..., (1, 0)],  # y, x -> x, y
            mode='bilinear', align_corners=True)

        sampled_x = sampled_x.reshape(B, C, 1, n_sample)
        sampled_y = sampled_y.reshape(B, C, 1, n_sample)
        sampled_z = sampled_z.reshape(B, C, 1, n_sample)
        input = torch.cat([sampled_x, sampled_y, sampled_z], dim=-1)
        q = self.proj_q(input)
        q = q.reshape(B * self.n_heads, self.n_head_channels, 3 * Hk * Wk)
        k = self.proj_k(input).reshape(B * self.n_heads, self.n_head_channels, 3 * Hk * Wk)
        v = self.proj_v(input).reshape(B * self.n_heads, self.n_head_channels, 3 * Hk * Wk)
        attn = torch.einsum('b c m, b c n -> b m n', q, k)  # B * h, HW, Ns
        attn = attn.mul(self.scale)
        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)
        out = torch.einsum('b m n, b c n -> b c m', attn, v)
        out = out.reshape(B, C, 1, 3 * Hk * Wk)
        out = self.proj_drop(self.proj_out(out))
        out = input + out
        sampled_x, sampled_y, sampled_z = out.chunk(3, dim=-1)

        sampled_x = torch.mean(sampled_x, dim=-1, keepdim=True)
        sampled_y = torch.mean(sampled_y, dim=-1, keepdim=True)
        sampled_z = torch.mean(sampled_z, dim=-1, keepdim=True)

        sampled = torch.cat([sampled_x, sampled_y, sampled_z], dim=-1)
        sampled_2 = torch.cat([sampled, sampled], dim=-1)
        return sampled_2.squeeze(2)

    def forward_woSample_wCrossAttn(self, query, x, y, z, writer=None, epoch=None, img_path=None):
        B, C, H, W = x.size()
        b_, c_, h_, w_ = query.size()
        n_sample = H * W
        sampled_x = x.reshape(B, C, 1, n_sample)
        sampled_y = y.reshape(B, C, 1, n_sample)
        sampled_z = z.reshape(B, C, 1, n_sample)
        sampled = torch.cat([sampled_x, sampled_y, sampled_z], dim=-1)
        q = self.proj_q(query)
        q = q.reshape(B * self.n_heads, self.n_head_channels, h_ * w_)
        k = self.proj_k(sampled).reshape(B * self.n_heads, self.n_head_channels, 3 * n_sample)
        v = self.proj_v(sampled).reshape(B * self.n_heads, self.n_head_channels, 3 * n_sample)
        attn = torch.einsum('b c m, b c n -> b m n', q, k)  # B * h, HW, Ns
        attn = attn.mul(self.scale)
        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)
        out = torch.einsum('b m n, b c n -> b c m', attn, v)
        out = out.reshape(B, C, 1, h_ * w_)
        out = self.proj_drop(self.proj_out(out))
        out = query + out
        return out.squeeze(2)

    def forward_woSample_woCrossAttn(self, query, x, y, z, writer=None, epoch=None, img_path=None):
        B, C, H, W = x.size()
        n_sample = H * W
        sampled_x = x.reshape(B, C, 1, n_sample)
        sampled_y = y.reshape(B, C, 1, n_sample)
        sampled_z = z.reshape(B, C, 1, n_sample)
        input = torch.cat([sampled_x, sampled_y, sampled_z], dim=-1)
        q = self.proj_q(input)
        q = q.reshape(B * self.n_heads, self.n_head_channels, 3 * n_sample)
        k = self.proj_k(input).reshape(B * self.n_heads, self.n_head_channels, 3 * n_sample)
        v = self.proj_v(input).reshape(B * self.n_heads, self.n_head_channels, 3 * n_sample)
        attn = torch.einsum('b c m, b c n -> b m n', q, k)  # B * h, HW, Ns
        attn = attn.mul(self.scale)
        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)
        out = torch.einsum('b m n, b c n -> b c m', attn, v)
        out = out.reshape(B, C, 1, 3 * n_sample)
        out = self.proj_drop(self.proj_out(out))
        out = input + out
        sampled_x, sampled_y, sampled_z = out.chunk(3, dim=-1)

        sampled_x = torch.mean(sampled_x, dim=-1, keepdim=True)
        sampled_y = torch.mean(sampled_y, dim=-1, keepdim=True)
        sampled_z = torch.mean(sampled_z, dim=-1, keepdim=True)

        sampled = torch.cat([sampled_x, sampled_y, sampled_z], dim=-1)
        sampled_2 = torch.cat([sampled, sampled], dim=-1)
        return sampled_2.squeeze(2)

    def forward_woOffset(self, query, x, y, z, writer=None, epoch=None, img_path=None):
        B, C, H, W = x.size()
        b_, c_, h_, w_ = query.size()
        data = torch.cat([x, y, z], dim=1)
        x = self.conv_v(data)
        y = self.conv_n(data)
        z = self.conv_t(data)
        h_new, w_new = x.size(2), x.size(3)
        n_sample = h_new * w_new
        sampled_x = x.reshape(B, C, 1, n_sample)
        sampled_y = y.reshape(B, C, 1, n_sample)
        sampled_z = z.reshape(B, C, 1, n_sample)
        sampled = torch.cat([sampled_x, sampled_y, sampled_z], dim=-1)
        q = self.proj_q(query)
        q = q.reshape(B * self.n_heads, self.n_head_channels, h_ * w_)
        k = self.proj_k(sampled).reshape(B * self.n_heads, self.n_head_channels, 3 * n_sample)
        v = self.proj_v(sampled).reshape(B * self.n_heads, self.n_head_channels, 3 * n_sample)
        attn = torch.einsum('b c m, b c n -> b m n', q, k)  # B * h, HW, Ns
        attn = attn.mul(self.scale)
        attn = F.softmax(attn, dim=2)
        attn = self.attn_drop(attn)
        out = torch.einsum('b m n, b c n -> b c m', attn, v)
        out = out.reshape(B, C, 1, h_ * w_)
        out = self.proj_drop(self.proj_out(out))
        out = query + out
        return out.squeeze(2)


class CDA(nn.Module):

    def __init__(self, window_size=(5, 5), q_size=(16, 8), n_heads=1, n_head_channels=512, n_groups=1, attn_drop=0.,
                 proj_drop=0., stride=2, stride_block=(4, 4),
                 offset_range_factor=5, ksize=5, share=False):
        super(CDA, self).__init__()
        self.q_size = q_size
        self.window_size = window_size
        self.stride_block = stride_block
        self.feat_dim = n_head_channels * n_heads
        self.num_da = self.calculate_num_blocks(q_size, window_size, stride_block)
        self.da_group = nn.ModuleList([
            DAttentionBaseline(
                window_size, n_heads, n_head_channels, n_groups, attn_drop, proj_drop, stride,
                offset_range_factor, ksize, share
            ) for _ in range(self.num_da)
        ])

    def calculate_num_blocks(self, input_size, block_size, stride):
        H, W = input_size  # 输入特征图的高和宽
        block_h, block_w = block_size  # 块的高和宽
        stride_h, stride_w = stride  # 步长的高和宽

        # 计算在高度和宽度方向上的块数
        num_blocks_h = (H - block_h) // stride_h + 1
        num_blocks_w = (W - block_w) // stride_w + 1

        # 总块数
        return num_blocks_h * num_blocks_w

    def split_into_blocks_with_overlap(self, input_tensor, block_size=(4, 4), stride=(4, 4)):
        """
        将输入特征图分割成指定大小的重叠块。

        :param input_tensor: 输入特征图，形状为 (B, C, H, W)
        :param block_size: 块的大小，默认为 (4, 4)，表示高和宽
        :param stride: 块的滑动步长，默认为 (2, 2)，表示高和宽
        :return: 分割后的块，形状为 (B, num_blocks_h, num_blocks_w, C, block_h, block_w)
        """
        B, C, H, W = input_tensor.shape
        block_h, block_w = block_size
        stride_h, stride_w = stride

        # 确保输入的高宽能够支持步长和块大小
        assert H >= block_h and W >= block_w, "Block size should be smaller than the input feature map."

        # 使用 unfold 操作实现滑动窗口式的分割
        # unfold(2) 是在高度 H 维度上分块, unfold(3) 是在宽度 W 维度上分块
        unfolded = input_tensor.unfold(2, block_h, stride_h).unfold(3, block_w, stride_w)

        # 结果的形状为 (B, C, num_blocks_h, num_blocks_w, block_h, block_w)
        # 需要将通道 C 维度移到最后，符合 (B, num_blocks_h, num_blocks_w, C, block_h, block_w)
        unfolded = unfolded.permute(0, 2, 3, 1, 4, 5).contiguous()

        return unfolded

    def visualize_blocks(self, input_tensor, block_size=(4, 4)):
        """
        可视化输入特征图的分块结果。

        :param input_tensor: 输入特征图，形状为 (B, C, H, W)
        :param block_size: 块的大小，默认为 (4, 4)，表示高和宽
        """
        # 分块
        blocks = self.split_into_blocks(input_tensor, block_size)

        B, num_blocks_h, num_blocks_w, C, block_h, block_w = blocks.shape
        fig, axes = plt.subplots(num_blocks_h, num_blocks_w, figsize=(10, 10))

        for i in range(num_blocks_h):
            for j in range(num_blocks_w):
                # 选择第一个batch中的一个块进行可视化
                block = blocks[0, i, j].detach().cpu().numpy()

                # 将块中的通道数据降维到单通道（选择第一个通道或使用平均值）
                block_image = block.mean(axis=0)  # 这里用通道平均值进行可视化
                axes[i, j].imshow(block_image, cmap='gray')
                axes[i, j].axis('off')

        plt.suptitle('Visualization of Blocks', fontsize=16)
        plt.show()

    def forward(self, x, y, z, boss, writer=None, epoch=None, img_path=None, texts=''):
        x = x.reshape(x.size(0), self.q_size[0], self.q_size[1], -1).permute(0, 3, 1, 2)
        y = y.reshape(y.size(0), self.q_size[0], self.q_size[1], -1).permute(0, 3, 1, 2)
        z = z.reshape(z.size(0), self.q_size[0], self.q_size[1], -1).permute(0, 3, 1, 2)
        x_blocks = self.split_into_blocks_with_overlap(x, self.window_size, self.stride_block).flatten(1, 2)
        y_blocks = self.split_into_blocks_with_overlap(y, self.window_size, self.stride_block).flatten(1, 2)
        z_blocks = self.split_into_blocks_with_overlap(z, self.window_size, self.stride_block).flatten(1, 2)
        boss = boss.permute(0, 2, 1).unsqueeze(-2)
        query_cash = []
        for i in range(self.num_da):
            query_cash.append(
                self.da_group[i](boss, x_blocks[:, i], y_blocks[:, i], z_blocks[:, i], writer=writer, epoch=epoch,
                                 img_path=img_path, text=texts).squeeze(-1))
        fea = query_cash[0].permute(0, 2, 1)
        vision = torch.flatten(fea[:, :3], start_dim=1, end_dim=2)
        text = torch.flatten(fea[:, 3:], start_dim=1, end_dim=2)
        return vision, text
