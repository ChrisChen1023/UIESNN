from spikingjelly.activation_based import functional, layer
import torch
import torch.nn as nn
import torch.nn.functional as F

v_th = 0.15
alpha = 1 / (2 ** 0.5)
decay = 0.25


class MultiSpike4(nn.Module):
    class quant4(torch.autograd.Function):
        @staticmethod
        def forward(ctx, input):
            ctx.save_for_backward(input)
            quantized = torch.round(torch.clamp(input, min=0, max=4))
            return quantized / 4.0

        @staticmethod
        def backward(ctx, grad_output):
            input, = ctx.saved_tensors
            grad_input = grad_output.clone()
            grad_input[input < 0] = 0
            grad_input[input > 4] = 0
            return grad_input / 4.0

    def forward(self, x):
        return self.quant4.apply(x)


class mem_update(nn.Module):
    def __init__(self):
        super(mem_update, self).__init__()
        self.qtrick = MultiSpike4()

    def forward(self, x):
        spike = torch.zeros_like(x[0]).to(x.device)
        output = torch.zeros_like(x)
        mem_old = 0
        time_window = x.shape[0]
        for i in range(time_window):
            if i >= 1:
                mem = (mem_old - spike.detach()) * decay + x[i]
            else:
                mem = x[i]
            spike = self.qtrick(mem)
            mem_old = mem.clone()
            output[i] = spike
        return output


class MultiScalePoolingLIFBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        functional.set_step_mode(self, step_mode='m')

        self.channel_per_group = in_channels // 4

        self.pool_1 = nn.AvgPool2d(kernel_size=4, stride=4)
        self.pool_2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.pool_3 = nn.AdaptiveAvgPool2d((1, 1))

        self.lif_0 = mem_update()
        self.conv_0 = layer.Conv2d(self.channel_per_group, self.channel_per_group, kernel_size=1, bias=False, step_mode='m')
        self.bn_0 = layer.ThresholdDependentBatchNorm2d(num_features=self.channel_per_group, alpha=alpha, v_th=v_th, affine=True)

        self.lif_1 = mem_update()
        self.conv_1 = layer.Conv2d(self.channel_per_group, self.channel_per_group, kernel_size=1, bias=False, step_mode='m')
        self.bn_1 = layer.ThresholdDependentBatchNorm2d(num_features=self.channel_per_group, alpha=alpha, v_th=v_th, affine=True)

        self.lif_2 = mem_update()
        self.conv_2 = layer.Conv2d(self.channel_per_group, self.channel_per_group, kernel_size=1, bias=False, step_mode='m')
        self.bn_2 = layer.ThresholdDependentBatchNorm2d(num_features=self.channel_per_group, alpha=alpha, v_th=v_th, affine=True)

        self.lif_3 = mem_update()
        self.conv_3 = layer.Conv2d(self.channel_per_group, self.channel_per_group, kernel_size=1, bias=False, step_mode='m')
        self.bn_3 = layer.ThresholdDependentBatchNorm2d(num_features=self.channel_per_group, alpha=alpha, v_th=v_th, affine=True)

        self.conv_m1 = layer.Conv2d(self.channel_per_group * 2, self.channel_per_group, kernel_size=1, bias=False, step_mode='m')
        self.bn_m1 = layer.ThresholdDependentBatchNorm2d(num_features=self.channel_per_group, alpha=alpha, v_th=v_th, affine=True)

        self.conv_m2 = layer.Conv2d(self.channel_per_group * 2, self.channel_per_group, kernel_size=1, bias=False, step_mode='m')
        self.bn_m2 = layer.ThresholdDependentBatchNorm2d(num_features=self.channel_per_group, alpha=alpha, v_th=v_th, affine=True)

        self.conv_m3 = layer.Conv2d(self.channel_per_group * 2, self.channel_per_group, kernel_size=1, bias=False, step_mode='m')
        self.bn_m3 = layer.ThresholdDependentBatchNorm2d(num_features=self.channel_per_group, alpha=alpha, v_th=v_th, affine=True)

        self.theta1 = nn.Parameter(torch.ones(1))
        self.theta2 = nn.Parameter(torch.ones(1))
        self.theta3 = nn.Parameter(torch.ones(1))
        self.theta4 = nn.Parameter(torch.ones(1))

        self.final_conv = layer.Conv2d(self.channel_per_group * 4, in_channels, kernel_size=3, padding=1, bias=False, step_mode='m')
        self.final_bn = layer.ThresholdDependentBatchNorm2d(num_features=in_channels, alpha=alpha, v_th=v_th, affine=True)

    def forward(self, x):
        T, B, C, H, W = x.shape

        x_splits = torch.split(x, self.channel_per_group, dim=2)

        x0_flat = x_splits[0].reshape(T * B, self.channel_per_group, H, W)
        x0_processed = x0_flat
        x0_processed = x0_processed.reshape(T, B, self.channel_per_group, H, W)
        x0_processed = self.lif_0(x0_processed)
        x0_processed = self.conv_0(x0_processed)
        F1 = self.bn_0(x0_processed)

        x1_flat = x_splits[1].reshape(T * B, self.channel_per_group, H, W)
        x1_pooled = self.pool_1(x1_flat)
        x1_pooled = x1_pooled.reshape(T, B, self.channel_per_group, x1_pooled.shape[2], x1_pooled.shape[3])
        x1_pooled = self.lif_1(x1_pooled)
        x1_pooled = self.conv_1(x1_pooled)
        x1_pooled = self.bn_1(x1_pooled)
        x1_flat_processed = x1_pooled.reshape(T * B, self.channel_per_group, x1_pooled.shape[3], x1_pooled.shape[4])
        x1_upsampled = F.interpolate(x1_flat_processed, size=(H, W), mode='bilinear', align_corners=False)
        F2 = x1_upsampled.reshape(T, B, self.channel_per_group, H, W)

        x2_flat = x_splits[2].reshape(T * B, self.channel_per_group, H, W)
        x2_pooled = self.pool_2(x2_flat)
        x2_pooled = x2_pooled.reshape(T, B, self.channel_per_group, x2_pooled.shape[2], x2_pooled.shape[3])
        x2_pooled = self.lif_2(x2_pooled)
        x2_pooled = self.conv_2(x2_pooled)
        x2_pooled = self.bn_2(x2_pooled)
        x2_flat_processed = x2_pooled.reshape(T * B, self.channel_per_group, x2_pooled.shape[3], x2_pooled.shape[4])
        x2_upsampled = F.interpolate(x2_flat_processed, size=(H, W), mode='bilinear', align_corners=False)
        F3 = x2_upsampled.reshape(T, B, self.channel_per_group, H, W)

        x3_flat = x_splits[3].reshape(T * B, self.channel_per_group, H, W)
        x3_pooled = self.pool_3(x3_flat)
        x3_pooled = x3_pooled.reshape(T, B, self.channel_per_group, 1, 1)
        x3_pooled = self.lif_3(x3_pooled)
        x3_pooled = self.conv_3(x3_pooled)
        x3_pooled = self.bn_3(x3_pooled)
        x3_flat_processed = x3_pooled.reshape(T * B, self.channel_per_group, 1, 1)
        x3_upsampled = F.interpolate(x3_flat_processed, size=(H, W), mode='bilinear', align_corners=False)
        F4 = x3_upsampled.reshape(T, B, self.channel_per_group, H, W)

        concat_f1_f2 = torch.cat([F1, F2], dim=2)
        M1 = self.conv_m1(concat_f1_f2)
        M1 = self.bn_m1(M1)

        concat_f1_f3 = torch.cat([F1, F3], dim=2)
        M2 = self.conv_m2(concat_f1_f3)
        M2 = self.bn_m2(M2)

        concat_f1_f4 = torch.cat([F1, F4], dim=2)
        M3 = self.conv_m3(concat_f1_f4)
        M3 = self.bn_m3(M3)

        F1_scaled = self.theta1 * F1
        M1_scaled = self.theta2 * M1
        M2_scaled = self.theta3 * M2
        M3_scaled = self.theta4 * M3

        final_concat = torch.cat([F1_scaled, M1_scaled, M2_scaled, M3_scaled], dim=2)

        output = self.final_conv(final_concat)
        output = self.final_bn(output)

        return output


class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=32, spike_mode="lif", LayerNorm_type='WithBias', bias=False, T=4):
        super(OverlapPatchEmbed, self).__init__()
        functional.set_step_mode(self, step_mode='m')
        self.proj = layer.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias, step_mode='m')

    def forward(self, x):
        x = self.proj(x)
        return x


class Spiking_Residual_Block(nn.Module):
    def __init__(self, dim):
        super(Spiking_Residual_Block, self).__init__()
        functional.set_step_mode(self, step_mode='m')

        self.lif_1 = mem_update()
        self.conv1 = layer.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False, step_mode='m')
        self.bn1 = layer.ThresholdDependentBatchNorm2d(num_features=dim, alpha=alpha, v_th=v_th, affine=True)

        self.high_freq_scale_1 = nn.Parameter(torch.ones(1))
        self.low_freq_scale_1 = nn.Parameter(torch.ones(1))

        self.lif_2 = MultiScalePoolingLIFBlock(in_channels=dim)
        self.conv2 = layer.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=False, step_mode='m')
        self.bn2 = layer.ThresholdDependentBatchNorm2d(num_features=dim, alpha=alpha, v_th=v_th * 0.2, affine=True)

        self.shortcut = nn.Sequential(
            layer.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1,
                         bias=False, step_mode='m'),
            layer.ThresholdDependentBatchNorm2d(num_features=dim, alpha=alpha,
                                                v_th=v_th, affine=True),
        )

        self.attn = layer.MultiDimensionalAttention(T=4, reduction_t=4, reduction_c=16, kernel_size=3, C=dim)

    def forward(self, x):
        x_h_1 = self.lif_1(x)
        x_l_1 = x - x_h_1

        x_h_1_scaled = self.high_freq_scale_1 * x_h_1
        x_l_1_scaled = self.low_freq_scale_1 * x_l_1

        x_enhanced_1 = x * x_h_1

        combined_features_1 = x_h_1_scaled + x_l_1_scaled + x_enhanced_1

        out = self.conv1(combined_features_1)
        out = self.bn1(out)

        multi_scale_features = self.lif_2(out)

        out = self.conv2(multi_scale_features)
        out = self.bn2(out)

        shortcut = torch.clone(x)
        out = out + self.shortcut(shortcut)
        out = self.attn(out) + shortcut
        return out


class DownSampling(nn.Module):
    def __init__(self, dim):
        super(DownSampling, self).__init__()
        functional.set_step_mode(self, step_mode='m')

        self.lif = mem_update()
        self.conv = layer.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1, step_mode='m', bias=False)
        self.bn = layer.ThresholdDependentBatchNorm2d(alpha=alpha, v_th=v_th, num_features=dim * 2, affine=True)

    def forward(self, x):
        x = self.lif(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


class UpSampling(nn.Module):
    def __init__(self, dim):
        super(UpSampling, self).__init__()
        self.scale_factor = 2

        self.lif = mem_update()
        self.conv = layer.Conv2d(dim, dim // 2, kernel_size=3, stride=1, padding=1, step_mode='m', bias=False)
        self.bn = layer.ThresholdDependentBatchNorm2d(alpha=alpha, v_th=v_th, num_features=dim // 2, affine=True)

    def forward(self, input):
        temp = torch.zeros((input.shape[0], input.shape[1], input.shape[2], input.shape[3] * self.scale_factor,
                            input.shape[4] * self.scale_factor)).cuda()
        output = []
        for i in range(input.shape[0]):
            temp[i] = F.interpolate(input[i], scale_factor=self.scale_factor, mode='bilinear')
            output.append(temp[i])
        out = torch.stack(output, dim=0)

        out = self.lif(out)
        out = self.conv(out)
        out = self.bn(out)
        return out


class UIESNN(nn.Module):
    def __init__(self, inp_channels=3, out_channels=3, dim=24, en_num_blocks=[4, 4, 6, 6], de_num_blocks=[4, 4, 6, 6],
                 bias=False, T=4):
        super(UIESNN, self).__init__()

        functional.set_backend(self, backend='cupy')
        functional.set_step_mode(self, step_mode='m')

        self.T = T
        self.patch_embed = OverlapPatchEmbed(in_c=inp_channels, embed_dim=dim, T=T)

        self.input_embed_level2_conv3x3 = layer.Conv2d(inp_channels, dim * 2, kernel_size=3, stride=1, padding=1, bias=False, step_mode='m')
        self.input_embed_level2_conv1x1 = layer.Conv2d(dim * 2, dim * 2, kernel_size=1, bias=False, step_mode='m')
        self.input_concat_level2 = layer.Conv2d(dim * 2 * 2, dim * 2, kernel_size=1, bias=False, step_mode='m')

        self.input_embed_level3_conv3x3 = layer.Conv2d(inp_channels, dim * 4, kernel_size=3, stride=1, padding=1, bias=False, step_mode='m')
        self.input_embed_level3_conv1x1 = layer.Conv2d(dim * 4, dim * 4, kernel_size=1, bias=False, step_mode='m')
        self.input_concat_level3 = layer.Conv2d(dim * 4 * 2, dim * 4, kernel_size=1, bias=False, step_mode='m')

        self.encoder_level1 = nn.Sequential(*[
            Spiking_Residual_Block(dim=int(dim * 1)) for i in range(en_num_blocks[0])])

        self.down1_2 = DownSampling(dim)
        self.encoder_level2 = nn.Sequential(*[
            Spiking_Residual_Block(dim=int(dim * 2 ** 1)) for i in range(en_num_blocks[1])])

        self.down2_3 = DownSampling(int(dim * 2 ** 1))
        self.encoder_level3 = nn.Sequential(*[
            Spiking_Residual_Block(dim=int(dim * 2 ** 2)) for i in range(en_num_blocks[2])])

        self.decoder_level3 = nn.Sequential(*[
            Spiking_Residual_Block(dim=int(dim * 2 ** 2)) for i in range(de_num_blocks[2])])

        self.up3_2 = UpSampling(int(dim * 2 ** 2))

        self.lif_level2 = mem_update()
        self.reduce_conv_level2 = layer.Conv2d(int(dim * 2 ** 2), int(dim * 2 ** 1), kernel_size=1, bias=bias, step_mode='m')
        self.reduce_bn_level2 = layer.ThresholdDependentBatchNorm2d(num_features=int(dim * 2 ** 1), alpha=alpha, v_th=v_th)

        self.decoder_level2 = nn.Sequential(*[
            Spiking_Residual_Block(dim=int(dim * 2 ** 1)) for i in range(de_num_blocks[1])])

        self.up2_1 = UpSampling(int(dim * 2 ** 1))

        self.lif_level1 = mem_update()
        self.reduce_conv_level1 = layer.Conv2d(int(dim * 2 ** 1), int(dim * 2 ** 0), kernel_size=1, bias=bias, step_mode='m')
        self.reduce_bn_level1 = layer.ThresholdDependentBatchNorm2d(num_features=int(dim * 2 ** 0), alpha=alpha, v_th=v_th)

        self.decoder_level1 = nn.Sequential(*[
            Spiking_Residual_Block(dim=int(dim * 2 ** 0)) for i in range(de_num_blocks[0])])

        self.output_level3 = nn.Sequential(
            nn.Conv2d(in_channels=int(dim * 2 ** 2), out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        )
        self.output_level2 = nn.Sequential(
            nn.Conv2d(in_channels=int(dim * 2 ** 1), out_channels=out_channels, kernel_size=3, stride=1, padding=1)
        )

        self.output = nn.Sequential(
            nn.Conv2d(in_channels=int(dim * 2 ** 0), out_channels=out_channels, kernel_size=3, stride=1,
                      padding=1)
        )

    def forward(self, inp_img):
        short = inp_img.clone()
        if len(inp_img.shape) < 5:
            inp_img = (inp_img.unsqueeze(0)).repeat(self.T, 1, 1, 1, 1)

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)

        T, B, C, H, W = inp_img.shape
        inp_img_flat = inp_img.reshape(T * B, C, H, W)
        inp_img_level2 = F.interpolate(inp_img_flat, size=(H // 2, W // 2), mode='bilinear', align_corners=False)
        inp_img_level2 = inp_img_level2.reshape(T, B, C, H // 2, W // 2)

        inp_img_level2_embed = self.input_embed_level2_conv3x3(inp_img_level2)
        inp_img_level2_embed = self.input_embed_level2_conv1x1(inp_img_level2_embed)

        inp_enc_level2_concat = torch.cat([inp_enc_level2, inp_img_level2_embed], dim=2)
        inp_enc_level2 = self.input_concat_level2(inp_enc_level2_concat)

        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)

        inp_img_level3 = F.interpolate(inp_img_flat, size=(H // 4, W // 4), mode='bilinear', align_corners=False)
        inp_img_level3 = inp_img_level3.reshape(T, B, C, H // 4, W // 4)

        inp_img_level3_embed = self.input_embed_level3_conv3x3(inp_img_level3)
        inp_img_level3_embed = self.input_embed_level3_conv1x1(inp_img_level3_embed)

        inp_enc_level3_concat = torch.cat([inp_enc_level3, inp_img_level3_embed], dim=2)
        inp_enc_level3 = self.input_concat_level3(inp_enc_level3_concat)

        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        out_dec_level3 = self.decoder_level3(out_enc_level3)

        output_level3 = self.output_level3(out_dec_level3.mean(0))
        output_level3 = F.interpolate(output_level3, size=(H, W), mode='bilinear', align_corners=False) + short

        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], dim=2)

        inp_dec_level2 = self.lif_level2(inp_dec_level2)
        inp_dec_level2 = self.reduce_conv_level2(inp_dec_level2)
        inp_dec_level2 = self.reduce_bn_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        output_level2 = self.output_level2(out_dec_level2.mean(0))
        output_level2 = F.interpolate(output_level2, size=(H, W), mode='bilinear', align_corners=False) + short

        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], dim=2)

        inp_dec_level1 = self.lif_level1(inp_dec_level1)
        inp_dec_level1 = self.reduce_conv_level1(inp_dec_level1)
        inp_dec_level1 = self.reduce_bn_level1(inp_dec_level1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)

        output_final = (self.output(out_dec_level1.mean(0))) + short

        return output_level3, output_level2, output_final


model = UIESNN()
