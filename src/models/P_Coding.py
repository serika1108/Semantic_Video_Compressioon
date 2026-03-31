import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from .video_net import ME_Spynet, GDN, flow_warp, ResBlock, ResBlock_LeakyReLU_0_Point_1
from src.entropy_models.video_entropy_models import BitEstimator, GaussianEncoder
from src.models.stream_helper import get_downsampled_shape
from src.layers.layers import MaskedConv2d, subpel_conv3x3


# 卷积尺度变化的公式是 out = [(in + 2Padding - kernel_size) / stride] + 1
# kernel=3，stride=1，padding=1时，图像的长宽不变
# kernel=3，stride=2，padding=1时，图像的长宽减半
# kernel=5，stride=2，padding=2时，图像的长宽减半
class DCVC_net(nn.Module):
    def __init__(self):
        super().__init__()
        # 运动特征的编码器、解码器的输出通道设置为128
        out_channel_mv = 128

        out_channel_M = 96
        out_channel_N = 64

        self.out_channel_mv = out_channel_mv
        self.out_channel_N = out_channel_N
        self.out_channel_M = out_channel_M

        # 超先验参数z的编码过程。和compressAI的实现一样，用MLP拟合概率分布
        self.bitEstimator_z = BitEstimator(out_channel_N)
        self.bitEstimator_z_mv = BitEstimator(out_channel_N)

        # 特征提取，用于对x_{t-1}的帧提取特征，方便对运动信息warp
        self.feature_extract = nn.Sequential(
            nn.Conv2d(3, out_channel_N, 3, stride=1, padding=1),
            ResBlock(out_channel_N, out_channel_N, 3),
        )

        # 对warp后的图像特征做refine
        self.context_refine = nn.Sequential(
            ResBlock(out_channel_N, out_channel_N, 3),
            nn.Conv2d(out_channel_N, out_channel_N, 3, stride=1, padding=1),
        )

        self.gaussian_encoder = GaussianEncoder()

        # 动作特征编码器，这里编码的是光流，因此输入的通道维度是2
        self.mvEncoder = nn.Sequential(
            nn.Conv2d(2, out_channel_mv, 3, stride=2, padding=1),
            GDN(out_channel_mv),
            nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1),
            GDN(out_channel_mv),
            nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1),
            GDN(out_channel_mv),
            nn.Conv2d(out_channel_mv, out_channel_mv, 3, stride=2, padding=1),
        )

        # 运动特征解码分两部分，第一部分是对正常的运动特征解码
        self.mvDecoder_part1 = nn.Sequential(
            nn.ConvTranspose2d(out_channel_mv, out_channel_mv, 3,
                               stride=2, padding=1, output_padding=1),
            GDN(out_channel_mv, inverse=True),
            nn.ConvTranspose2d(out_channel_mv, out_channel_mv, 3,
                               stride=2, padding=1, output_padding=1),
            GDN(out_channel_mv, inverse=True),
            nn.ConvTranspose2d(out_channel_mv, out_channel_mv, 3,
                               stride=2, padding=1, output_padding=1),
            GDN(out_channel_mv, inverse=True),
            nn.ConvTranspose2d(out_channel_mv, 2, 3,
                               stride=2, padding=1, output_padding=1),
        )
        # 运动特征解码的第二部分，这里将第一部分的输出和像素图像的参考帧x_{t-1}在通道维拼接后继续解码
        self.mvDecoder_part2 = nn.Sequential(
            nn.Conv2d(5, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(negative_slope=0.1),
            nn.Conv2d(64, 2, 3, stride=1, padding=1),
        )

        # 上下文编码
        self.contextualEncoder = nn.Sequential(
            nn.Conv2d(out_channel_N+3, out_channel_N, 5, stride=2, padding=2),
            GDN(out_channel_N),
            ResBlock_LeakyReLU_0_Point_1(out_channel_N),
            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
            GDN(out_channel_N),
            ResBlock_LeakyReLU_0_Point_1(out_channel_N),
            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
            GDN(out_channel_N),
            nn.Conv2d(out_channel_N, out_channel_M, 5, stride=2, padding=2),
        )

        # 上下文解码分为两部分，因为要添加运动特征。第一部分是正常的解码
        self.contextualDecoder_part1 = nn.Sequential(
            subpel_conv3x3(out_channel_M, out_channel_N, 2),
            GDN(out_channel_N, inverse=True),
            subpel_conv3x3(out_channel_N, out_channel_N, 2),
            GDN(out_channel_N, inverse=True),
            ResBlock_LeakyReLU_0_Point_1(out_channel_N),
            subpel_conv3x3(out_channel_N, out_channel_N, 2),
            GDN(out_channel_N, inverse=True),
            ResBlock_LeakyReLU_0_Point_1(out_channel_N),
            subpel_conv3x3(out_channel_N, out_channel_N, 2),
        )
        # 第二部分添加了运动特征，也就是warp后refine的结果。context_refine的输出维度为out_channel_N，因此这里输入维度要乘2
        self.contextualDecoder_part2 = nn.Sequential(
            nn.Conv2d(out_channel_N*2, out_channel_N, 3, stride=1, padding=1),
            ResBlock(out_channel_N, out_channel_N, 3),
            ResBlock(out_channel_N, out_channel_N, 3),
            nn.Conv2d(out_channel_N, 3, 3, stride=1, padding=1),
        )

        # 超先验z的编解码模块
        self.priorEncoder = nn.Sequential(
            nn.Conv2d(out_channel_M, out_channel_N, 3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
        )

        self.priorDecoder = nn.Sequential(
            nn.ConvTranspose2d(out_channel_N, out_channel_M, 5,
                               stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(out_channel_M, out_channel_M, 5,
                               stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(out_channel_M, out_channel_M, 3, stride=1, padding=1)
        )

        # 运动特征的超先验模块
        self.mvpriorEncoder = nn.Sequential(
            nn.Conv2d(out_channel_mv, out_channel_N, 3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
        )

        self.mvpriorDecoder = nn.Sequential(
            nn.ConvTranspose2d(out_channel_N, out_channel_N, 5,
                               stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(out_channel_N, out_channel_N * 3 // 2, 5,
                               stride=2, padding=2, output_padding=1),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(out_channel_N * 3 // 2, out_channel_mv*2, 3, stride=1, padding=1)
        )

        # 用在自回归中，计算一个高斯分布的均值和方差。均值用于量化，方差用于熵编码
        # 这里的写法有两个目的，第一个是保证输出维度是2的倍数，代表均值和方差
        # 第二个是乘一个数后再除，能确保通道的比例是12:10:8，同时，可以自由更换out_channel_M的维度
        self.entropy_parameters = nn.Sequential(
            nn.Conv2d(out_channel_M * 12 // 3, out_channel_M * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_M * 10 // 3, out_channel_M * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_M * 8 // 3, out_channel_M * 6 // 3, 1),
        )

        # 掩膜卷积，卷积核的形状如下
        # 1 1 1 1 1
        # 1 1 1 1 1
        # 1 1 0 0 0
        # 0 0 0 0 0
        # 0 0 0 0 0
        # 掩码卷积的核心作用是，根据上下文，预测当前位置的特征
        self.auto_regressive = MaskedConv2d(
            out_channel_M, 2 * out_channel_M, kernel_size=5, padding=2, stride=1
        )

        self.auto_regressive_mv = MaskedConv2d(
            out_channel_mv, 2 * out_channel_mv, kernel_size=5, padding=2, stride=1
        )

        # 用在动作特征的自回归部分
        self.entropy_parameters_mv = nn.Sequential(
            nn.Conv2d(out_channel_mv * 12 // 3, out_channel_mv * 10 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_mv * 10 // 3, out_channel_mv * 8 // 3, 1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(out_channel_mv * 8 // 3, out_channel_mv * 6 // 3, 1),
        )

        self.temporalPriorEncoder = nn.Sequential(
            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
            GDN(out_channel_N),
            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
            GDN(out_channel_N),
            nn.Conv2d(out_channel_N, out_channel_N, 5, stride=2, padding=2),
            GDN(out_channel_N),
            nn.Conv2d(out_channel_N, out_channel_M, 5, stride=2, padding=2),
        )

        self.opticFlow = ME_Spynet()

    def motioncompensation(self, ref, mv):
        """
        运动补偿模块
        """
        # 提取参考帧的特征，这里是上一轮的解码的像素图像x_{t-1}。虽然是提取特征，但是长宽尺寸未发生变化，只有通道数增加了
        ref_feature = self.feature_extract(ref)
        # 根据运动特征和参考帧，在特征阈做warp
        prediction_init = flow_warp(ref_feature, mv)
        # 在warp后做一次refine
        context = self.context_refine(prediction_init)

        return context

    def mv_refine(self, ref, mv):
        # 运动特征的解码包含两部分，这里还加了个参考帧
        return self.mvDecoder_part2(torch.cat((mv, ref), 1)) + mv

    def quantize(self, inputs, mode, means=None):
        # 推理阶段的量化，这里加了个均值操作，减去均值后量化，相对于归一化操作
        assert(mode == "dequantize")
        outputs = inputs.clone()
        outputs -= means
        outputs = torch.round(outputs)
        outputs += means
        return outputs

    def feature_probs_based_sigma(self, feature, mean, sigma):
        """
        用拉普拉斯分布计算熵编码，估计比特率。返回总编码的比特数和概率分布
        """
        # 对数据量化并用均值归一化。归一化的优势是减少数据分布的宽度，让熵编码在数值上更稳定
        outputs = self.quantize(feature, "dequantize", mean)
        values = outputs - mean
        # 生成拉普拉斯分布
        mu = torch.zeros_like(sigma)
        sigma = sigma.clamp(1e-5, 1e10)
        gaussian = torch.distributions.laplace.Laplace(mu, sigma)
        # 将连续概率分布转换成离散概率分布，用于对应量化后的符号的概率
        probs = gaussian.cdf(values + 0.5) - gaussian.cdf(values - 0.5)
        total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-5) / math.log(2.0), 0, 50))
        return total_bits, probs

    def estrate_bits_z(self, z):
        # 先验z的比特估计
        prob = self.bitEstimator_z(z + 0.5) - self.bitEstimator_z(z - 0.5)
        total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))
        return total_bits, prob

    def estrate_bits_z_mv(self, z_mv):
        # 运动信息对应的先验z的比特估计
        prob = self.bitEstimator_z_mv(z_mv + 0.5) - self.bitEstimator_z_mv(z_mv - 0.5)
        total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-5) / math.log(2.0), 0, 50))
        return total_bits, prob

    def update(self, force=False):
        # 更新熵编码的概率分布表
        self.bitEstimator_z_mv.update(force=force)
        self.bitEstimator_z.update(force=force)
        self.gaussian_encoder.update(force=force)

    def encode_decode(self, ref_frame, input_image, output_path):
        # encoded里面就已经返回了重构的图像，但是直接拿量化的结果重构的。decoded是把比特流解码后重构的
        encoded = self.encode(ref_frame, input_image, output_path)
        decoded = self.decode(ref_frame, output_path)
        # 用比特流解码的图像代替encoded里重构的图像
        encoded['recon_image'] = decoded
        return encoded

    def encode(self, ref_frame, input_image, output_path):
        from src.models.stream_helper import encode_p
        N, C, H, W = ref_frame.size()
        compressed = self.compress(ref_frame, input_image)
        mv_y_string = compressed['mv_y_string']
        mv_z_string = compressed['mv_z_string']
        y_string = compressed['y_string']
        z_string = compressed['z_string']
        # 将编码器的比特流以自定义的格式写入到文件中
        encode_p(H, W, mv_y_string, mv_z_string, y_string, z_string, output_path)
        return {
            'bpp_mv_y': compressed['bpp_mv_y'],
            'bpp_mv_z': compressed['bpp_mv_z'],
            'bpp_y': compressed['bpp_y'],
            'bpp_z': compressed['bpp_z'],
            'bpp': compressed['bpp'],
        }

    def decode(self, ref_frame, input_path):
        from src.models.stream_helper import decode_p
        # 从文件中读取比特流
        height, width, mv_y_string, mv_z_string, y_string, z_string = decode_p(input_path)
        return self.decompress(ref_frame, mv_y_string, mv_z_string,
                               y_string, z_string, height, width)

    def compress_ar(self, y, kernel_size, context_prediction, params, entropy_parameters):
        """
        对y做自回归熵编码
        context_prediction是掩膜卷积
        params是其他熵编码所依赖的参数，例如超先验z和运动补偿得到的上下文
        entropy_parameters是一个卷积网络，用于计算当前位置的均值和方差
        自回归是通过for循环完成的，掩膜卷积只是用来计算一个自回归的参数。自回归的参数和params共同作为熵编码所需的参数
        基本思路是先做量化，再做熵编码。量化的时候用的是自回归的形式，将自回归的量化结果用作熵编码
        """
        kernel_size = 5
        # 这个padding是为了编码第一个位置的像素而设置的，第一个位置需要有上下文，用padding代替
        padding = (kernel_size - 1) // 2

        height = y.size(2)
        width = y.size(3)

        # 存储初始值y
        y_hat = F.pad(y, (padding, padding, padding, padding))
        # 存储量化后的y
        y_q = torch.zeros_like(y)
        # 存储每个位置的概率分布参数sigma，用于拉普拉斯的熵编码
        y_scales = torch.zeros_like(y)

        # 自回归的形式扫描整个图像。注意，自回归是通过for循环实现的，而不是掩膜卷积
        # 掩膜卷积只能卷一次，但是自回归要拿之前的编码结果取编码后面的值，所以要用for循环实现
        for h in range(height):
            for w in range(width):
                # 取5*5的块，在这个块内做掩膜卷积，实际上这里只处理(h,w)位置的特征值
                # 取0:1是为了保留batch维度。后续卷积输入需要是[N,C,H,W]，因此这里保留了第一维
                # 因为是一帧一帧图像处理的，所以这个batch只有一个图像，所以是0:1
                y_crop = y_hat[0:1, :, h:h + kernel_size, w:w + kernel_size]
                # 用掩膜卷积预测y_crop每个点的上下文特征
                ctx_p = F.conv2d(y_crop, context_prediction.weight, bias=context_prediction.bias)

                # 取对应位置的params，和掩膜卷积的结果合并后，共同送入熵模块，生成概率分布的参数
                # 对于运动特征，params仅对应超先验z，对于帧，params对应的超先验z和动作补偿后得到的context
                p = params[0:1, :, h:h + 1, w:w + 1]
                # 辅助的参数params和掩膜卷积预测的概率分布合并，生成编码的均值和方差。用神经网络预测均值和方差
                gaussian_params = entropy_parameters(torch.cat((p, ctx_p), dim=1))
                means_hat, scales_hat = gaussian_params.chunk(2, 1)

                # 从5*5的块中截取中间位置对应的值，对中间位置的值做均值量化，因为中间位置才是真正编码的数值
                y_crop = y_crop[0:1, :, padding:padding+1, padding:padding+1]
                # 减去均值后量化对应的y
                y_crop_q = torch.round(y_crop - means_hat)
                # 用量化后的值更新y_hat，加padding是因为y_hat是被padding的，取原图像特征的坐标需要加上padding
                y_hat[0, :, h + padding, w + padding] = (y_crop_q + means_hat)[0, :, 0, 0]
                y_q[0, :, h, w] = y_crop_q[0, :, 0, 0]
                y_scales[0, :, h, w] = scales_hat[0, :, 0, 0]
        # 重新排序后交给高斯编码器，生成字节流。也就是整张图量化结束后，才进行真正的熵编码
        y_q = y_q.permute(0, 2, 3, 1)
        y_scales = y_scales.permute(0, 2, 3, 1)
        # 将量化的y和对应的方差输入到编码器，计算码流。这里的y_q隐含了均值，所以输入没有均值，只有方差
        # 所以说，只有均值和方差的计算是神经网络相关的，其他都是传统的编码算法
        # 之所以用超先验、context和上下文，就是为了更精准的估计均值和方差
        y_string = self.gaussian_encoder.compress(y_q, y_scales)
        y_hat = y_hat[:, :, padding:-padding, padding:-padding]
        return y_string, y_hat

    def decompress_ar(self, y_string, channel, height, width, downsample, kernel_size,
                      context_prediction, params, entropy_parameters):
        """
        根据码流y_string，解码y
        """
        device = next(self.parameters()).device
        padding = (kernel_size - 1) // 2

        # 这个height和width是z的大小，所以这里要根据下采样的因子扩大
        y_size = get_downsampled_shape(height, width, downsample)
        y_height = y_size[0]
        y_width = y_size[1]

        # 用于存储解码后的y
        y_hat = torch.zeros(
            (1, channel, y_height + 2 * padding, y_width + 2 * padding),
            device=params.device,
        )
        # 解码要提前把y的码流放到解码器里
        self.gaussian_encoder.set_stream(y_string)

        for h in range(y_height):
            for w in range(y_width):
                # 和自回归压缩完全相反的操作
                y_crop = y_hat[0:1, :, h:h + kernel_size, w:w + kernel_size]
                ctx_p = F.conv2d(y_crop, context_prediction.weight, bias=context_prediction.bias)
                p = params[0:1, :, h:h + 1, w:w + 1]
                gaussian_params = entropy_parameters(torch.cat((p, ctx_p), dim=1))
                means_hat, scales_hat = gaussian_params.chunk(2, 1)
                rv = self.gaussian_encoder.decode_stream(scales_hat)
                rv = rv.to(device)
                rv = rv + means_hat
                y_hat[0, :, h + padding: h + padding + 1, w + padding: w + padding + 1] = rv

        y_hat = y_hat[:, :, padding:-padding, padding:-padding]
        return y_hat

    def compress(self, referframe, input_image):
        """
        该函数是用在模型测试中的，不是训练
        视频帧的完整压缩过程，包含了y和z的压缩。y用的是之前的自回归压缩
        但注意的是，这个函数也包含了解压缩的部分。因此，这个函数是一个完整的压缩-解压缩
        不同的是，这里是虽然计算了y的码流，但没有用上，而是直接用量化的结果，也就是跳过了码流的的解码过程，直接拿到了结果
        """
        device = input_image.device

        # ----------------------处理运动特征----------------------
        # 提取运动信息，光流特征，然后用编码器提取一次语义特征
        estmv = self.opticFlow(input_image, referframe)
        mvfeature = self.mvEncoder(estmv)
        z_mv = self.mvpriorEncoder(mvfeature)
        compressed_z_mv = torch.round(z_mv)
        mv_z_string = self.bitEstimator_z_mv.compress(compressed_z_mv)
        mv_z_size = [compressed_z_mv.size(2), compressed_z_mv.size(3)]
        mv_z_hat = self.bitEstimator_z_mv.decompress(mv_z_string, mv_z_size)
        mv_z_hat = mv_z_hat.to(device)

        # 对于光流的自回归编码，仅采用超先验的z，不需要其他额外的特征
        params_mv = self.mvpriorDecoder(mv_z_hat)
        # mvfeature是运动特征的原始编码结果，params_mv是经过量化和解码后的有损结果
        # mv_y_string是运动特征编码的字节流，mv_y_hat是去掉小数后的整数y
        mv_y_string, mv_y_hat = self.compress_ar(mvfeature, 5, self.auto_regressive_mv,
                                                 params_mv, self.entropy_parameters_mv)

        # 理论上应该对mv_y_string解码得到mv_y_hat，但这里跳过了
        quant_mv_upsample = self.mvDecoder_part1(mv_y_hat)
        quant_mv_upsample_refine = self.mv_refine(referframe, quant_mv_upsample)
        context = self.motioncompensation(referframe, quant_mv_upsample_refine)

        # ----------------------处理图像特征----------------------
        # 在原始图的特征上添加上下文，一起做编码
        feature = self.contextualEncoder(torch.cat((input_image, context), dim=1))
        z = self.priorEncoder(feature)
        compressed_z = torch.round(z)
        z_string = self.bitEstimator_z.compress(compressed_z)
        z_size = [compressed_z.size(2), compressed_z.size(3)]
        z_hat = self.bitEstimator_z.decompress(z_string, z_size)
        z_hat = z_hat.to(device)

        # 对y的自回归编码结合了超先验z和上下文（运动特征）
        params = self.priorDecoder(z_hat)
        # 从运动补偿中提取语义特征，代表一种时间先验。此外，这里也统一了和params的维度，方便后面的cat
        temporal_prior_params = self.temporalPriorEncoder(context)
        # 用z和上下文计算自回归，分别对应超先验和时间先验
        y_string, y_hat = self.compress_ar(feature, 5, self.auto_regressive,
                                           torch.cat((temporal_prior_params, params), dim=1), self.entropy_parameters)

        # y_hat是量化后的y，这里跳过了码流解码的部分，直接用y重构图像了
        # y_string的码流是对残差进行的，如果对y_string解码，则需要decompress_ar，再算个均值加上去
        recon_image_feature = self.contextualDecoder_part1(y_hat)
        recon_image = self.contextualDecoder_part2(torch.cat((recon_image_feature, context), dim=1))

        im_shape = input_image.size()
        pixel_num = im_shape[0] * im_shape[2] * im_shape[3]
        bpp_y = len(y_string) * 8 / pixel_num
        bpp_z = len(z_string) * 8 / pixel_num
        bpp_mv_y = len(mv_y_string) * 8 / pixel_num
        bpp_mv_z = len(mv_z_string) * 8 / pixel_num

        bpp = bpp_y + bpp_z + bpp_mv_y + bpp_mv_z

        return {"bpp_mv_y": bpp_mv_y,
                "bpp_mv_z": bpp_mv_z,
                "bpp_y": bpp_y,
                "bpp_z": bpp_z,
                "bpp": bpp,
                "recon_image": recon_image,
                "mv_y_string": mv_y_string,
                "mv_z_string": mv_z_string,
                "y_string": y_string,
                "z_string": z_string,
                }

    def decompress(self, referframe, mv_y_string, mv_z_string, y_string, z_string, height, width):
        """
        对string码流解码，用解码的结果重构图像
        """
        device = next(self.parameters()).device
        mv_z_size = get_downsampled_shape(height, width, 64)
        mv_z_hat = self.bitEstimator_z_mv.decompress(mv_z_string, mv_z_size)
        mv_z_hat = mv_z_hat.to(device)
        params_mv = self.mvpriorDecoder(mv_z_hat)
        mv_y_hat = self.decompress_ar(mv_y_string, self.out_channel_mv, height, width, 16, 5,
                                      self.auto_regressive_mv, params_mv,
                                      self.entropy_parameters_mv)

        quant_mv_upsample = self.mvDecoder_part1(mv_y_hat)
        quant_mv_upsample_refine = self.mv_refine(referframe, quant_mv_upsample)
        context = self.motioncompensation(referframe, quant_mv_upsample_refine)
        temporal_prior_params = self.temporalPriorEncoder(context)

        z_size = get_downsampled_shape(height, width, 64)
        z_hat = self.bitEstimator_z.decompress(z_string, z_size)
        z_hat = z_hat.to(device)
        params = self.priorDecoder(z_hat)
        y_hat = self.decompress_ar(y_string, self.out_channel_M, height, width, 16, 5,
                                   self.auto_regressive, torch.cat((temporal_prior_params, params), dim=1),
                                   self.entropy_parameters)
        recon_image_feature = self.contextualDecoder_part1(y_hat)
        recon_image = self.contextualDecoder_part2(torch.cat((recon_image_feature, context), dim=1))
        recon_image = recon_image.clamp(0, 1)

        return recon_image

    def forward(self, referframe, input_image):

        # 运动特征提取，运动特征超先验z提取，分别量化
        estmv = self.opticFlow(input_image, referframe)
        mvfeature = self.mvEncoder(estmv)
        z_mv = self.mvpriorEncoder(mvfeature)
        quant_mv = torch.round(mvfeature)
        compressed_z_mv = torch.round(z_mv)

        # 对量化的z解码，对运动特征做自回归，提取上下文特征
        params_mv = self.mvpriorDecoder(compressed_z_mv)
        ctx_params_mv = self.auto_regressive_mv(quant_mv)
        # 结合超先验z和自回归上下文特征，估计运动特征概率分布，并计算码率
        gaussian_params_mv = self.entropy_parameters_mv(
            torch.cat((params_mv, ctx_params_mv), dim=1)
        )
        means_hat_mv, scales_hat_mv = gaussian_params_mv.chunk(2, 1)

        # 对量化后的运动特征第一次解码，这里解码出通道为2，和原像素图像等大的光流图
        quant_mv_upsample = self.mvDecoder_part1(quant_mv)
        # 这是第二次解码，把运动特征和像素的参考图像cat后执行第二次解码，然后将quant_mv_upsample加到解码的结果上
        quant_mv_upsample_refine = self.mv_refine(referframe, quant_mv_upsample)
        # 提取参考帧的特征，和上下文quant_mv_upsample_refine做一次warp，然后做contest refine
        context = self.motioncompensation(referframe, quant_mv_upsample_refine)

        # 结合上下文，对图像特征编码
        feature = self.contextualEncoder(torch.cat((input_image, context), dim=1))
        z = self.priorEncoder(feature)
        compressed_z = torch.round(z)
        params = self.priorDecoder(compressed_z)

        feature_renorm = feature

        compressed_y_renorm = torch.round(feature_renorm)

        # 结合超先验z，自回归和动作信息上下文进行熵编码
        temporal_prior_params = self.temporalPriorEncoder(context)
        ctx_params = self.auto_regressive(compressed_y_renorm)
        gaussian_params = self.entropy_parameters(
            torch.cat((temporal_prior_params, params, ctx_params), dim=1)
        )
        means_hat, scales_hat = gaussian_params.chunk(2, 1)

        # 图像还原，这里直接跳过了解码步骤
        recon_image_feature = self.contextualDecoder_part1(compressed_y_renorm)
        recon_image = self.contextualDecoder_part2(torch.cat((recon_image_feature, context), dim=1))

        # 估计比特率。注意，这里用的是减去均值的比特率估计，也就是和compress的代码一样
        total_bits_y, _ = self.feature_probs_based_sigma(feature_renorm, means_hat, scales_hat)
        total_bits_mv, _ = self.feature_probs_based_sigma(mvfeature, means_hat_mv, scales_hat_mv)
        total_bits_z, _ = self.estrate_bits_z(compressed_z)
        total_bits_z_mv, _ = self.estrate_bits_z_mv(compressed_z_mv)

        im_shape = input_image.size()
        pixel_num = im_shape[0] * im_shape[2] * im_shape[3]
        bpp_y = total_bits_y / pixel_num
        bpp_z = total_bits_z / pixel_num
        bpp_mv_y = total_bits_mv / pixel_num
        bpp_mv_z = total_bits_z_mv / pixel_num

        bpp = bpp_y + bpp_z + bpp_mv_y + bpp_mv_z

        return {"bpp_mv_y": bpp_mv_y,
                "bpp_mv_z": bpp_mv_z,
                "bpp_y": bpp_y,
                "bpp_z": bpp_z,
                "bpp": bpp,
                "recon_image": recon_image,
                "context": context,
                }

    def load_dict(self, pretrained_dict):
        result_dict = {}
        for key, weight in pretrained_dict.items():
            result_key = key
            if key[:7] == "module.":
                result_key = key[7:]
            result_dict[result_key] = weight

        self.load_state_dict(result_dict)
