from src.models.utils import *
from src.models.image_net import PatchMerging, PatchSplitting, PatchEmbed, BasicLayer
from src.entropy_models.entropy_models import EntropyBottleneck
from src.layers.layers import GDN
from src.entropy_models.MLCodec_rans import BufferedRansEncoder
from src.entropy_models.MLCodec_rans import RansDecoder
from src.models.priors import MeanScaleHyperprior
from src.models.stream_helper import encode_i, decode_i


class ConvHyperprior(MeanScaleHyperprior):
    def __init__(self, main_dim, hyper_dim, **kwargs):
        super().__init__(hyper_dim, main_dim, **kwargs)

        # 分别用于图像x的压缩和hyper参数z的压缩
        self.main_dim = main_dim
        self.hyper_dim = hyper_dim

        # 计算hyper参数z的量化编码
        self.entropy_bottleneck = EntropyBottleneck(self.hyper_dim)

        self.g_a = nn.Sequential(
            conv(3, self.main_dim),
            GDN(self.main_dim),
            conv(self.main_dim, self.main_dim),
            GDN(self.main_dim),
            conv(self.main_dim, self.main_dim),
            GDN(self.main_dim),
            conv(self.main_dim, self.main_dim),
        )

        self.g_s = nn.Sequential(
            deconv(self.main_dim, self.main_dim),
            GDN(self.main_dim, inverse=True),
            deconv(self.main_dim, self.main_dim),
            GDN(self.main_dim, inverse=True),
            deconv(self.main_dim, self.main_dim),
            GDN(self.main_dim, inverse=True),
            deconv(self.main_dim, 3),
        )

        self.h_a = nn.Sequential(
            conv(self.main_dim, self.hyper_dim, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(self.hyper_dim, self.hyper_dim),
            nn.ReLU(inplace=True),
            conv(self.hyper_dim, self.hyper_dim),
        )

        self.h_s = nn.Sequential(
            deconv(self.hyper_dim, self.hyper_dim),
            nn.ReLU(inplace=True),
            deconv(self.hyper_dim, self.hyper_dim),
            nn.ReLU(inplace=True),
            conv(self.hyper_dim, self.main_dim * 2, stride=1, kernel_size=3),
        )

    @classmethod
    def from_state_dict(cls, state_dict):
        """根据 checkpoint 推断出模型构造参数，然后用这个类（或者子类）自己new一个实例再load权重.
        """
        main_dim = state_dict["g_a.0.weight"].size(0)
        hyper_dim = state_dict["h_a.0.weight"].size(0)
        net = cls(main_dim, hyper_dim)
        net.load_state_dict(state_dict)
        return net

    def forward(self, x):
        # 编码x和y
        y = self.g_a(x)
        z = self.h_a(y)

        # 计算z的熵编码概率分布。对z的概率模型是用一个小网络拟合CDF（EntropyBottleneck），
        # 相当于学习一个每通道的一维分布，而不是手写高斯之类的参数形式。
        # 注意，其实第一个返回值就是量化好的z，但是是用噪声量化的。代码不想用噪声量化，所以第一个返回值没有接收
        _, z_likelihoods = self.entropy_bottleneck(z)

        z_offset = self.entropy_bottleneck._get_medians()
        # 这是代码的量化方式，用直通的方式模拟量化过程，来源于https://arxiv.org/abs/2007.08739
        z_hat = quantize_ste(z - z_offset) + z_offset

        # 解码hyper参数z，用z估计y分布的均值和方差
        gaussian_params = self.h_s(z_hat)
        # chunk的作用是把一个张量沿着某个维度平均切成几段，返回切割后的元组
        # gaussian_params的返回值为[N, C, H, W]，按照第一维，切成两块，分别表示方差和均值
        scales_hat, means_hat = gaussian_params.chunk(2, 1)

        # 用z的高斯概率分布，计算y的熵编码概率分布
        _, y_likelihoods = self.gaussian_conditional(y, scales_hat, means=means_hat)
        # 和z的量化方案相同
        y_hat = quantize_ste(y - means_hat) + means_hat
        # 训练时 不需要真正算术编码器，用likelihoods就能得到理论上等价的码长估计。
        # 所以这里没有compress和decompress，而是直接解码y了
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(y)

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes, means=means_hat)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, means=means_hat
        )
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {"x_hat": x_hat}

    def encode_decode(self, input, bin_path):
        N, C, H, W = input.size()
        assert N == 1
        encoded = self.compress(input)
        y_string = encoded['strings'][0][0]
        z_string = encoded['strings'][1][0]
        shape = encoded['shape']
        encode_i(H, W, shape[0], shape[1], y_string, z_string, bin_path)
        bpp = (len(y_string) + len(z_string)) * 8 / (H * W)

        height, width, z_height, z_width, y_string, z_string = decode_i(bin_path)
        decoded = self.decompress([[y_string], [z_string]], (z_height, z_width))
        return {
            "x_hat": decoded["x_hat"],
            "bpp": bpp,
        }


class ChARMBlockHalf(nn.Module):
    """实现逐通道卷积的每个slice过程，但是没有按照论文里一样设计encoder-decoder，而是之间卷积出结果
    """
    def __init__(self, in_dim, out_dim) -> None:
        super().__init__()
        c1 = (out_dim - in_dim) // 3 + in_dim
        c2 = 2 * (out_dim - in_dim) // 3 + in_dim
        self.layers = nn.Sequential(
            conv(in_dim, c1, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(c1, c2, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(c2, out_dim, kernel_size=3, stride=1)
        )

    def forward(self, x):
        return self.layers(x)


class ConvChARM(ConvHyperprior):
    """
    在普通的卷积编码基础上添加了上下文，上下文是通过逐通道卷积实现的
    论文参考：Channel-Wise Autoregressive Entropy Models for Learned Image Compression
    """
    def __init__(self, main_dim, hyper_dim, **kwargs):
        super().__init__(main_dim, hyper_dim, **kwargs)

        # 论文中的slice个数，也就是通道维度的切片个数
        self.num_slices = 10
        # 这里隐含了main_dim=320的假设，所以切分成10块后，每块的维度是32
        self.charm_mean_transforms = nn.ModuleList(
            [ChARMBlockHalf(in_dim=32 + 32 * i, out_dim=32) for i in range(self.num_slices)]
        )
        self.charm_scale_transforms = nn.ModuleList(
            [ChARMBlockHalf(in_dim=32 + 32 * i, out_dim=32) for i in range(self.num_slices)]
        )

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(y)

        _, z_likelihoods = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_hat = quantize_ste(z - z_offset) + z_offset

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)

        # 把mean和scale都切片
        means_hat_slices = means_hat.chunk(self.num_slices, 1)
        scales_hat_slices = scales_hat.chunk(self.num_slices, 1)

        # 对y切片
        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_likelihood = []

        for slice_index, y_slice in enumerate(y_slices):
            # 把mean和对应的y拼接后输入，输出结果预测一次概率，并进行一次量化，将结果以列表的形式存储
            mean_support = torch.cat([means_hat_slices[slice_index]] + y_hat_slices, dim=1)
            mu = self.charm_mean_transforms[slice_index](mean_support)

            scale_support = torch.cat([scales_hat_slices[slice_index]] + y_hat_slices, dim=1)
            scale = self.charm_scale_transforms[slice_index](scale_support)

            _, y_slice_likelihood = self.gaussian_conditional(y_slice, scale, mu)
            y_likelihood.append(y_slice_likelihood)
            y_hat_slice = quantize_ste(y_slice - mu) + mu

            y_hat_slices.append(y_hat_slice)

        # 拼接所有的y和概率
        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihood, dim=1)
        x_hat = self.g_s(y_hat)

        return {
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(y)

        # 编码z的比特流
        z_strings = self.entropy_bottleneck.compress(z)
        # 保证编码端的z和解码端完全一致
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        means_hat_slices = means_hat.chunk(self.num_slices, 1)
        scales_hat_slices = scales_hat.chunk(self.num_slices, 1)

        y_slices = y.chunk(self.num_slices, 1)
        y_hat_slices = []
        y_scales = []
        y_means = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        # 存所有要编码的符号（离散整数）
        symbols_list = []
        # 存每个符号用哪一个scale分布表
        indexes_list = []
        y_strings = []

        for slice_index, y_slice in enumerate(y_slices):
            # 逐通道计算均值和方差
            mean_support = torch.cat([means_hat_slices[slice_index]] + y_hat_slices, dim=1)
            mu = self.charm_mean_transforms[slice_index](mean_support)

            scale_support = torch.cat([scales_hat_slices[slice_index]] + y_hat_slices, dim=1)
            scale = self.charm_scale_transforms[slice_index](scale_support)

            # 用方差定位最匹配的CDF分布表
            index = self.gaussian_conditional.build_indexes(scale)
            # 量化y，这里和训练不一样，是真的在量化，转化成整数了
            y_q_slice = self.gaussian_conditional.quantize(y_slice, "symbols", mu)
            y_hat_slice = y_q_slice + mu

            symbols_list.extend(y_q_slice.reshape(-1).tolist())
            indexes_list.extend(index.reshape(-1).tolist())

            y_hat_slices.append(y_hat_slice)
            y_scales.append(scale)
            y_means.append(mu)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        # 取出编码的结果，是一个比特流，本质是01序列，但是计算机为了方便存储，会把8个比特打包成一个字节
        # 注意，这是比特流的字节串，而不是一个字符串
        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        gaussian_params = self.h_s(z_hat)
        scales_hat, means_hat = gaussian_params.chunk(2, 1)
        means_hat_slices = means_hat.chunk(self.num_slices, 1)
        scales_hat_slices = scales_hat.chunk(self.num_slices, 1)

        # 这里的4表示y和z尺度的倍数关系，y比z在宽高上大4倍
        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        # 第一个[0]是y_strings，第二个[0]是取出列表中的第一个元素，列表中只有一个元素，就是编码的比特流
        decoder.set_stream(strings[0][0])

        for slice_index in range(self.num_slices):
            mean_support = torch.cat([means_hat_slices[slice_index]] + y_hat_slices, dim=1)
            mu = self.charm_mean_transforms[slice_index](mean_support)
            # 这里是在做保险，保证y的尺度一定是z的4倍，如果不是就裁减，以应对一些随机尺度图片输入造成的尺度不一致现象
            mu = mu[:, :, :y_shape[0], :y_shape[1]]

            scale_support = torch.cat([scales_hat_slices[slice_index]] + y_hat_slices, dim=1)
            scale = self.charm_scale_transforms[slice_index](scale_support)
            scale = scale[:, :, :y_shape[0], :y_shape[1]]

            index = self.gaussian_conditional.build_indexes(scale)

            rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
            rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
            y_hat_slice = self.gaussian_conditional.dequantize(rv, mu)

            y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        x_hat = self.g_s(y_hat).clamp_(0, 1)

        return {"x_hat": x_hat}


class SwinTAnalysisTransform(nn.Module):
    """
    SwinTrans的编码端，编码y
    """

    def __init__(self, embed_dim, embed_out_dim, depths, window_size, input_dim):
        super().__init__()
        self.patch_embed = PatchEmbed(in_chans=input_dim, embed_dim=embed_dim[0])
        # self.patch_embed = nn.Conv2d(input_dim, embed_dim[0], 2, 2)
        num_layers = len(depths)
        self.layers = nn.ModuleList(
            [BasicLayer(dim=embed_dim[i],
                        out_dim=embed_out_dim[i],
                        depth=depths[i],
                        window_size=window_size[i],
                        downsample=PatchMerging if (i < num_layers - 1) else None)
             for i in range(num_layers)]
        )

    def forward(self, x):
        x = self.patch_embed(x)
        for layer in self.layers:
            x = layer(x)
        return x.permute(0, 3, 1, 2)


class SwinTSynthesisTransform(nn.Module):
    """SwinTrans的解码端，解码y
    """

    def __init__(self, embed_dim, embed_out_dim, depths, window_size):
        super().__init__()
        num_layers = len(depths)
        self.layers = nn.ModuleList(
            [BasicLayer(dim=embed_dim[i],
                        out_dim=embed_out_dim[i],
                        depth=depths[i],
                        window_size=window_size[i],
                        downsample=PatchSplitting)
             for i in range(num_layers)]
        )

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        for layer in self.layers:
            x = layer(x)
        return x.permute(0, 3, 1, 2)


class SwinTHyperAnalysisTransform(nn.Module):
    """SwinTrans的编码端，编码z
    """

    def __init__(self, embed_dim, embed_out_dim, depths, window_size, input_dim):
        super().__init__()
        self.patch_merger = PatchEmbed(in_chans=input_dim, embed_dim=embed_dim[0])
        # self.patch_merger = nn.Conv2d(input_dim, embed_dim[0], 2, 2)
        num_layers = len(depths)
        self.layers = nn.ModuleList(
            [BasicLayer(dim=embed_dim[i],
                        out_dim=embed_out_dim[i],
                        depth=depths[i],
                        window_size=window_size[i],
                        downsample=PatchMerging if (i < num_layers - 1) else None)
             for i in range(num_layers)]
        )

    def forward(self, x):
        x = self.patch_merger(x)
        for layer in self.layers:
            x = layer(x)
        return x.permute(0, 3, 1, 2)


class SwinTHyperSynthesisTransform(nn.Module):
    """SwinTrans的解码端，解码z
    """

    def __init__(self, embed_dim, embed_out_dim, depths, window_size):
        super().__init__()
        num_layers = len(depths)
        self.layers = nn.ModuleList(
            [BasicLayer(dim=embed_dim[i],
                        out_dim=embed_out_dim[i],
                        depth=depths[i],
                        window_size=window_size[i],
                        downsample=PatchSplitting)
             for i in range(num_layers)]
        )

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        for layer in self.layers:
            x = layer(x)
        return x.permute(0, 3, 1, 2)


class SwinTHyperprior(ConvHyperprior):
    """
    SwinT无上下文版本的编码器
    """

    def __init__(self, g_a, g_s, h_a, h_s):
        super().__init__(main_dim=g_a["embed_dim"][-1], hyper_dim=h_a["embed_dim"][-1])
        self.g_a = SwinTAnalysisTransform(**g_a)
        self.g_s = SwinTSynthesisTransform(**g_s)
        self.h_a = SwinTHyperAnalysisTransform(**h_a)
        self.h_s = SwinTHyperSynthesisTransform(**h_s)

    def _config_from_state_dict(self, state_dict):
        raise NotImplementedError(
            "_config_from_state_dict is not supported for SwinTHyperprior."
        )

    @classmethod
    def from_state_dict(cls, state_dict):
        raise NotImplementedError(
            "from_state_dict is not supported for SwinTHyperprior."
        )


class SwinTChARM(ConvChARM):
    """
    SwinT上下文版本的编码器
    """

    def __init__(self, g_a, g_s, h_a, h_s):
        super().__init__(main_dim=g_a["embed_dim"][-1], hyper_dim=h_a["embed_dim"][-1])
        self.g_a = SwinTAnalysisTransform(**g_a)
        self.g_s = SwinTSynthesisTransform(**g_s)
        self.h_a = SwinTHyperAnalysisTransform(**h_a)
        self.h_s = SwinTHyperSynthesisTransform(**h_s)

    def _config_from_state_dict(self, state_dict):
        raise NotImplementedError(
            "_config_from_state_dict is not supported for SwinTChARM."
        )

    @classmethod
    def from_state_dict(cls, state_dict):
        raise NotImplementedError(
            "from_state_dict is not supported for SwinTChARM."
        )






