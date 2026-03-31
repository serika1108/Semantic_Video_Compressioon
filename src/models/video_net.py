import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

# 光流warp时的缓存网格，分为cpu和gpu两种场景。这个是为了存储不同尺度光流图的网格
Backward_tensorGrid = {}
Backward_tensorGrid_cpu = {}


class LowerBound(Function):
    """
    对训练梯度截取的类，控制梯度的变化，类似于反向传播的clamp
    """

    @staticmethod
    def forward(ctx, inputs, bound):
        b = torch.ones_like(inputs) * bound
        ctx.save_for_backward(inputs, b)
        return torch.max(inputs, b)

    @staticmethod
    def backward(ctx, grad_output):
        inputs, b = ctx.saved_tensors

        # 控制梯度的核心操作，如果大于下界b，则放行。如果小于下界，但梯度和loss反方向，也放行
        pass_through_1 = inputs >= b
        pass_through_2 = grad_output < 0
        pass_through = pass_through_1 | pass_through_2

        return pass_through.type(grad_output.dtype) * grad_output, None


class GDN(nn.Module):
    """
    图像重构的归一化层，用于代替BN，出自论文:
    Density Modeling of Images using a Generalized Normalization Transformation
    """

    def __init__(self, ch, inverse=False, beta_min=1e-6, gamma_init=0.1, reparam_offset=2 ** -18):
        super(GDN, self).__init__()
        self.inverse = inverse
        self.beta_min = beta_min
        self.gamma_init = gamma_init
        self.reparam_offset = reparam_offset

        self.build(ch)

    def build(self, ch):
        # 设置beta和gamma的下界
        self.pedestal = self.reparam_offset ** 2
        self.beta_bound = ((self.beta_min + self.reparam_offset ** 2) ** 0.5)
        self.gamma_bound = self.reparam_offset

        # 初始化beta
        beta = torch.sqrt(torch.ones(ch) + self.pedestal)
        self.beta = nn.Parameter(beta)
        # 初始化gamma
        eye = torch.eye(ch)
        g = self.gamma_init * eye
        g = g + self.pedestal
        gamma = torch.sqrt(g)

        self.gamma = nn.Parameter(gamma)
        self.pedestal = self.pedestal

    def forward(self, inputs):
        unfold = False
        if inputs.dim() == 5:
            # 兼容5D的输入
            unfold = True
            bs, ch, d, w, h = inputs.size()
            inputs = inputs.view(bs, ch, d * w, h)

        _, ch, _, _ = inputs.size()

        # 通过梯度的控制，对beta和gamma的数值进行调整，保证训练的数值稳定
        beta = LowerBound.apply(self.beta, self.beta_bound)
        beta = beta ** 2 - self.pedestal
        gamma = LowerBound.apply(self.gamma, self.gamma_bound)
        gamma = gamma ** 2 - self.pedestal
        gamma = gamma.view(ch, ch, 1, 1)

        # GDN的核心操作。GDN的分母可看做卷积核为gamma，偏置为beta的卷积操作
        norm_ = nn.functional.conv2d(inputs ** 2, gamma, beta)
        norm_ = torch.sqrt(norm_)

        # Apply norm
        if self.inverse:
            outputs = inputs * norm_
        else:
            outputs = inputs / norm_

        if unfold:
            outputs = outputs.view(bs, ch, d, w, h)
        return outputs


def torch_warp(tensorInput, tensorFlow):
    """
    warp操作，在特征图上附加光流，计算运动补偿后的特征
    tensorInput: [B, C, H, W]
    tensorFlow: [B, 2, H, W]，两个通道代表水平位移dx和垂直位移dy
    光流是偏移量(dx, dy)，warp的思想是根据偏移的(dx, dy)，把原图坐标的值映射到新的坐标位置。如果(dx, dy)是小数，则用插值近似
    """
    # 根据cpu和gpu选择不同的操作
    if tensorInput.device == torch.device('cpu'):
        # 这里在构建一个归一化的坐标网格。因为torch的grid_sample做warp只能用(x+dx, y+dy)，而不能用(dx, dy)
        # 同时，grid_sample需要用归一化的坐标，也就是(-1,1)区间，所以这里生成了(-1,1)的均匀分布，在此基础上加上归一化的(dx, dy)

        # 检测当前尺度的光流图网格是否存储在缓存中
        if str(tensorFlow.size()) not in Backward_tensorGrid_cpu:
            # 构建(-1,1)的均匀分布作为基础坐标，根据图像的长宽分别构建
            tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(
                1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
            tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(
                1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))
            # 将长宽两个维度的基础坐标网格在通道维cat到一起，[B, 1, H, W] -> [B, 2, H, W]
            Backward_tensorGrid_cpu[str(tensorFlow.size())] = torch.cat(
                [tensorHorizontal, tensorVertical], 1).cpu()

        # tensorFlow是光流，通常是像素单位的位移，也就是H，W维度的。这里根据(-1,1)维度归一化
        tensorFlow = torch.cat([
            tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0),
            tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0)], 1)

        # 把光流的dx，dy加到基础坐标网格上，得到(x+dx, y+dy)
        grid = (Backward_tensorGrid_cpu[str(tensorFlow.size())] + tensorFlow)
        # 用torch的函数做warp，它会处理小数位移的情况。对于小数位移，将采用双线性插值近似
        return torch.nn.functional.grid_sample(input=tensorInput,
                                               grid=grid.permute(0, 2, 3, 1),
                                               mode='bilinear',
                                               padding_mode='border',
                                               align_corners=True)
    else:
        if str(tensorFlow.size()) not in Backward_tensorGrid:
            tensorHorizontal = torch.linspace(-1.0, 1.0, tensorFlow.size(3)).view(
                1, 1, 1, tensorFlow.size(3)).expand(tensorFlow.size(0), -1, tensorFlow.size(2), -1)
            tensorVertical = torch.linspace(-1.0, 1.0, tensorFlow.size(2)).view(
                1, 1, tensorFlow.size(2), 1).expand(tensorFlow.size(0), -1, -1, tensorFlow.size(3))
            Backward_tensorGrid[str(tensorFlow.size())] = torch.cat(
                [tensorHorizontal, tensorVertical], 1).to(tensorInput.device)

        tensorFlow = torch.cat([tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0),
                                tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0)], 1)

        grid = (Backward_tensorGrid[str(tensorFlow.size())] + tensorFlow)
        return torch.nn.functional.grid_sample(input=tensorInput,
                                               grid=grid.permute(0, 2, 3, 1),
                                               mode='bilinear',
                                               padding_mode='border',
                                               align_corners=True)


def flow_warp(im, flow):
    """
    调用torch_warp函数，进行光流warp
    """
    warp = torch_warp(im, flow)
    return warp


def load_weight_form_np(me_model_dir, layername):
    index = layername.find('modelL')
    if index == -1:
        print('load models error!!')
    else:
        name = layername[index:index + 11]
        modelweight = me_model_dir + name + '-weight.npy'
        modelbias = me_model_dir + name + '-bias.npy'
        weightnp = np.load(modelweight)
        biasnp = np.load(modelbias)
        return torch.from_numpy(weightnp), torch.from_numpy(biasnp)


def bilinearupsacling(inputfeature):
    """
    通过双线性插值将输入特征inputfeature放大两倍，得到上采样的特征图
    """
    inputheight = inputfeature.size()[2]
    inputwidth = inputfeature.size()[3]
    outfeature = F.interpolate(
        inputfeature, (inputheight * 2, inputwidth * 2), mode='bilinear', align_corners=False)
    return outfeature


class ResBlock(nn.Module):
    def __init__(self, inputchannel, outputchannel, kernel_size, stride=1):
        super(ResBlock, self).__init__()
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(inputchannel, outputchannel,
                               kernel_size, stride, padding=kernel_size // 2)
        torch.nn.init.xavier_uniform_(self.conv1.weight.data)
        torch.nn.init.constant_(self.conv1.bias.data, 0.0)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(outputchannel, outputchannel,
                               kernel_size, stride, padding=kernel_size // 2)
        torch.nn.init.xavier_uniform_(self.conv2.weight.data)
        torch.nn.init.constant_(self.conv2.bias.data, 0.0)
        # 处理残差的通道数不一致问题。函数设定的inputchannel和outputchannel可能不相同，此时残差不能直接相加
        if inputchannel != outputchannel:
            # 通过一个1*1的卷积扩展通道维度
            self.adapt_conv = nn.Conv2d(inputchannel, outputchannel, 1)
            torch.nn.init.xavier_uniform_(self.adapt_conv.weight.data)
            torch.nn.init.constant_(self.adapt_conv.bias.data, 0.0)
        else:
            self.adapt_conv = None

    def forward(self, x):
        x_1 = self.relu1(x)
        firstlayer = self.conv1(x_1)
        firstlayer = self.relu2(firstlayer)
        seclayer = self.conv2(firstlayer)
        if self.adapt_conv is None:
            return x + seclayer
        else:
            return self.adapt_conv(x) + seclayer


class ResBlock_LeakyReLU_0_Point_1(nn.Module):
    def __init__(self, d_model):
        super(ResBlock_LeakyReLU_0_Point_1, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(d_model, d_model, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(d_model, d_model, 3, stride=1, padding=1),
            nn.LeakyReLU(0.1, inplace=True))

    def forward(self, x):
        x = x + self.conv(x)
        return x


class MEBasic(nn.Module):
    """
    光流估计基本模块，对相邻帧的特征做refine，得到真正的光流
    """

    def __init__(self):
        super(MEBasic, self).__init__()
        self.conv1 = nn.Conv2d(8, 32, 7, 1, padding=3)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 7, 1, padding=3)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(64, 32, 7, 1, padding=3)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(32, 16, 7, 1, padding=3)
        self.relu4 = nn.ReLU()
        self.conv5 = nn.Conv2d(16, 2, 7, 1, padding=3)

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.relu4(self.conv4(x))
        x = self.conv5(x)
        return x


class ME_Spynet(nn.Module):
    """
    运动估计网络motion estimation，Spynet是一个多尺度金字塔的架构
    """

    def __init__(self):
        super(ME_Spynet, self).__init__()
        # 金字塔层数
        self.L = 4
        # intLevel代表金字塔层数，为每层金字塔的输出设置一个MEBasic做refinement
        self.moduleBasic = torch.nn.ModuleList([MEBasic() for intLevel in range(4)])

    def forward(self, im1, im2):
        batchsize = im1.size()[0]

        # 构建图像金字塔结构
        im1_pre = im1
        im2_pre = im2
        im1list = [im1_pre]
        im2list = [im2_pre]
        # 每一层金字塔下采样一倍
        for intLevel in range(self.L - 1):
            im1list.append(F.avg_pool2d(im1list[intLevel], kernel_size=2, stride=2))
            im2list.append(F.avg_pool2d(im2list[intLevel], kernel_size=2, stride=2))

        shape_fine = im2list[self.L - 1].size()
        # 初始化小尺度特征图，用于初始的金字塔特征融合
        zeroshape = [batchsize, 2, shape_fine[2] // 2, shape_fine[3] // 2]
        device = im1.device
        flowfileds = torch.zeros(zeroshape, dtype=torch.float32, device=device)
        for intLevel in range(self.L):
            flowfiledsUpsample = bilinearupsacling(flowfileds) * 2.0
            flowfileds = flowfiledsUpsample + self.moduleBasic[intLevel](
                torch.cat([
                    im1list[self.L - 1 - intLevel],
                    flow_warp(im2list[self.L - 1 - intLevel], flowfiledsUpsample),
                    flowfiledsUpsample], 1))

        return flowfileds
