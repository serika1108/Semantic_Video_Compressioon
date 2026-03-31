import argparse
import math
import os
from datetime import datetime
import json
from src.models.P_Coding import DCVC_net
from src.models.architecture import getModel
import time
from tqdm import tqdm
from pytorch_msssim import ms_ssim


def bool_flag(s):
    FALSY_STRINGS = {"off", "false", "0", "False"}
    TRUTHY_STRINGS = {"on", "true", "1", "True"}

    if isinstance(s, bool):
        return s

    if s.lower() in FALSY_STRINGS:
        return False
    elif s.lower() in TRUTHY_STRINGS:
        return True
    else:
        print(s)
        raise argparse.ArgumentTypeError("invalid value for a boolean flag")


def parse_args():
    parser = argparse.ArgumentParser(description="Example testing script")

    parser.add_argument('--i_frame_model_name', type=str, default="ConvHyperprior")
    parser.add_argument('--i_frame_model_path', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--test_config', type=str, required=True)
    parser.add_argument("--cuda", type=bool_flag, default=True)
    parser.add_argument('--write_stream', type=bool_flag, default=False)
    parser.add_argument("--write_recon_frame", type=bool_flag, default=True)
    parser.add_argument('--output_dir', type=str, default="output_dir")
    parser.add_argument('--output_json_result_path', type=str, required=True)
    parser.add_argument("--model_type", type=str, default="psnr", help="psnr, msssim")

    args = parser.parse_args()
    return args


def PSNR(input1, input2):
    mse = torch.mean((input1 - input2) ** 2)
    psnr = 20 * torch.log10(1 / torch.sqrt(mse))
    return psnr.item()


from PIL import Image
import numpy as np
import torch


def center_crop_to_multiple(img: np.ndarray, multiple: int):
    h, w, c = img.shape
    new_h = (h // multiple) * multiple
    new_w = (w // multiple) * multiple
    if new_h == 0 or new_w == 0:
        raise ValueError(
            f"Image size ({h}, {w}) is smaller than required multiple {multiple}."
        )
    if new_h == h and new_w == w:
        return img
    top = (h - new_h) // 2
    left = (w - new_w) // 2
    return img[top:top + new_h, left:left + new_w, :]


def read_frame_to_torch(model_name, path):
    """
    读取图像并转成tensor。
    根据模型名决定裁剪倍数：
    - Conv*: 高宽裁剪到 64 的倍数
    - Swin*: 高宽裁剪到 128 的倍数
    """
    prefix = model_name[:4]

    if prefix == "Swin":
        multiple = 256
    else:
        multiple = 64

    input_image = Image.open(path).convert("RGB")
    input_image = np.asarray(input_image).astype("float32")

    input_image = center_crop_to_multiple(input_image, multiple)

    input_image = input_image.transpose(2, 0, 1)
    input_image = torch.from_numpy(input_image).float()
    input_image = input_image.unsqueeze(0) / 255.0

    return input_image


def write_torch_frame(frame, path):
    frame_result = frame.clone()
    frame_result = frame_result.cpu().detach().numpy().transpose(1, 2, 0) * 255
    frame_result = np.clip(np.rint(frame_result), 0, 255)
    frame_result = Image.fromarray(frame_result.astype("uint8"))
    frame_result.save(path)


def encode_one(args, device, recon_dir, i_frame_net, video_net):

    # 数据集中某一段测试数据的视频序列路径
    sub_dir_name = args.video_name

    ref_frame = None  # 参考帧
    frame_types = []  # 记录每一帧是I帧还是P帧，I帧为0，P帧为1
    qualitys_psnr = []  # 帧重构的指标
    qualitys_msssim = []
    bits = []  # 每一帧的总比特数

    # 记录运动信息的y，z比特数，记录特征图的y，z比特数。这两类信息只适用于P帧，不对I帧进行记录
    bits_mv_y = []
    bits_mv_z = []
    bits_y = []
    bits_z = []

    # GoP大小，总像素大小，总帧数
    gop_size = args.gop
    frame_pixel_num = 0
    frame_num = args.frame_num

    pngs = os.listdir(os.path.join(args.dataset_path, sub_dir_name))
    if 'im1.png' in pngs:
        padding = 1
    elif 'im00001.png' in pngs:
        padding = 5
    else:
        raise ValueError('未知图像命名格式，请统一化命名')

    with torch.no_grad():
        # 取每一帧，frame_idx是帧编号
        for frame_idx in range(frame_num):
            # 从文件夹取图像帧，读取到torch中。这里假设帧的命名是im1.png或im00001.png的格式
            ori_frame = read_frame_to_torch(args.i_frame_model_name,
                                            os.path.join(args.dataset_path, sub_dir_name,
                                                         f"im{str(frame_idx + 1).zfill(padding)}.png"))
            ori_frame = ori_frame.to(device)

            # 确保每个图像帧的大小都一样。frame_pixel_num初始为0，因此初始记录一次像素大小
            if frame_pixel_num == 0:
                frame_pixel_num = ori_frame.shape[2] * ori_frame.shape[3]
            else:
                # 如果后面的某一帧大小和初始图像大小不一样，则停止计算并报错
                assert (frame_pixel_num == ori_frame.shape[2] * ori_frame.shape[3])

            # 把编码的码率写入到输出目录的bin文件中
            if args.write_stream:
                recon_dir_bit = os.path.join(recon_dir, "bit_stream")
                os.makedirs(recon_dir_bit, exist_ok=True)
                bin_path = os.path.join(recon_dir_bit, f"{frame_idx}.bin")
                # I帧编码
                if frame_idx % gop_size == 0:
                    result = i_frame_net.encode_decode(ori_frame, bin_path)
                    ref_frame = result["x_hat"]
                    bpp = result["bpp"]
                    # 记录帧的类型，关键帧为0
                    frame_types.append(0)
                    # 记录编码帧的总比特数
                    bits.append(bpp * frame_pixel_num)
                    # 对于关键帧，只记录了编码总比特，其他都记为0。尽管关键帧有y和z，但为了统一，仍记为0
                    bits_mv_y.append(0)
                    bits_mv_z.append(0)
                    bits_y.append(0)
                    bits_z.append(0)
                # P帧编码
                else:
                    result = video_net.encode_decode(ref_frame, ori_frame, bin_path)
                    ref_frame = result['recon_image']
                    bpp = result['bpp']
                    # P帧类型记录为1
                    frame_types.append(1)
                    # 记录总比特数，运动信息比特数，图像特征比特数
                    bits.append(bpp * frame_pixel_num)
                    bits_mv_y.append(result['bpp_mv_y'] * frame_pixel_num)
                    bits_mv_z.append(result['bpp_mv_z'] * frame_pixel_num)
                    bits_y.append(result['bpp_y'] * frame_pixel_num)
                    bits_z.append(result['bpp_z'] * frame_pixel_num)
            # 如果不写入比特流，就直接过一遍网络，比特用香农熵估计，给一个理论值
            else:
                if frame_idx % gop_size == 0:
                    result = i_frame_net(ori_frame)
                    bit = sum((torch.log(likelihoods).sum() / (-math.log(2)))
                              for likelihoods in result["likelihoods"].values())
                    ref_frame = result["x_hat"]
                    frame_types.append(0)
                    bits.append(bit.item())
                    bits_mv_y.append(0)
                    bits_mv_z.append(0)
                    bits_y.append(0)
                    bits_z.append(0)
                else:
                    result = video_net(ref_frame, ori_frame)
                    ref_frame = result['recon_image']
                    bpp = result['bpp']
                    frame_types.append(1)
                    bits.append(bpp.item() * frame_pixel_num)
                    bits_mv_y.append(result['bpp_mv_y'].item() * frame_pixel_num)
                    bits_mv_z.append(result['bpp_mv_z'].item() * frame_pixel_num)
                    bits_y.append(result['bpp_y'].item() * frame_pixel_num)
                    bits_z.append(result['bpp_z'].item() * frame_pixel_num)

            ref_frame = ref_frame.clamp_(0, 1)
            if args.write_recon_frame:
                write_torch_frame(ref_frame.squeeze(),
                                  os.path.join(recon_dir, f"recon_frame_{frame_idx}.png"))
            qualitys_psnr.append(PSNR(ref_frame, ori_frame))
            qualitys_msssim.append(ms_ssim(ref_frame, ori_frame, data_range=1.0).item())
    # 分为I帧和P帧，分别记录实验数据
    cur_all_i_frame_bit = 0
    cur_all_i_frame_quality_psnr = 0
    cur_all_i_frame_quality_msssim = 0
    cur_all_p_frame_bit = 0
    cur_all_p_frame_bit_mv_y = 0
    cur_all_p_frame_bit_mv_z = 0
    cur_all_p_frame_bit_y = 0
    cur_all_p_frame_bit_z = 0
    cur_all_p_frame_quality_psnr = 0
    cur_all_p_frame_quality_msssim = 0
    cur_i_frame_num = 0
    cur_p_frame_num = 0
    for idx in range(frame_num):
        if frame_types[idx] == 0:
            cur_all_i_frame_bit += bits[idx]
            cur_all_i_frame_quality_psnr += qualitys_psnr[idx]
            cur_all_i_frame_quality_msssim += qualitys_msssim[idx]
            cur_i_frame_num += 1
        else:
            cur_all_p_frame_bit += bits[idx]
            cur_all_p_frame_bit_mv_y += bits_mv_y[idx]
            cur_all_p_frame_bit_mv_z += bits_mv_z[idx]
            cur_all_p_frame_bit_y += bits_y[idx]
            cur_all_p_frame_bit_z += bits_z[idx]
            cur_all_p_frame_quality_psnr += qualitys_psnr[idx]
            cur_all_p_frame_quality_msssim += qualitys_msssim[idx]
            cur_p_frame_num += 1

    log_result = {}
    log_result['name'] = f"{os.path.basename(args.model_path)}_{sub_dir_name}"
    log_result['ds_name'] = args.ds_name
    log_result['frame_pixel_num'] = frame_pixel_num
    log_result['i_frame_num'] = cur_i_frame_num
    log_result['p_frame_num'] = cur_p_frame_num
    log_result['ave_i_frame_bpp'] = cur_all_i_frame_bit / cur_i_frame_num / frame_pixel_num
    log_result['ave_i_frame_quality_psnr'] = cur_all_i_frame_quality_psnr / cur_i_frame_num
    log_result['ave_i_frame_quality_msssim'] = cur_all_i_frame_quality_msssim / cur_i_frame_num
    # 记录了p帧的平均码率
    if cur_p_frame_num > 0:
        total_p_pixel_num = cur_p_frame_num * frame_pixel_num
        log_result['ave_p_frame_bpp'] = cur_all_p_frame_bit / total_p_pixel_num
        log_result['ave_p_frame_bpp_mv_y'] = cur_all_p_frame_bit_mv_y / total_p_pixel_num
        log_result['ave_p_frame_bpp_mv_z'] = cur_all_p_frame_bit_mv_z / total_p_pixel_num
        log_result['ave_p_frame_bpp_y'] = cur_all_p_frame_bit_y / total_p_pixel_num
        log_result['ave_p_frame_bpp_z'] = cur_all_p_frame_bit_z / total_p_pixel_num
        log_result['ave_p_frame_quality_psnr'] = cur_all_p_frame_quality_psnr / cur_p_frame_num
        log_result['ave_p_frame_quality_msssim'] = cur_all_p_frame_quality_msssim / cur_p_frame_num
    else:
        log_result['ave_p_frame_bpp'] = 0
        log_result['ave_p_frame_quality'] = 0
        log_result['ave_p_frame_bpp_mv_y'] = 0
        log_result['ave_p_frame_bpp_mv_z'] = 0
        log_result['ave_p_frame_bpp_y'] = 0
        log_result['ave_p_frame_bpp_z'] = 0
    # 计算一次总的帧率，和总的重构质量
    log_result['ave_all_frame_bpp'] = (cur_all_i_frame_bit + cur_all_p_frame_bit) / (frame_num * frame_pixel_num)
    log_result['ave_all_frame_quality_psnr'] = (cur_all_i_frame_quality_psnr + cur_all_p_frame_quality_psnr) / frame_num
    log_result['ave_all_frame_quality_msssim'] = (
                                                             cur_all_i_frame_quality_msssim + cur_all_p_frame_quality_msssim) / frame_num

    return log_result


def filter_dict(result):
    """
    对模型输出结果进行筛选，可以再这里指定保存哪些内容
    """
    keys = ['i_frame_num', 'p_frame_num', 'ave_i_frame_bpp', 'ave_i_frame_quality_psnr', 'ave_i_frame_quality_msssim',
            'ave_p_frame_bpp', 'ave_p_frame_bpp_mv_y', 'ave_p_frame_bpp_mv_z', 'ave_p_frame_bpp_y',
            'ave_p_frame_bpp_z', 'ave_p_frame_quality_psnr', 'ave_p_frame_quality_msssim',
            'ave_all_frame_bpp', 'ave_all_frame_quality_psnr', 'ave_all_frame_quality_msssim']
    res = {k: v for k, v in result.items() if k in keys}
    return res


def main():
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    args = parse_args()

    # 设备类型判断
    if args.cuda and torch.cuda.is_available():
        device = "cuda:0"
    else:
        device = "cpu"
    # 固定随机化参数
    torch.manual_seed(0)
    np.random.seed(seed=0)
    if args.write_stream:
        torch.backends.cudnn.benchmark = False
        if 'use_deterministic_algorithms' in dir(torch):
            torch.use_deterministic_algorithms(True)
        else:
            torch.set_deterministic(True)

    with open(args.test_config) as f:
        config = json.load(f)

    count_frames = 0  # 总帧数
    count_sequences = 0  # 统计编码了多少个视频序列
    log_result = {}  # 记录运行结果

    # 统计待测试的视频序列总数，创建进度条
    total_sequences = sum(len(config[ds_name]['sequences']) for ds_name in config)
    pbar = tqdm(total=total_sequences, desc='Testing sequences')

    # 输出总目录
    base_output_dir = args.output_dir
    time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    # 在时间序列上加上模型名，-4是为了去掉.pth后缀
    suffix = os.path.splitext(os.path.basename(args.model_path))[0] + "_" + time_str
    output_dir = os.path.join(base_output_dir, suffix)
    os.makedirs(output_dir, exist_ok=True)
    output_dir_json = os.path.join(output_dir, f"{time_str}.json")

    # 加载编码I帧的模型
    i_frame_load_checkpoint = torch.load(args.i_frame_model_path, map_location=torch.device('cpu'),
                                         weights_only=True)
    i_frame_net = getModel(args.i_frame_model_name)
    i_frame_net.load_state_dict(i_frame_load_checkpoint)
    # i_frame_net = architectures[args.i_frame_model_name].from_state_dict(
    #     i_frame_load_checkpoint)

    # 加载编码P帧的模型
    video_net = DCVC_net()
    load_checkpoint = torch.load(args.model_path, map_location=torch.device('cpu'), weights_only=True)
    video_net.load_dict(load_checkpoint)

    # 单独提取光流网络
    # spynet_checkpoint = {
    #     "model_name": "ME_Spynet",
    #     "state_dict": {
    #         k: v.cpu()
    #         for k, v in video_net.opticFlow.state_dict().items()
    #     }
    # }
    # torch.save(spynet_checkpoint, "spynet_checkpoint.pth")
    # print("Saved SpyNet checkpoint to spynet_checkpoint.pth")

    # 如果要写入比特流，就刷新一下熵模型的概率分布参数
    if args.write_stream:
        video_net.update(force=True)
        i_frame_net.update(force=True)

    i_frame_net = i_frame_net.to(device).eval()
    video_net = video_net.to(device).eval()

    begin_time = time.time()
    # 遍历每个数据集
    for ds_name in config:
        log_result[ds_name] = {}
        # 遍历数据集中每一段视频序列
        for seq_name in config[ds_name]['sequences']:
            log_result[ds_name][seq_name] = {}

            count_sequences += 1
            args.gop = config[ds_name]['sequences'][seq_name]['gop']
            args.frame_num = config[ds_name]['sequences'][seq_name]['frames']
            args.dataset_path = config[ds_name]['base_path']
            args.ds_name = ds_name
            args.video_name = seq_name
            count_frames += args.frame_num

            seq_output_dir = os.path.join(output_dir, seq_name)
            os.makedirs(seq_output_dir, exist_ok=True)

            result = encode_one(args, device, seq_output_dir, i_frame_net, video_net)
            # 筛选结果并存入到json文件中
            log_result[ds_name][seq_name] = filter_dict(result)

            # 更新进度条
            pbar.update(1)

    # 将结果写入到json文件里
    with open(output_dir_json, 'w') as fp:
        json.dump(log_result, fp, indent=2)

    total_minutes = (time.time() - begin_time) / 60

    print('Test finished')
    print(f'Tested on {count_frames} frames from {count_sequences} sequences')
    print(f'Total elapsed time: {total_minutes:.1f} min')


if __name__ == "__main__":
    main()
