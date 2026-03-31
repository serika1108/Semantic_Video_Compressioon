import struct
from pathlib import Path
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor


# compressAI熵编码得到的是字节码流，这里需要把字节码流写入到文件中，代码规定了一种写入文件的格式
# 具体而言，对于元数据，例如y和z的码流长度，使用unint格式写入文件。对于实际码流，使用byte格式写入
# 这样，就确立了一种文件格式，码流的读取和写入都遵循这个格式来
# 因此，这个python文件的主要作用是，给神经网络的编解码器定义了一套I帧和P帧的二进制文件格式，确保数据的排列顺序，方便跨平台

def get_downsampled_shape(height, width, p):
    """
    对于步长为p的下采样，判断当前长宽是否是p的整数倍，如果不是，扩充到整数倍，返回新的长宽值
    这里没有更改原始图像的长宽，只是计算了期望的长宽尺度
    """
    new_h = (height + p - 1) // p * p
    new_w = (width + p - 1) // p * p
    return int(new_h / p + 0.5), int(new_w / p + 0.5)


def filesize(filepath: str) -> int:
    """
    返回文件的字节大小，可用来算bpp
    """
    if not Path(filepath).is_file():
        raise ValueError(f'Invalid file "{filepath}".')
    return Path(filepath).stat().st_size


def load_image(filepath: str) -> Image.Image:
    return Image.open(filepath).convert("RGB")


def img2torch(img: Image.Image) -> torch.Tensor:
    return ToTensor()(img).unsqueeze(0)


def torch2img(x: torch.Tensor) -> Image.Image:
    return ToPILImage()(x.clamp_(0, 1).squeeze())


# struct定义了一套码流格式，用于写入和读取相关数值。数据的类型包含三种，unint，unchar，byte
def write_uints(fd, values, fmt=">{:d}I"):
    # pack会根据给定的格式写入数据，I代表无符号整数，>代表大端字节存储
    fd.write(struct.pack(fmt.format(len(values)), *values))


def write_uchars(fd, values, fmt=">{:d}B"):
    fd.write(struct.pack(fmt.format(len(values)), *values))


def read_uints(fd, n, fmt=">{:d}I"):
    # 这里在计算一个unint类型的数据占多少字节，例如这里I代表无符号整数，那么sz就是4字节
    sz = struct.calcsize("I")
    # 读取文件的数据，并按照给定的规则把二进制比特流解包成unint
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def read_uchars(fd, n, fmt=">{:d}B"):
    sz = struct.calcsize("B")
    return struct.unpack(fmt.format(n), fd.read(n * sz))


def write_bytes(fd, values, fmt=">{:d}s"):
    if len(values) == 0:
        return
    fd.write(struct.pack(fmt.format(len(values)), values))


def read_bytes(fd, n, fmt=">{:d}s"):
    sz = struct.calcsize("s")
    return struct.unpack(fmt.format(n), fd.read(n * sz))[0]


def pad(x, p=2 ** 6):
    # 把x的长宽补到64的倍数
    h, w = x.size(2), x.size(3)
    H = (h + p - 1) // p * p
    W = (w + p - 1) // p * p
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(x, (padding_left, padding_right, padding_top, padding_bottom),
                 mode="constant", value=0)


def crop(x, size):
    # 和pad配对，对图像做中心裁剪，去掉填充的数据
    H, W = x.size(2), x.size(3)
    h, w = size
    padding_left = (W - w) // 2
    padding_right = W - w - padding_left
    padding_top = (H - h) // 2
    padding_bottom = H - h - padding_top
    return F.pad(
        x,
        (-padding_left, -padding_right, -padding_top, -padding_bottom),
        mode="constant",
        value=0,
    )


def encode_i(height, width, z_height, z_width, y_string, z_string, output):
    """
    写入编码I帧产生的比特流
    """
    with Path(output).open("wb") as f:
        y_string_length = len(y_string)
        z_string_length = len(z_string)

        # 对于元数据，使用int写入。对于编码流，使用比特写入
        write_uints(f, (height, width, z_height, z_width, y_string_length, z_string_length))
        write_bytes(f, y_string)
        write_bytes(f, z_string)


def decode_i(inputpath):
    """
    读取编码I帧对应的比特流
    """
    with Path(inputpath).open("rb") as f:
        header = read_uints(f, 6)
        height = header[0]
        width = header[1]
        z_height = header[2]
        z_width = header[3]
        y_string_length = header[4]
        z_string_length = header[5]

        y_string = read_bytes(f, y_string_length)
        z_string = read_bytes(f, z_string_length)

    return height, width, z_height, z_width, y_string, z_string


def encode_p(height, width, mv_y_string, mv_z_string, y_string, z_string, output):
    """
    写入编码P帧产生的比特流，包含了运动信息和特征信息，分别写入y和z
    """
    with Path(output).open("wb") as f:
        mv_y_string_length = len(mv_y_string)
        mv_z_string_length = len(mv_z_string)
        y_string_length = len(y_string)
        z_string_length = len(z_string)

        write_uints(f, (height, width,  mv_y_string_length, mv_z_string_length,
                    y_string_length, z_string_length))
        write_bytes(f, mv_y_string)
        write_bytes(f, mv_z_string)
        write_bytes(f, y_string)
        write_bytes(f, z_string)


def decode_p(inputpath):
    """
    读取编码P帧对应的比特流，包含了运动信息和特征信息，分别写入y和z
    """
    with Path(inputpath).open("rb") as f:
        header = read_uints(f, 6)
        height = header[0]
        width = header[1]
        mv_y_string_length = header[2]
        mv_z_string_length = header[3]
        y_string_length = header[4]
        z_string_length = header[5]

        mv_y_string = read_bytes(f, mv_y_string_length)
        mv_z_string = read_bytes(f, mv_z_string_length)
        y_string = read_bytes(f, y_string_length)
        z_string = read_bytes(f, z_string_length)

    return height, width, mv_y_string, mv_z_string, y_string, z_string
