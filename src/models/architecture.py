from src.models.I_Coding import *


cfgs = {
    "ConvHyperprior": (320, 192),
    "ConvChARM": (320, 192),
    "SwinTHyperprior":  {
        'g_a': {
            'input_dim': 3,
            'embed_dim': [128, 192, 256, 320],
            'embed_out_dim': [192, 256, 320, None],
            'depths': [2, 2, 6, 2],
            'window_size': [8, 8, 8, 8]
        },
        'g_s': {
            'embed_dim': [320, 256, 192, 128],
            'embed_out_dim': [256, 192, 128, 3],
            'depths': [2, 6, 2, 2],
            'window_size': [8, 8, 8, 8],
        },
        'h_a': {
            'input_dim': 320,
            'embed_dim': [192, 192],
            'embed_out_dim': [192, None],
            'depths': [5, 1],
            'window_size': [4, 4]
        },
        'h_s': {
            'embed_dim': [192, 192],
            'embed_out_dim': [192, int(2 * 320)],
            'depths': [1, 5],
            'window_size': [4, 4],
        },
    },
    "SwinTChARM": {
        'g_a': {
            'input_dim': 3,
            'embed_dim': [128, 192, 256, 320],
            'embed_out_dim': [192, 256, 320, None],
            'depths': [2, 2, 6, 2],
            'window_size': [8, 8, 8, 8]
        },
        'g_s': {
            'embed_dim': [320, 256, 192, 128],
            'embed_out_dim': [256, 192, 128, 3],
            'depths': [2, 6, 2, 2],
            'window_size': [8, 8, 8, 8],
        },
        'h_a': {
            'input_dim': 320,
            'embed_dim': [192, 192],
            'embed_out_dim': [192, None],
            'depths': [5, 1],
            'window_size': [4, 4]
        },
        'h_s': {
            'embed_dim': [192, 192],
            'embed_out_dim': [192, int(2 * 320)],
            'depths': [1, 5],
            'window_size': [4, 4],
        }
    },
}


def getModel(model_name):

    if model_name == "ConvHyperprior":
        return ConvHyperprior(*cfgs["ConvHyperprior"])

    if model_name == "ConvChARM":
        return ConvChARM(*cfgs["ConvChARM"])

    if model_name == "SwinTHyperprior":
        return SwinTHyperprior(**cfgs["SwinTHyperprior"])

    if model_name == "SwinTChARM":
        return SwinTChARM(**cfgs["SwinTChARM"])

    raise ValueError(
        f"Unknown model_name: {model_name}. "
        f"Valid names: ConvHyperprior, ConvChARM, SwinTHyperprior, SwinTChARM"
    )
