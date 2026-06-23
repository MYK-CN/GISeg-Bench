import ml_collections
import os
import wget


os.makedirs('./weights', exist_ok=True)


# HiFormer-S Configs
def get_hiformer_s_configs():
    
    cfg = ml_collections.ConfigDict()

    # Swin Transformer Configs
    cfg.swin_pyramid_fm = [96, 192, 384]
    cfg.image_size = 224
    cfg.patch_size = 4
    cfg.num_classes = 9
    swin_path = './weights/swin_tiny_patch4_window7_224.pth'
    if not os.path.isfile(swin_path):
        print('Downloading Swin-transformer model ...')
        try:
            wget.download("https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth", swin_path)
        except Exception as e:
            print(f'[Warn] Swin download failed: {e}. Using random init.')
            swin_path = None
    cfg.swin_pretrained_path = swin_path

    # CNN Configs
    cfg.cnn_backbone = "resnet34"
    cfg.cnn_pyramid_fm  = [64, 128, 256]
    cfg.resnet_pretrained = True

    # DLF Configs
    cfg.depth = [[1, 1, 0]]
    cfg.num_heads = (3, 3)
    cfg.mlp_ratio=(1., 1., 1.)
    cfg.drop_rate = 0.
    cfg.attn_drop_rate = 0.
    cfg.drop_path_rate = 0.
    cfg.qkv_bias = True
    cfg.qk_scale = None
    cfg.cross_pos_embed = True

    return cfg


# HiFormer-B Configs
def get_hiformer_b_configs():

    cfg = ml_collections.ConfigDict()
    
    # Swin Transformer Configs
    cfg.swin_pyramid_fm = [96, 192, 384]
    cfg.image_size = 224
    cfg.patch_size = 4
    cfg.num_classes = 9
    swin_path = './weights/swin_tiny_patch4_window7_224.pth'
    if not os.path.isfile(swin_path):
        print('Downloading Swin-transformer model ...')
        try:
            wget.download("https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth", swin_path)
        except Exception as e:
            print(f'[Warn] Swin download failed: {e}. Using random init.')
            swin_path = None
    cfg.swin_pretrained_path = swin_path

    # CNN Configs
    cfg.cnn_backbone = "resnet50"
    cfg.cnn_pyramid_fm  = [256,512,1024]
    cfg.resnet_pretrained = True

    # DLF Configs
    cfg.depth = [[1, 2, 0]]
    cfg.num_heads = (6, 12)
    cfg.mlp_ratio=(2., 2., 1.)
    cfg.drop_rate = 0.
    cfg.attn_drop_rate = 0.
    cfg.drop_path_rate = 0.
    cfg.qkv_bias = True
    cfg.qk_scale = None
    cfg.cross_pos_embed = True

    return cfg


# HiFormer-L Configs
def get_hiformer_l_configs():
    cfg = ml_collections.ConfigDict()

    # Swin Transformer Configs
    cfg.swin_pyramid_fm = [96, 192, 384]
    cfg.image_size = 224
    cfg.patch_size = 4
    cfg.num_classes = 9
    swin_path = './weights/swin_tiny_patch4_window7_224.pth'
    if not os.path.isfile(swin_path):
        print('Downloading Swin-transformer model ...')
        try:
            wget.download("https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth", swin_path)
        except Exception as e:
            print(f'[Warn] Swin download failed: {e}. Using random init.')
            swin_path = None
    cfg.swin_pretrained_path = swin_path

    # CNN Configs
    cfg.cnn_backbone = "resnet34"
    cfg.cnn_pyramid_fm  = [64, 128, 256]
    cfg.resnet_pretrained = True

    # DLF Configs
    cfg.depth = [[1, 4, 0]]
    cfg.num_heads = (6, 6)
    cfg.mlp_ratio=(4., 4., 1.)
    cfg.drop_rate = 0.
    cfg.attn_drop_rate = 0.
    cfg.drop_path_rate = 0.
    cfg.qkv_bias = True
    cfg.qk_scale = None
    cfg.cross_pos_embed = True

    return cfg