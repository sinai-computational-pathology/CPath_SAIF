import os
import torch
import timm

def get_model():
    model = timm.create_model(
        "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
    )
    model.load_state_dict(torch.load('/sc/arion/projects/comppath_500k/SSLbenchmarks/code_tensors/pytorch_model.bin', map_location="cpu"), strict=True)
    return model
