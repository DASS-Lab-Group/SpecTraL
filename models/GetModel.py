import timm
from .structure import *
from .structuremixer import *

_PRETRAINED_CACHE = {}


def build_promptmodel(num_classes=2, edge_size=224, modeltype='ViT', patch_size=16,
                      Prompt_Token_num=10, VPT_type="Shallow", depth = 12,depth_cls=0):
    if modeltype == 'ViT':
        basic_model = timm.create_model('vit_base_patch16_224',pretrained=True,)

        model = VPT_ViT(img_size=edge_size, patch_size=patch_size, Prompt_Token_num=Prompt_Token_num,
                        VPT_type=VPT_type,depth=depth,depth_cls=depth_cls)
        model.load_state_dict(basic_model.state_dict(), strict = False)
        model.New_CLS_head(num_classes)
        model.Freeze()
# below elif block added by ~ Tanishk
    elif modeltype == 'ViT_L':
        timm_name = 'vit_large_patch16_224.augreg_in21k'
        if timm_name not in _PRETRAINED_CACHE:
            print(f"[GetModel] Loading {timm_name} pretrained weights (once)...")
            basic_model = timm.create_model(timm_name, pretrained=True)
            # Store on CPU to avoid holding a full GPU copy just for weight loading
            _PRETRAINED_CACHE[timm_name] = {
                k: v.cpu() for k, v in basic_model.state_dict().items()
                if not k.startswith('head.')
            }
            del basic_model
            torch.cuda.empty_cache()
        
        model = VPT_ViT(img_size=edge_size, patch_size=patch_size, Prompt_Token_num=Prompt_Token_num,
                        VPT_type=VPT_type, depth=depth, depth_cls=depth_cls,
                        embed_dim=1024, num_heads=16)
        model.load_state_dict(_PRETRAINED_CACHE[timm_name], strict=False)
        model.New_CLS_head(num_classes)
        model.Freeze()

    elif modeltype == 'mixer':
        basic_model = timm.create_model('mixer_b16_224',pretrained=True,)
        model = Mixer(embed_dim=768,num_blocks=depth,depth_cls=depth_cls)
        model.load_state_dict(basic_model.state_dict(), strict = False)
        model.reset_classifier(num_classes)
        
    else:
        print("The model is not difined now！！")
        return -1

    return model
