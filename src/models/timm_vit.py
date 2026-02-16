import timm
import torch
from torchvision import transforms


def load_timm_vit(model_name: str = "vit_base_patch16_224", pretrained: bool = True):
    """
    Loading a timm vit model for CPU inference

    It will return:
        model: torch.nn.Module (eval mode, on CPU)
        transform: torchvision.transforms.Compose for preprocessing PIL images
        class_names: list[str] or None (ImageNet labels if available in timm cfg)
    """
    device = torch.device("cpu")

    model = timm.create_model(model_name, pretrained=pretrained)
    model.eval()
    model.to(device)

    cfg = model.default_cfg
    # timm default_cfg input_size is usually (3, H, W)
    input_size = cfg.get("input_size", (3, 224, 224))
    h, w = input_size[-2], input_size[-1]

    mean = cfg.get("mean", (0.485, 0.456, 0.406))
    std = cfg.get("std", (0.229, 0.224, 0.225))

    transform = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # Some of these models have label_map
    class_names = cfg.get("label_map", None)

    return model, transform, class_names
