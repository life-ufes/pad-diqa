import torchvision.models as models
import torch.nn as nn
from model.regressor import Regressor
import timm

def select_backbone_model(model_choice):
    if model_choice == 'mobilenet_v3_large':
        weights = models.MobileNet_V3_Large_Weights.DEFAULT
        base_model = models.mobilenet_v3_large(weights=weights)
        n_features = 960

        base_model.classifier = nn.Identity() 

    elif model_choice == 'mobilenet_v3_small':
        weights = models.MobileNet_V3_Small_Weights.DEFAULT
        base_model = models.mobilenet_v3_small(weights=weights)
        n_features = 576

        base_model.classifier = nn.Identity()

    elif model_choice == 'resnet50':
        weights = models.ResNet50_Weights.DEFAULT
        base_model = models.resnet50(weights=weights)
        n_features = 2048

        base_model.fc = nn.Identity()

    elif model_choice == 'davit_tiny':
        base_model = timm.create_model("davit_tiny.msft_in1k", pretrained=True)
        n_features = 768

        base_model.head.fc = nn.Identity()

    else:
        raise ValueError(f"Invalid model '{model_choice}'.")


    model = Regressor(
        base_model=base_model, 
        n_features=n_features, 
        neurons_reducer_block=256
    )
    
    return model