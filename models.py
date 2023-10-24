import torch.nn as nn
from utils import *
import torch
import torchvision.transforms as tt

#Defining models classes
def Shufflenet():
    from torchvision.models import shufflenet_v2_x0_5, ShuffleNet_V2_X0_5_Weights
    model = shufflenet_v2_x0_5()

    for param in model.parameters():
        param.requires_grad_ = False

    model.fc = nn.Sequential(
        nn.Linear(1024, 512),
        nn.ReLU(),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 1),
        nn.Sigmoid()
    )

    transform = tt.Compose([tt.ToTensor(), ShuffleNet_V2_X0_5_Weights.DEFAULT.transforms(antialias = True)])
    
    return transform, model

def Mobilenet():
    from torchvision.models.quantization import mobilenet_v2, MobileNet_V2_QuantizedWeights

    torch.backends.quantized.engine = 'qnnpack'

    model = mobilenet_v2(weights=MobileNet_V2_QuantizedWeights.DEFAULT, quantize=True)

    for param in model.parameters():
        param.requires_grad = False
    
    model.classifier = nn.Sequential(
        nn.quantized.Linear(1280, 640),
        nn.quantized.LeakyReLU(0.1, zero_point=0),
        nn.quantized.Linear(640, 320),
        nn.quantized.LeakyReLU(0.2,  zero_point=0),
        nn.quantized.Linear(320, 1),
        nn.quantized.Sigmoid(1, 0))

    transform = tt.Compose([tt.ToTensor(), MobileNet_V2_QuantizedWeights.IMAGENET1K_QNNPACK_V1.transforms(antialias=True)])
    return transform, model

