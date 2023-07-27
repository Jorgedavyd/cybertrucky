import torch.nn as nn
from utils import *
import torch
import torchvision.transforms as tt

#Defining models classes
class Model(ImageClassificationBase):
    def __init__(self, model):
        super().__init__()
        self.model = model
    def forward(self, xb):
        return self.model(xb)

def Shufflenet(n_classes):
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
        nn.Linear(128, n_classes)
    )

    transform = tt.Compose([tt.ToTensor(), ShuffleNet_V2_X0_5_Weights.DEFAULT.transforms(antialias = True)])
    
    return transform, model

def Mobilenet(n_classes):
    from torchvision.models.quantization import mobilenet_v2, MobileNet_V2_QuantizedWeights

    torch.backends.quantized.engine = 'qnnpack'
    model = mobilenet_v2(pretrained=True, quantized=True)
    model = torch.jit.script(model)

    for param in model.parameters():
        param.requires_grad_=False

    model.fc = nn.Sequential(
        nn.Linear(),
        nn.Linear(),
        nn.Linear(, n_classes)
    )

    transform = tt.Compose([tt.ToTensor, MobileNet_V2_QuantizedWeights.IMAGENET1K_QNNPACK_V1.transforms(antialias=True)])
    return transform, model

