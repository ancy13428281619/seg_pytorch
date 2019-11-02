from .unet import Unet
from .linknet import Linknet
from .fpn import FPN
from .pspnet import PSPNet
from . import encoders


def getModels(cfg):
    net_name = cfg['Model']['Name']
    encoder_name = cfg['Model']['Encoder']
    encoder_weights = cfg['Model']['EncoderWeights']
    classes = cfg['Model']['NumClass']
    activation = cfg['Model']['Activation']

    if net_name.lower() == 'unet':
        net = Unet
    elif net_name.lower() == 'fpn':
        net = FPN
    elif net_name.lower() == 'linknet':
        net = Linknet
    elif net_name.lower() == 'pspnet':
        net = PSPNet
    else:
        raise ImportError

    preprocessing_fn = encoders.get_preprocessing_fn(encoder_name, encoder_weights)
    return net(encoder_name=encoder_name,
               encoder_weights=encoder_weights,
               classes=classes,
               activation=activation), preprocessing_fn
