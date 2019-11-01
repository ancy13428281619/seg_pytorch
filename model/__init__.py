import segmentation_models_pytorch as smp


def getModels(cfg):
    net_name = cfg['Model']['Name']
    encoder_name = cfg['Model']['Encoder']
    encoder_weights = cfg['Model']['EncoderWeights']
    classes = cfg['Model']['NumClass']
    activation = cfg['Model']['Activation']

    if net_name.lower() == 'unet':
        net = smp.Unet
    elif net_name.lower() == 'fpn':
        net = smp.FPN
    elif net_name.lower() == 'linknet':
        net = smp.Linknet
    elif net_name.lower() == 'pspnet':
        net = smp.PSPNet
    else:
        raise ImportError

    preprocessing_fn = smp.encoders.get_preprocessing_fn(encoder_name, encoder_weights)
    return net(encoder_name=encoder_name,
               encoder_weights=encoder_weights,
               classes=classes,
               activation=activation), preprocessing_fn
