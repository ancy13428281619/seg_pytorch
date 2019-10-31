import segmentation_models_pytorch as smp


def getModels(net_name):
    if net_name.lower() == 'unet':
        return smp.Unet
    elif net_name.lower() == 'fpn':
        return smp.FPN
    elif net_name.lower() == 'linknet':
        return smp.Linknet
    elif net_name.lower() == 'pspnet':
        return smp.PSPNet
    else:
        raise ImportError
