import segmentation_models_pytorch.utils.losses as losses


def getLossFuns(loss_name):
    if loss_name.lower() == 'bcedice':
        return losses.BCEDiceLoss(eps=1.)
    elif loss_name.lower() == 'bceJaccard':
        return losses.BCEJaccardLoss(eps=1.)
    elif loss_name.lower() == 'dice':
        return losses.DiceLoss(eps=1.)
    elif loss_name.lower() == 'jaccard':
        return losses.JaccardLoss(eps=1.)
    else:
        raise ImportError
