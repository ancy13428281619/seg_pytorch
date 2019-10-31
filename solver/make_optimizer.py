import torch


def make_optimizer(cfg, model):
    # 支持的optimizer列表
    optimizer_dict = {
        'sgd': [torch.optim.SGD, dict(momentum=0.9)],
        'adam': [torch.optim.Adam, dict(betas=(0.9, 0.999))]
    }
    # 判断用户输入的optimizer是否在支持的列表内
    assert cfg['Solver']['OptimName'].lower() in [*optimizer_dict.keys()]
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg['Solver']['BaseLR']
        weight_decay = cfg['Solver']['WeightDecay']
        if "bias" in key:
            lr = cfg['Solver']['BaseLR'] * cfg['Solver']['BiasLRFactor']
            weight_decay = cfg['Solver']['WeightDecayBias']
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]
    # 构建optimizer
    optimizer = optimizer_dict[cfg['Solver']['OptimName'].lower()][0](params,
                                                                      **optimizer_dict[
                                                                          cfg['Solver']['OptimName'].lower()][1])
    return optimizer
