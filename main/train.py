from __future__ import print_function
import os
import sys

_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_FILE_PATH, "../"))
import torch
import utils.misc
import utils.metrics
import torchvision
from model import getModels
from loss import getLossFuns
from tqdm import tqdm as tqdm
from utils.meter import AverageValueMeter
from torch.utils.data import DataLoader
from utils.logger import setup_logger
from torchvision import transforms as tf
from datalayer.datalayer import Datalayer
from torch.utils.tensorboard import SummaryWriter
from solver.lr_scheduler import GeneralLR_Scheduler
from solver.make_optimizer import make_optimizer
from datalayer.augmentations import get_training_augmentation


class Trainer(object):
    def __init__(self):
        # 加载配置文件
        self.config = utils.misc.loadYaml('../config/config.yaml')
        # 训练结果保存路径
        self.output_model_path = os.path.join('../output/', self.config['Misc']['OutputFolderName'])
        if self.config['Misc']['OutputFolderName']:
            utils.misc.mkdir(self.output_model_path)
        else:
            raise IOError('请输入训练结果保存路径...')
        # gpu使用
        self.device = utils.misc.set_gpu(self.config)
        self.summaryWriter = SummaryWriter(log_dir=self.output_model_path)
        # logger日志
        self.logger = setup_logger(self.output_model_path)

    def _format_logs(self, logs):
        str_logs = ['{} - {:.4}'.format(k, v) for k, v in logs.items()]
        s = ', '.join(str_logs)
        return s

    def train_one_epoch(self, dataloader, model, criterion, metrics, optimizer, device, epoch, verbose=True):
        model.train()
        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {metric.__name__: AverageValueMeter() for metric in metrics}
        with tqdm(dataloader, desc='train', file=sys.stdout, disable=not (verbose)) as iterator:
            for x, y in iterator:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                y_pred = model.forward(x)
                loss = criterion(y_pred, y)
                loss.backward()
                optimizer.step()
                # update loss logs
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {criterion.__name__: loss_meter.mean}
                logs.update(loss_logs)
                # update metrics logs
                for metric_fn in metrics:
                    metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)
                if iterator.n % 100 == 0:
                    grid = torchvision.utils.make_grid(x, normalize=True, scale_each=True, padding=0)
                    self.summaryWriter.add_image('images', grid, len(dataloader) * epoch + iterator.n)
                    grid = torchvision.utils.make_grid(y, normalize=True, scale_each=True, padding=0)
                    self.summaryWriter.add_image('label', grid, len(dataloader) * epoch + iterator.n)
                    grid = torchvision.utils.make_grid(y_pred, normalize=True, scale_each=True, padding=0)
                    self.summaryWriter.add_image('pred', grid, len(dataloader) * epoch + iterator.n)
                    self.summaryWriter.add_scalar('loss', loss, len(dataloader) * epoch + iterator.n)
                    self.summaryWriter.add_scalar('lr', optimizer.param_groups[0]["lr"],
                                                  len(dataloader) * epoch + iterator.n)

        return logs

    def run(self):
        # 创建模型
        model, preprocessing_fn = getModels(self.config)
        model = model.to(self.device)
        # 加载数据
        dataset = Datalayer(self.config, get_training_augmentation(), transform=tf.Compose([tf.ToTensor(), ]),
                            target_transform=tf.Compose([tf.ToTensor(), ]))
        data_loader = DataLoader(dataset=dataset,
                                 shuffle=True,
                                 batch_size=self.config['Dataset']['BatchSize'],
                                 num_workers=self.config['Dataset']['NumWorkers'],
                                 pin_memory=True)
        # 创建损失函数
        criterion = getLossFuns(self.config)
        # 创建优化器
        optimizer = make_optimizer(cfg=self.config, model=model)
        metrics = [
            utils.metrics.IoUMetric(eps=1.),
            utils.metrics.FscoreMetric(eps=1.),
        ]

        lr_scheduler = GeneralLR_Scheduler(optimizer, self.config,
                                           max_iter=len(data_loader) * self.config['Dataset']['Epochs'])
        start_epoch = 0
        # 恢复训练
        if self.config['Model']['IsResume']:
            checkpoint = torch.load(os.path.join(self.output_model_path, self.config['Misc']['BestModelName']),
                                    map_location='cpu')
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch'] + 1
        # 开始训练
        print("Start training")
        for epoch in range(start_epoch, self.config['Dataset']['Epochs']):

            print('\nEpoch: {}'.format(epoch))
            self.train_one_epoch(dataloader=data_loader,
                                 model=model,
                                 criterion=criterion,
                                 metrics=metrics,
                                 optimizer=optimizer,
                                 device=self.device,
                                 epoch=epoch,
                                 verbose=True)
            # do something (save model, change lr, etc.)
            if self.output_model_path:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch
                }
                utils.misc.save_on_master(
                    checkpoint,
                    os.path.join(self.output_model_path, self.config['Misc']['BestModelName']))
                if epoch % self.config['Model']['OutputFreq'] == 0:
                    utils.misc.save_on_master(
                        checkpoint,
                        os.path.join(self.output_model_path, 'model_{}.pth'.format(epoch)))
                    del_weight_path = os.path.join(self.output_model_path,
                                                   'model_{}.pth'.format(
                                                       epoch - self.config['Model']['OutputFreq'] * self.config['Misc'][
                                                           'StoreWeightNum']))
                    if os.path.exists(del_weight_path):
                        os.remove(del_weight_path)


if __name__ == "__main__":
    traner = Trainer()
    traner.run()
