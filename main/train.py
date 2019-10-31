from __future__ import print_function
import os
import sys

_FILE_PATH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_FILE_PATH, "../"))
import torch
import torchvision
from utils import utils
from torch.utils.data import DataLoader
from utils.logger import get_logger, setup_logger
from torchvision import transforms as tf
from datalayer.datalayer import Datalayer
from model import getModels
from loss import getLossFuns
from torch.utils.tensorboard import SummaryWriter
from solver.lr_scheduler import GeneralLR_Scheduler
from solver.make_optimizer import make_optimizer


class Trainer(object):
    def __init__(self):
        # 加载配置文件
        self.config = utils.loadYaml('../config/config.yaml')
        # 训练结果保存路径
        self.output_model_path = os.path.join('../output/', self.config['Misc']['OutputFolderName'])
        if self.config['Misc']['OutputFolderName']:
            utils.mkdir(self.output_model_path)
        else:
            raise IOError('请输入训练结果保存路径...')
        # gpu使用
        self.device = utils.set_gpu(self.config)
        self.summaryWriter = SummaryWriter(log_dir=self.output_model_path)

        # logger日志
        setup_logger(self.output_model_path)
        self.logger = get_logger()

    def train_one_epoch(self, model, criterion, optimizer, lr_scheduler, data_loader, device, epoch, print_freq):
        model.train()
        one_epoch_loss = 0
        one_epoch_acc = 0
        for i, (image, target) in enumerate(data_loader):

            image = image.to(device)
            target = target.to(device).long()
            pred = model(image)
            loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            acc = utils.accuracy(pred, target, topk=(1,))
            grid = torchvision.utils.make_grid(image, normalize=True, scale_each=True, padding=0)

            one_epoch_loss += loss
            one_epoch_acc += acc[0]
            if i % print_freq == 0:
                self.logger.debug('{0}({1}) - loss: {2:.6f} -lr:{3:.6f} - acc{4: .6f}'.format(i, epoch + 1,
                                                                                              loss.item(),
                                                                                              optimizer.param_groups[0][
                                                                                                  "lr"], acc[0]))
                self.summaryWriter.add_image('images', grid, len(data_loader) * epoch + i)
                self.summaryWriter.add_scalar('loss', loss, len(data_loader) * epoch + i)
                self.summaryWriter.add_scalar('acc', acc[0], len(data_loader) * epoch + i)
                self.summaryWriter.add_scalar('lr', optimizer.param_groups[0]["lr"], len(data_loader) * epoch + i)

        self.logger.debug(
            'Epoch{} finished ! mean loss: {}, mean acc: {}'.format(epoch, one_epoch_loss / len(data_loader),
                                                                    one_epoch_acc / len(data_loader)))

    def run(self):

        dataset = Datalayer(self.config, transform=tf.Compose([tf.ToTensor(), ]))
        data_loader = DataLoader(dataset=dataset,
                                 shuffle=True,
                                 batch_size=self.config['Dataset']['BatchSize'],
                                 num_workers=self.config['Dataset']['NumWorkers'],
                                 pin_memory=True)
        # 创建模型
        model = getModels(self.config['Model']['Name'], num_classes=self.config['Model']['NumClass'],
                          pretrained=self.config['Model']['IsPretrained']).to(self.device)
        # 创建损失函数
        criterion = getLossFuns()
        # 创建优化器
        optimizer = make_optimizer(cfg=self.config, model=model)
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

            self.train_one_epoch(model, criterion, optimizer, lr_scheduler, data_loader, self.device, epoch,
                                 print_freq=10)

            if self.output_model_path:
                checkpoint = {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch
                }

                utils.save_on_master(
                    checkpoint,
                    os.path.join(self.output_model_path, self.config['Misc']['BestModelName']))
                if epoch % self.config['Model']['OutputFreq'] == 0:
                    utils.save_on_master(
                        checkpoint,
                        os.path.join(self.output_model_path, 'model_{}.pth'.format(epoch)))


if __name__ == "__main__":
    traner = Trainer()
    traner.run()
