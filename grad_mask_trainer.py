import torch
import argparse
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms
from model import LeNet
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import TensorBoardLogger

class MaskGradSystem(LightningModule):
    def __init__(self, hparams):
        super(MaskGradSystem,self).__init__()
        self.hparams = hparams
        self.model = LeNet()
        self.criterion = nn.CrossEntropyLoss()
        self.mask = torch.nn.Parameter(torch.randn(self.model.mask_len))
        #self.c = torch.nn.Parameter(10*torch.randn(1))
        #self.mask = torch.nn.Parameter(torch.ones(self.model.mask_len))
        #self.mask = torch.nn.Parameter(10*torch.ones(self.model.mask_len))
    

    def configure_optimizers(self):
        #l = [self.mask]
        return optim.SGD(self.model.parameters(),lr=0.01)
    

    def prepare_data(self):
        transformss = transforms.Compose([transforms.ToTensor()])
        if self.hparams.dataset == 'mnist':
            self.train_dataset = MNIST(self.hparams.data_dir, train=True, download=True, transform=transformss)
            self.test_datset = MNIST(self.hparams.data_dir, train=False, download=True, transform=transformss)
            self.train_dataset, self.valid_dataset = torch.utils.data.random_split(self.train_dataset,[50000,10000])
        if self.hparams.dataset == 'cifar10':
            self.train_dataset = CIFAR10(self.hparams.data_dir, train=True, download=True, transform=transformss)
            self.test_datset = CIFAR10(self.hparams.data_dir, train=False, download=True, transform=transformss)
            self.train_dataset, self.valid_dataset = torch.utils.data.random_split(self.train_dataset,[40000,10000])
        #self.len_train_datset = len(self.train_dataset)
    
    
    def train_dataloader(self):
        loader = DataLoader(self.train_dataset, batch_size=32, num_workers=4)
        return loader
    
    
    def val_dataloader(self):
        loader = DataLoader(self.valid_dataset, batch_size=32, num_workers=4)
        return loader
    
    
    def test_dataloader(self):
        loader = DataLoader(self.test_dataset, batch_size=32, num_workers=4)
        return loader
    
    
    def validation_step(self,batch,batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat,y)
        y_pred = torch.argmax(y_hat,dim=-1)
        correct = torch.mean(1.0*(y_pred==y))
        return {'val_loss':loss,'num_correct':correct,'batch_size':torch.Tensor(x.size(0))}
    
    def test_step(self,batch,batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat,y)
        y_pred = torch.argmax(y_hat,dim=-1)
        correct = torch.mean(1.0*(y_pred==y))
        return {'test_loss':loss,'num_correct':correct,'batch_size':torch.Tensor(x.size(0))}
    
    
    def training_step(self,batch,batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat,y)
        return {'loss':loss}
    
    
    def validation_epoch_end(self,outputs):
        val_loss = torch.mean(torch.stack([output['val_loss'] for output in outputs])) 
        acc = torch.mean(torch.stack([output['num_correct'] for output in outputs]))
        batch_size = torch.sum(torch.cat([output['batch_size'] for output in outputs]))

        #acc = acc/10000
        #val_loss = val_loss/10000
        return {'progress_bar':{'accuracy':acc, 'val_loss':val_loss},'log':{'accuracy':acc, 'val_loss':val_loss}}

    def test_epoch_end(self,outputs):
        test_loss = torch.mean(torch.stack([output['test_loss'] for output in outputs])) 
        test_acc = torch.mean(torch.stack([output['num_correct'] for output in outputs]))
        batch_size = torch.sum(torch.cat([output['batch_size'] for output in outputs]))

        #test_acc = test_acc/10000
        #test_loss = test_loss/10000
        return {'progress_bar':{'accuracy':test_acc, 'val_loss':test_loss},'log':{'accuracy':test_acc, 'val_loss':test_loss}}
    
    
    def forward(self,x):
        return self.model(x,self.mask)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    parser.add_argument('--data_dir',type=str,help='dir to store train data')
    parser.add_argument('--dataset',type=str,help='cifar10|mnist')
    args = parser.parse_args()
    logger = TensorBoardLogger('tb_logs', name='my_model')
    trainer = Trainer.from_argparse_args(args)
    trainer.logger = logger
    system = MaskGradSystem(args)
    trainer.fit(system)
     


    
    


