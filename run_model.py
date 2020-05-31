import sys
import torch
import torch_xla
import torch_xla.core.xla_model as xm
from torch.utils.data import DataLoader
import torch_xla.distributed.parallel_loader as pl

from dataset.trends_data import TrendsDataset
from models.model import ResNetBasicBlock
import time

def train_with_tpu(train_loader, model, optimizer, criterion):
    for epoch in range(max_epoch):
        # Training time
        loader = pl.ParallelLoader(train_loader, [device])
        start = time.time()
        print('Start iterating through')
        for index, batch in enumerate(loader.per_device_loader(device)):
            optimizer.zero_grad()
            scans = batch['scans']
            targets = batch['targets']
            output = model(scans)
            del scans
            loss = criterion(output, targets)
            del targets
            loss.backward()
            xm.master_print(f'index: {index} loss: {loss.item()}')
            xm.optimizer_step(optimizer)

        end = time.time()
        print('Time for one epoch: ' + str(end-start))

def train_with_cpu(train_loader, model, optimizer, criterion):
    for epoch in range(max_epoch):
        # Training time
        start = time.time()
        print('Start iterating through')
        for index, batch in enumerate(train_loader):
            optimizer.zero_grad()
            scans = batch['scans']
            targets = batch['targets']
            output = model(scans)
            del scans
            loss = criterion(output, targets)
            del targets
            loss.backward()
            xm.master_print(f'index: {index} loss: {loss.item()}')
            xm.optimizer_step(optimizer)

        end = time.time()
        print('Time for one epoch: ' + str(end-start))

if __name__ == '__main__':
    is_tpu = sys.argv[0].lower() == 'tpu'
    if is_tpu:
        device = xm.xla_device()

    batch_size = 4

    max_epoch = 1

    # XLA distributed sampler for more than 1 TPU
    train_dataset = TrendsDataset(is_tpu)
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=False)
    criterion = torch.nn.L1Loss(reduction='mean')

    if is_tpu:
        print('Train with TPU')
        resnet = ResNetBasicBlock(batch_size=batch_size).to(device)
        optimizer = torch.optim.SGD(resnet.parameters(), lr=1e-4)
        train_with_tpu(train_loader, resnet, optimizer, criterion)
    else:
        print('Train without TPU')
        resnet = ResNetBasicBlock(batch_size=batch_size)
        optimizer = torch.optim.SGD(resnet.parameters(), lr=1e-4)
        train_with_cpu(train_loader, resnet, optimizer, criterion)





