import torch
import config
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

from torch.utils.data import DataLoader
import torch_xla.distributed.parallel_loader as pl

from sklearn.model_selection import train_test_split

from dataset.trends_data import TrendsDataset
from models.model import ResNetBasicBlock
import time


# def train_with_cpu(train_loader, model, optimizer, criterion):
#     for epoch in range(max_epoch):
#         # Training time
#         start = time.time()
#         print('Start iterating through')
#         for index, batch in enumerate(train_loader):
#             optimizer.zero_grad()
#             scans = batch['scans']
#             targets = batch['targets']
#             output = model(scans)
#             del scans
#             loss = criterion(output, targets)
#             del targets
#             loss.backward()
#             xm.master_print(f'index: {index} loss: {loss.item()}')
#             xm.optimizer_step(optimizer)
#
#         end = time.time()
#         print('Time for one epoch: ' + str(end-start))

# def cpu_single_core():
#     batch_size = 4
#
#     max_epoch = 1
#
#     # XLA distributed sampler for more than 1 TPU
#     train_dataset = TrendsDataset(is_tpu)
#     train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=0, pin_memory=False)
#     criterion = torch.nn.L1Loss(reduction='mean')
#
#     print('Train without TPU')
#     resnet = ResNetBasicBlock(batch_size=batch_size)
#     optimizer = torch.optim.SGD(resnet.parameters(), lr=1e-4)
#     train_with_cpu(train_loader, resnet, optimizer, criterion)


def multi_core(index, flags):
    torch.manual_seed(flags['seed'])
    batch_size = 4
    device = xm.xla_device()
    max_epoch = 1

    #Only download X, Y on one process
    if not xm.is_master_ordinal():
        xm.rendezvous('download_only_once')

    train_dataset = TrendsDataset(mode='train')
    valid_dataset = TrendsDataset(mode='validation')
    val_loss = []

    if xm.is_master_ordinal():
        xm.rendezvous('download_only_once')

    # XLA distributed sampler for more than 1 TPU
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=flags['batch_size'],
        sampler=train_sampler,
        num_workers=flags['num_workers'],
        drop_last=True)
    valid_sampler = torch.utils.data.distributed.DistributedSampler(
        valid_dataset,
        num_replicas=xm.xrt_world_size(),
        rank=xm.get_ordinal(),
        shuffle=True)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=flags['batch_size'],
        sampler=valid_sampler,
        num_workers=flags['num_workers'],
        drop_last=True)

    print('Train with TPU')
    model = ResNetBasicBlock(batch_size=batch_size).to(device).train()

    criterion = torch.nn.L1Loss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=config.hyper_params['lr'], betas=config.hyper_params['betas'], eps=1e-08)
    # optimizer = torch.optim.SGD(model.parameters(), lr=config.hyper_params['lr'], momentum=0.9, nesterov=True)
    train_start = time.time()
    for epoch in range(flags['num_epochs']):
        # Training time
        train_pl_loader = pl.ParallelLoader(train_loader, [device]).per_device_loader(device)
        start = time.time()
        print('Start iterating through')
        for batch_num, batch in enumerate(train_pl_loader):
            optimizer.zero_grad()
            print("Process", index, "saving scan")
            scans = batch['scans']
            targets = batch['targets']
            output = model(scans)
            del scans
            loss = criterion(output, targets)
            del targets
            loss.backward()
            xm.master_print(f'training: index: {batch_num} loss: {loss.item()}')
            xm.optimizer_step(optimizer)
        print("Starting validation")
        del loss
        with torch.no_grad():
            valid_pl_loader = pl.ParallelLoader(valid_loader, [device]).per_device_loader(device)
            model.eval()
            for batch_num, batch in enumerate(valid_pl_loader):
                scans = batch['scans']
                targets = batch['targets']
                output = model(scans)
                del scans
                validation_loss = criterion(output, targets)
                del targets
                xm.master_print(f'validation: index: {batch_num} loss: {validation_loss.item()}')
                val_loss.append(validation_loss)
            del valid_pl_loader
    elapsed_train_time = time.time() - train_start
    print("Process", index, "finished training. Train time was:", elapsed_train_time)
    torch.save(f'epoch: {epoch}, state_dict: {model.state_dict()}, validation loss: {val_loss}, optimizer: {optimizer.state_dict()}',
               f'{config.hyper_params["model_save_path"]}/validation_loss_{time.time()}.txt');




if __name__ == '__main__':
    # TrendsDataset(mode='train')
    # TrendsDataset(mode='validation')
    flags = {}
    # Configures training (and evaluation) parameters
    flags['batch_size'] = config.hyper_params['batch']
    flags['num_workers'] = config.hyper_params['num_workers']
    flags['num_epochs'] = config.hyper_params['epochs']
    flags['seed'] = config.hyper_params['seed']

    xmp.spawn(multi_core, args=(flags,), nprocs=8, start_method='spawn')