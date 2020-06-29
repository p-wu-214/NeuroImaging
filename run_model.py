import torch
import config

import numpy as np
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp
from torch.utils.data import DataLoader
import torch_xla.distributed.parallel_loader as pl

from dataset.trends_data import MRIDataset
# from dataset.trends_data import TrendsDataset

from models.model import NeuroImageModel
from models.dnn_model import TrendsDNNModel
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
#     train_dataset = MRIDataset(is_tpu)
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

    train_dataset = MRIDataset(mode='train')
    valid_dataset = MRIDataset(mode='validation')
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
    model = NeuroImageModel().to(device).train()

    criterion = torch.nn.L1Loss(reduction='mean')
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.hyper_params['lr'], betas=config.hyper_params['betas'], eps=1e-08)
    optimizer = torch.optim.SGD(model.parameters(), lr=config.hyper_params['lr'], momentum=0.9, nesterov=True)
    train_start = time.time()
    file = open('loss', 'w')
    average = 0
    for epoch in range(flags['num_epochs']):
        # Training time
        train_pl_loader = pl.ParallelLoader(train_loader, [device]).per_device_loader(device)
        start = time.time()
        average = 0
        count = 0
        for batch_num, batch in enumerate(train_pl_loader):
            optimizer.zero_grad()
            print("Process", index, "saving scan")
            scans = batch['scans']
            fnc = batch['fnc']
            sbm = batch['sbm']
            targets = batch['targets']
            output = model(fnc, sbm, scans)
            del scans
            loss = criterion(output, targets)
            del targets
            loss.backward()
            xm.master_print(f'training: index: {batch_num} loss: {loss.item()}')
            count = count + 1
            average = average + loss.item()
            xm.optimizer_step(optimizer, barrier=True)
        print(f'Training loss for epoch: {epoch} average of: {average/count} with count {count}')
        file.write(f'Training loss for epoch: {epoch} average of: {average/count} with count {count}')
        # average = 0
        # count = 0
        del loss
        # with torch.no_grad():
        #     valid_pl_loader = pl.ParallelLoader(valid_loader, [device]).per_device_loader(device)
        #     model.eval()
        #     for batch_num, batch in enumerate(valid_pl_loader):
        #         scans = batch['scans']
        #         fnc = batch['fnc']
        #         sbm = batch['sbm']
        #         targets = batch['targets']
        #         output = model(fnc, sbm, scans)
        #         del scans
        #         validation_loss = criterion(output, targets)
        #         del targets
        #         xm.master_print(f'validation: index: {batch_num} loss: {validation_loss.item()}')
        #         count = count + 1
        #         average = average + validation_loss.item()
        #         val_loss.append(validation_loss)
        #     del valid_pl_loader
    print(f'Training loss for epoch: {epoch} average of: {average/count} with count {count}')
    file.write(f'Training loss for epoch: {epoch} average of: {average/count} with count {count}')
    elapsed_train_time = time.time() - train_start
    print("Process", index, "finished training. Train time was:", elapsed_train_time)
    torch.save(f'epoch: {epoch}, state_dict: {model.state_dict()}, validation loss: {val_loss}, optimizer: {optimizer.state_dict()}',
               f'{config.hyper_params["model_save_path"]}/validation_loss_{time.time()}.txt')



# def dnn_train():
#     training = TrendsDataset(mode='train')
#     validation = TrendsDataset(mode='validation')
#     train_loader = DataLoader(training, batch_size=1, num_workers=8)
#     valid_loader = DataLoader(validation, batch_size=4, num_workers=8)
#     model = TrendsDNNModel()
#
#     criterion = torch.nn.L1Loss(reduction='mean')
#     optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
#     # optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=config.hyper_params['betas'], eps=1e-08)
#
#     for epoch in range(30):
#         model.train()
#         loss_array = []
#         for batch_num, batch in enumerate(train_loader):
#             optimizer.zero_grad()
#             sbm_data = batch['sbm_data']
#             fnc_data = batch['fnc_data']
#             targets = batch['targets']
#             output = model(fnc_data, sbm_data)
#             loss = criterion(output, targets)
#             loss.backward()
#             optimizer.step()
#             loss_array.append(loss.item())
#         print(f'loss average: {np.array(loss_array).mean()}')



if __name__ == '__main__':
    # Configures training (and evaluation) parameters
    flags = {}
    flags['batch_size'] = config.hyper_params['batch']
    flags['num_workers'] = config.hyper_params['num_workers']
    flags['num_epochs'] = config.hyper_params['epochs']
    flags['seed'] = config.hyper_params['seed']

    xmp.spawn(multi_core, args=(flags,), nprocs=8, start_method='spawn')