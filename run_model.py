import torch
import torch_xla
import torch_xla.core.xla_model as xm
from torch.utils.data import DataLoader

from dataset.trends_data import TrendsDataset
from models.model import ResNetBasicBlock

import time

if __name__ == '__main__':
    device = xm.xla_device()

    batch_size = 4

    max_epoch = 1
    trends_dataset = TrendsDataset()
    training_loader = DataLoader(trends_dataset, batch_size=4, shuffle=True, num_workers=0, pin_memory=False)
    resnet = ResNetBasicBlock(batch_size=batch_size).to(device)
    criterion = torch.nn.L1Loss(reduction='mean')
    optimizer = torch.optim.SGD(resnet.parameters(), lr=1e-4)


    for epoch in range(max_epoch):
        # Training time
        print('Start iterating through')
        start = time.time()
        for index, batch in enumerate(training_loader):
            print("Training time")
            scans, _, targets = batch
            output = resnet(scans)
            del scans
            loss = criterion(output, targets)
            del output, targets
            loss.backward()
            xm.optimizer_step(optimizer)
            xm.mark_step()
            del loss

        end = time.time()
        print('Time for one epoch: ' + str(end-start))

