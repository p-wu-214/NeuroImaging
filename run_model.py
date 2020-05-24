import torch

from torch.backends import cudnn
from torch.utils.data import DataLoader

from dataset.trends_data import TrendsDataset
from models.model import ResNetBasicBlock

import time

if __name__ == '__main__':
    batch_size = 4
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    cudnn.benchmark = True

    max_epoch = 1
    trends_dataset = TrendsDataset()
    training_loader = DataLoader(trends_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=False)
    resnet = ResNetBasicBlock(batch_size=batch_size)
    criterion = torch.nn.L1Loss(reduction='mean')
    optimizer = torch.optim.SGD(resnet.parameters(), lr=1e-4)

    torch.backends.cudnn.enabled = False

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
            optimizer.step()
            del loss
            
        end = time.time()
        print('Time for one epoch: ' + str(end-start))

