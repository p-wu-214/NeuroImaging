import torch

from torch.backends import cudnn
from torch.utils.data import DataLoader

from dataset.trends_data import TrendsDataset
from models.model import ResNetBasicBlock

import time

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    cudnn.benchmark = True

    max_epoch = 1
    trends_dataset = TrendsDataset()
    training_loader = DataLoader(trends_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=False)
    resnet = ResNetBasicBlock()
    criterion = torch.nn.L1Loss(reduction='mean')
    optimizer = torch.optim.Adam(resnet.parameters(), lr=1e-4)

    for epoch in range(max_epoch):
        # Training time
        print('Start iterating through')
        start = time.time()
        for index, batch in enumerate(training_loader):
            scans, _, targets = batch
            output = resnet(scans)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if index % 100 == 99:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, index + 1, running_loss / 100))
                running_loss = 0.0

            
        end = time.time()
        print('Time for one epoch: ' + str(end-start))

