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
    training_loader = DataLoader(trends_dataset, batch_size=24, shuffle=True, num_workers=4, pin_memory=True)

    resnet = ResNetBasicBlock()
    criterion = torch.nn.L1Loss(reduction='mean')
    optimizer = torch.optim.Adam(resnet.parameters(), lr=1e-4)

    for epoch in range(max_epoch):
        # Training time
        print('Start iterating through')
        start = time.time()
        for test_batch in training_loader:
            test_batch = [data.cuda() for data in test_batch]
            # scans, X, targets = test_batch
            # start
        end = time.time()
        print(end-start)

