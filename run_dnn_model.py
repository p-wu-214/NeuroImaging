import torch
import config
import numpy as np
from torch.utils.data import DataLoader
from models.dnn_model import TrendsDNNModel
from dataset.trends_data import TrendsDataset


def dnn_train():
    training = TrendsDataset(mode='train')
    train_loader = DataLoader(training, batch_size=10, num_workers=8)
    # validation = TrendsDataset(mode='validation')
    # valid_loader = DataLoader(validation, batch_size=4, num_workers=8)
    model = TrendsDNNModel()

    criterion = torch.nn.L1Loss(reduction='mean')
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, betas=config.hyper_params['betas'], eps=1e-08)

    for epoch in range(100):
        model.train()
        loss_array = []
        for batch_num, batch in enumerate(train_loader):
            optimizer.zero_grad()
            data = batch['data']
            targets = batch['targets']
            output = model(data)
            loss = criterion(output, targets)
            loss.backward()
            optimizer.step()
            loss_array.append(loss.item())
        print(f'loss average: {np.array(loss_array).mean()}')



if __name__ == '__main__':
    # Configures training (and evaluation) parameters
    dnn_train()
