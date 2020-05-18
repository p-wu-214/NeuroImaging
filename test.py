import torch
if __name__ == '__main__':
    t = torch.rand(4, 4)
    print(t.shape)
    print(t.view(-1, 20).shape)