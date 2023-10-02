import torch


class CrossEntropyLoss:
    def __init__(self, tag, weight):
        self.tag = tag
        self.weight = weight
        self.criterion = torch.nn.CrossEntropyLoss()

    def __call__(self, prediction, y):
        loss = self.criterion(prediction, y)
        return loss * self.weight
