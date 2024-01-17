from abc import ABC, abstractmethod


class Loss(ABC):
    @abstractmethod
    def __call__(self, *args, **kwargs):
        ...

    def __add__(self, other_loss):
        if isinstance(other_loss, CombinedLoss):
            return CombinedLoss(self, *other_loss.losses)
        else:
            return CombinedLoss(self, other_loss)


class CombinedLoss:
    def __init__(self, *losses):
        self.losses = losses

    def __call__(self, *args, **kwargs):
        loss = sum([loss(*args, **kwargs) for loss in self.losses])
        return loss

    def __add__(self, other_loss):
        if isinstance(other_loss, CombinedLoss):
            return CombinedLoss(*self.losses, *other_loss.losses)
        else:
            return CombinedLoss(*self.losses, other_loss)