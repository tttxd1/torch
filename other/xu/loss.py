import torch
import torch.nn as nn

class dice_bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.00000001  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        # score=[0.0,0.0]
        score = (2. * intersection + smooth) / (i + j + smooth)
        score1 = (intersection + smooth) / (i + j - intersection + smooth)#iou
        # print(score[1].mean())
        return score.mean(),score1

        # return score[1]

    def soft_dice_loss(self, y_true, y_pred):
        loss1, iou = self.soft_dice_coeff(y_true, y_pred)
        loss = 1 - loss1
        # loss = self.soft_dice_coeff(y_true, y_pred)
        return loss,iou

    def __call__(self, y_true, y_pred):
        a = self.bce_loss(y_pred, y_true)
        b,iou = self.soft_dice_loss(y_true, y_pred)
        return a + b,iou
        # return b