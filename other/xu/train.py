import torch
import torch.utils.data as data
import os
from time import time
from model.Swin_W import SwinTransformer
from framework import MyFrame
from loss import dice_bce_loss
from data import ImageFolder

SHAPE = (1024, 1024)
ROOT = 'dataset/train/'
imagelist = filter(lambda x: x.find('sat')!=-1, os.listdir(ROOT))
trainlist = list(map(lambda x: x[:-8], imagelist))

NAME = 'deNet'
BATCHSIZE_PER_CARD = 2
solver = MyFrame(SwinTransformer, dice_bce_loss, 2e-4)
batchsize = BATCHSIZE_PER_CARD
dataset = ImageFolder(trainlist, ROOT)
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batchsize,
    shuffle=False,
    num_workers=2)
tic = time()
no_optim = 0
total_epoch = 300
train_epoch_best_loss = 100.
pre_weight_path = 'pre_weight/log01_dink34.th'
def main():
    #接上次训练
    solver.load('weight/' + NAME + '.th')
    print("load model success")
    # solver.update_lr(100.0, factor=True)
    for epoch in range(1, total_epoch+1):
        data_loader_iter = iter(data_loader)
        train_epoch_loss = 0
        ious = 0
        print("epoch ", epoch, ": Start!")
        ite = 0
        for img, mask in data_loader_iter:
            ite += 1
            if ite%62==0:
                print('{:.2%}'.format(ite/6226))
            # print('',end='\r')
            solver.set_input(img, mask)
            train_loss,iou = solver.optimize()
            train_epoch_loss += train_loss
            ious += iou
        train_epoch_loss /= len(data_loader_iter)
        ious /= len(data_loader_iter)
        print('***********')
        print('epoch:', epoch, '    time:', int(time()-tic))
        print('train_loss:', train_epoch_loss)
        print('iou:',ious)
        print('SHAPE', SHAPE)

        global train_epoch_best_loss
        if train_epoch_loss >= train_epoch_best_loss:
            no_optim += 1
        else:
            no_optim = 0
            train_epoch_best_loss = train_epoch_loss
            solver.save('weight/' + NAME + '.th')

        if no_optim > 6:
            print('early stop at %d epoch' % epoch)
            break

        if no_optim > 3:
            if solver.old_lr < 5e-7:
                break

            solver.load('weight/' + NAME + '.th')
            solver.update_lr(5.0, factor=True)

    print('Finish!')
if __name__ == "__main__":
    main()