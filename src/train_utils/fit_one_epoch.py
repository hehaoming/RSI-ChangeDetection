import torch
from tqdm import tqdm
import os


def fit_one_epoch(module=None, train_data=None, test_data=None, optimizer=None, loss_dice=None, loss_bce=None,
                  is_cuda=False, val=None, epoch=None):
    val.reset()
    loss_set = 0
    module.train()
    print("start train Epoch: {}".format(epoch), end='\n')
    with tqdm(total=1600, desc="Epoch{}".format(epoch), postfix=dict, mininterval=0.3) as pbar:
        for step, (x1, x2, x1_label, x2_label, change) in enumerate(train_data):
            if is_cuda:
                x1 = x1.cuda()
                x2 = x2.cuda()
                x1_label = x1_label.cuda()
                x1_label = x1_label.view(1, -1)
                x2_label = x2_label.cuda()
                x2_label = x2_label.view(1, -1)
                change = change.cuda()
                change = change.view(1, -1)
            else:
                x1_label = x1_label.view(1, -1)
                x2_label = x2_label.view(1, -1)
                change = change.view(1, -1)

            # 清零梯度
            optimizer.zero_grad()
            # 前向传播
            pred = module(x1, x2)
            # x1
            x1_pred = torch.clamp(pred[4][0], min=0, max=1).view(1, -1)
            x1_pred[x1_pred < 0.5] = 0
            x1_pred[x1_pred >= 0.5] = 1
            x1_tp = torch.sum(torch.mul(x1_pred, x1_label), dim=-1)
            x1_fp = torch.sum(torch.mul(x1_pred, 1 - x1_label), dim=-1)
            x1_fn = torch.sum(torch.mul(1 - x1_pred, x1_label), dim=-1)
            # x2
            x2_pred = torch.clamp(pred[4][1], min=0, max=1).view(1, -1)
            x2_pred[x2_pred < 0.5] = 0
            x2_pred[x2_pred >= 0.5] = 1
            x2_tp = torch.sum(torch.mul(x2_pred, x2_label), dim=-1)
            x2_fp = torch.sum(torch.mul(x2_pred, 1 - x2_label), dim=-1)
            x2_fn = torch.sum(torch.mul(1 - x2_pred, x2_label), dim=-1)
            # change
            change_pred = torch.clamp(pred[4][2], min=0, max=1).view(1, -1)
            change_pred[change_pred < 0.5] = 0
            change_pred[change_pred >= 0.5] = 1
            change_tp = torch.sum(torch.mul(change_pred, change), dim=-1)
            change_fp = torch.sum(torch.mul(change_pred, 1 - change), dim=-1)
            change_fn = torch.sum(torch.mul(1 - change_pred, change), dim=-1)
            # loss
            x1_dice_loss = loss_dice(x1_pred, x1_label)
            x2_dice_loss = loss_dice(x2_pred, x2_label)
            change_dice_loss = loss_dice(change_pred, change)
            loss = 0.2 * x1_dice_loss + 0.2 * x2_dice_loss + 0.6 * change_dice_loss
            # 反向传播
            loss.backward()
            optimizer.step()
            loss_set += loss.item()
            # 计算指标
            tp = 0.2 * x1_tp + 0.2 * x2_tp + 0.6 * change_tp
            fp = 0.2 * x2_fp + 0.2 * x2_fp + 0.6 * change_fp
            fn = 0.2 * x1_fn + 0.2 * x2_fn + 0.6 * change_fn
            val.add(tp, fp, fn)
            pres, recall, f1 = val.value()
            pbar.set_postfix(**{
                "step": step + 1,
                "loss": loss_set / (step + 1),
                "pres": pres.item(),
                "recall": recall.item(),
                "F1": f1.item()
            })
            pbar.update(1)

    module.eval()
    val.reset()
    loss_set = 0
    print("start test Epoch:{}".format(epoch))
    with tqdm(total=400, desc="Epoch{}".format(epoch), postfix=dict, mininterval=0.3) as pbar:
        for step, (x1, x2, x1_label, x2_label, change) in enumerate(test_data):
            if is_cuda:
                x1 = x1.cuda()
                x2 = x2.cuda()
                x1_label = x1_label.cuda()
                x1_label = x1_label.view(1, -1)
                x2_label = x2_label.cuda()
                x2_label = x2_label.view(1, -1)
                change = change.cuda()
                change = change.view(1, -1)
            else:
                x1_label = x1_label.view(1, -1)
                x2_label = x2_label.view(1, -1)
                change = change.view(1, -1)

            with torch.no_grad():
                # 前向传播
                pred = module(x1, x2)
                # x1
                x1_pred = torch.clamp(pred[4][0], min=0, max=1).view(1, -1)
                x1_pred[x1_pred < 0.5] = 0
                x1_pred[x1_pred >= 0.5] = 1
                x1_tp = torch.sum(torch.mul(x1_pred, x1_label), dim=-1)
                x1_fp = torch.sum(torch.mul(x1_pred, 1 - x1_label), dim=-1)
                x1_fn = torch.sum(torch.mul(1 - x1_pred, x1_label), dim=-1)
                # x2
                x2_pred = torch.clamp(pred[4][1], min=0, max=1).view(1, -1)
                x2_pred[x2_pred < 0.5] = 0
                x2_pred[x2_pred >= 0.5] = 1
                x2_tp = torch.sum(torch.mul(x2_pred, x2_label), dim=-1)
                x2_fp = torch.sum(torch.mul(x2_pred, 1 - x2_label), dim=-1)
                x2_fn = torch.sum(torch.mul(1 - x2_pred, x2_label), dim=-1)
                # change
                change_pred = torch.clamp(pred[4][2], min=0, max=1).view(1, -1)
                change_pred[change_pred < 0.5] = 0
                change_pred[change_pred >= 0.5] = 1
                change_tp = torch.sum(torch.mul(change_pred, change), dim=-1)
                change_fp = torch.sum(torch.mul(change_pred, 1 - change), dim=-1)
                change_fn = torch.sum(torch.mul(1 - change_pred, change), dim=-1)
                # loss
                x1_dice_loss = loss_dice(x1_pred, x1_label)
                x2_dice_loss = loss_dice(x2_pred, x2_label)
                change_dice_loss = loss_dice(change_pred, change)
                loss = 0.2 * x1_dice_loss + 0.2 * x2_dice_loss + 0.6 * change_dice_loss
                loss_set += loss.item()

                # 计算指标
                tp = 0.2 * x1_tp + 0.2 * x2_tp + 0.6 * change_tp
                fp = 0.2 * x2_fp + 0.2 * x2_fp + 0.6 * change_fp
                fn = 0.2 * x1_fn + 0.2 * x2_fn + 0.6 * change_fn
                val.add(tp, fp, fn)
                pres, recall, f1 = val.value()
                pbar.set_postfix(**{
                    "step": step + 1,
                    "loss": loss_set / (step + 1),
                    "pres": pres.item(),
                    "recall": recall.item(),
                    "F1": f1.item()
                })
                pbar.update(1)
    if epoch > 0 and epoch % 10 == 0:
        weight_root = r"F:\mahcenglin\src\weights"
        weight_path = 'epoch_{}-loss_{:.3f}-F1_{:.3f}.pth'.format(epoch, loss_set, f1)

        torch.save(module.state_dict(), os.path.join(weight_root, weight_path))
