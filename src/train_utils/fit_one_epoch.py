from tqdm import tqdm

from src.utils import *


def fit_one_epoch(module, train_data, test_data, args, optimizer, loss_func,
                  is_cuda, epoch, train_step, test_step, device):
    module.train()
    print("start train Epoch: {}".format(epoch), end='\n')

    metricRe_s1 = MetricRe(device)
    metricRe_s2 = MetricRe(device)
    metricRe_ch = MetricRe(device)
    lossRe = LossRe(device)

    with tqdm(total=train_step, desc="Epoch{}".format(epoch), postfix=dict, mininterval=0.3) as pbar:
        for step, (x1, x2, x1_label, x2_label, change) in enumerate(train_data):
            if is_cuda:
                x1 = x1.cuda()
                x2 = x2.cuda()
                x1_label = x1_label.cuda()
                # x1_label = x1_label.view(x1.size(0), -1)
                x2_label = x2_label.cuda()
                # x2_label = x2_label.view(x2.size(0), -1)
                change = change.cuda()
                # change = change.view(change.size(0), -1)
            # else:
            #     x1_label = x1_label.view(args.batch_size, -1)
            #     x2_label = x2_label.view(args.batch_size, -1)
            #     change = change.view(args.batch_size, -1)

            # 前向传播
            x1_pred, x2_pred, change_pred = module(x1, x2)

            # loss
            x1_hybrid_loss = loss_func(x1_pred, x1_label)
            x2_hybrid_loss = loss_func(x2_pred, x2_label)
            change_hybrid_loss = loss_func(change_pred, change)
            loss = 0.2 * x1_hybrid_loss + 0.2 * x2_hybrid_loss + 0.6 * change_hybrid_loss
            lossRe.add(loss, x1.size(0))
            # x1

            x1_tp, x1_fp, x1_fn = metric(x1_pred, x1_label)
            metricRe_s1.add(x1_tp, x1_fp, x1_fn)

            # x2
            x2_tp, x2_fp, x2_fn = metric(x2_pred, x2_label)
            metricRe_s2.add(x2_tp, x2_fp, x2_fn)
            # change

            change_tp, change_fp, change_fn = metric(change_pred, change)
            metricRe_ch.add(change_tp, change_fp, change_fn)

            # 清零梯度
            optimizer.zero_grad()
            # 反向传播
            loss.backward()
            optimizer.step()

            # 计算指标

            x1_metric, x2_metric, c_metric = metricRe_s1.value(), metricRe_s2.value(), metricRe_ch.value()
            pbar.set_postfix(**{
                "step": step + 1,
                "loss": lossRe.value(),
                "pres": x1_metric[0] * 0.2 + x2_metric[0] * 0.2 + c_metric[0] * 0.6,
                "recall": x1_metric[1] * 0.2 + x2_metric[1] * 0.2 + c_metric[1] * 0.6,
                "F1": x1_metric[2] * 0.2 + x2_metric[2] * 0.2 + c_metric[2] * 0.6
            })
            pbar.update(1)

    module.eval()
    metricRe_s1.reset()
    metricRe_s2.reset()
    metricRe_ch.reset()
    lossRe.reset()

    print("start test Epoch:{}".format(epoch))
    with tqdm(total=test_step, desc="Epoch{}".format(epoch), postfix=dict, mininterval=0.3) as pbar:
        for step, (x1, x2, x1_label, x2_label, change) in enumerate(test_data):
            if is_cuda:
                x1 = x1.cuda()
                x2 = x2.cuda()
                x1_label = x1_label.cuda()
                # x1_label = x1_label.view(1, -1)
                x2_label = x2_label.cuda()
                # x2_label = x2_label.view(1, -1)
                change = change.cuda()
                # change = change.view(1, -1)
            # else:
            #     x1_label = x1_label.view(1, -1)
            #     x2_label = x2_label.view(1, -1)
            #     change = change.view(1, -1)

            with torch.no_grad():
                # 前向传播
                x1_pred, x2_pred, change_pred = module(x1, x2)
                # loss
                x1_hybrid_loss = loss_func(x1_pred, x1_label)
                x2_hybrid_loss = loss_func(x2_pred, x2_label)
                change_hybrid_loss = loss_func(change_pred, change)
                loss = 0.2 * x1_hybrid_loss + 0.2 * x2_hybrid_loss + 0.6 * change_hybrid_loss
                lossRe.add(loss, x1.size(0))
                # x1

                x1_tp, x1_fp, x1_fn = metric(x1_pred, x1_label)
                metricRe_s1.add(x1_tp, x1_fp, x1_fn)

                # x2
                x2_tp, x2_fp, x2_fn = metric(x2_pred, x2_label)
                metricRe_s2.add(x2_tp, x2_fp, x2_fn)
                # change

                change_tp, change_fp, change_fn = metric(change_pred, change)
                metricRe_ch.add(change_tp, change_fp, change_fn)

                x1_metric, x2_metric, c_metric = metricRe_s1.value(), metricRe_s2.value(), metricRe_ch.value()
                pbar.set_postfix(**{
                    "step": step + 1,
                    "loss": lossRe.value(),
                    "pres": x1_metric[0] * 0.2 + x2_metric[0] * 0.2 + c_metric[0] * 0.6,
                    "recall": x1_metric[1] * 0.2 + x2_metric[1] * 0.2 + c_metric[1] * 0.6,
                    "F1": x1_metric[2] * 0.2 + x2_metric[2] * 0.2 + c_metric[2] * 0.6
                })
                pbar.update(1)
        # if  epoch % args.write_step == 0:
        #
        #     weight_path = 'epoch_{}-loss_{:.3f}-F1_{:.3f}.pth'.format(epoch, loss_set.item(), f1.item())
        #     torch.save(module.state_dict(), os.path.join(args.root, weight_path))
