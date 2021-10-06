import argparse

import torch
from torch.optim import Adam

from dataset.RSIdataset import get_data
from losses import HybridLoss
from module import Sim_Att_UNet
from train_utils.fit_one_epoch import fit_one_epoch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="初始化参数")
    parser.add_argument('--root', dest='root', type=str, default=r'E:\RSI-ChangeDetection\RSI-ChangeDetection\trainData', help="数据集的路径")
    parser.add_argument('--epochs', dest='epochs', type=int, default=100, help="训练的完整轮数")
    parser.add_argument('--batch_size', dest="batch_size", type=int, default=1, help="每一个batch的大小")
    parser.add_argument('--lr', dest='lr', type=float, default=0.0001, help="学习率")
    parser.add_argument('--write_step', dest='write_step', type=int, default=1)
    parser.add_argument('--test_step', dest='test_step', type=int, default=1)
    parser.add_argument('--num_workers', dest='num_workers', type=int, default=4, help="加载数据的进程数")
    parser.add_argument('--cudnn', dest='cudnn', type=bool, default=True, help="cuDNN对多个卷积算法进行基准测试并选择最快的")
    parser.add_argument('--weight_dir', dest='weight_dir', type=str, default='weights', help="模型权重保存的位置")
    parser.add_argument('--split_train', dest='split_train', type=float, default=0.8, help="数据集拆分")
    parser.add_argument('--train_stage', dest='train_stage', type=bool, default=True, help="数据模式")
    parser.add_argument('--device', dest='device', type=str, default='cuda:0', help="数据模式")

    args = parser.parse_args()
    print(args)

    train_data, test_data = get_data(root=args.root, batch_size=args.batch_size, num_workers=args.num_workers,
                                     train_stage=args.train_stage, split_train=args.split_train)

    train_step = int((len(train_data) * args.split_train) / args.batch_size)
    test_step = int((len(test_data) * (1 - args.split_train)) / args.batch_size)
    # 加速设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 网络模型
    module = Sim_Att_UNet()
    if device.type != 'cpu':
        module.cuda()
        # if args.cudnn:
        #     torch.backends.cudnn.benchmark = True
    # 损失函数
    loss_func = HybridLoss()
    # 优化器
    optimizer = Adam(module.parameters(), lr=args.lr)
    # 训练及验证
    for epoch in range(1,args.epochs+1):
        fit_one_epoch(module=module, train_data=train_data, test_data=test_data, args=args, optimizer=optimizer,
                      loss_func=loss_func, is_cuda=(device.type != 'cpu'), epoch=epoch, train_step=train_step,
                      test_step=test_step, device=device)

        # if epoch > 0 and epoch % 10 == 0:
        #     save_model(module, epoch, args.weight_dir)
