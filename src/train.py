
import torch
import argparse
from dataset.RSIdataset import get_data
from module import NestedUNet
from losses import BalancedBinaryCrossEntropy, BinaryDiceLoss
from torch.optim import Adam
from train_utils.fit_one_epoch import fit_one_epoch
from utils.evalute import val
from utils.save_weights import save_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="初始化参数")
    parser.add_argument('--root', dest='root', type=str, default=r'F:\mahcenglin\trainData', help="数据集的路径")
    parser.add_argument('--epochs', dest='epochs', type=int, default=100, help="训练的完整轮数")
    parser.add_argument('--batch_size', dest="batch_size", type=int, default=1, help="每一个batch的大小")
    parser.add_argument('--lr', dest='lr', type=float, default=0.003, help="学习率")
    parser.add_argument('--num_workers', dest='num_workers', type=int, default=4, help="加载数据的进程数")
    parser.add_argument('--cudnn', dest='cudnn', type=bool, default=True, help="cuDNN对多个卷积算法进行基准测试并选择最快的")
    parser.add_argument('--weight_dir', dest='--weight_dir', type=str, default='weights', help="模型权重保存的位置")
    args = parser.parse_args()
    print(args)
    val = val()
    train_data, test_data = get_data(root=args.root, batch_size=args.batch_size, total_num=2000, test_num=400)
    # 加速设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 网络模型
    module = NestedUNet()
    if torch.cuda.is_available():
        module.cuda()
        if args.cudnn:
            torch.backends.cudnn.benchmark = True
    # 损失函数
    loss_1 = BalancedBinaryCrossEntropy()
    loss_2 = BinaryDiceLoss()
    # 优化器
    optimizer = Adam(module.parameters(), lr=args.lr)
    # 训练及验证
    for epoch in range(args.epochs):
        fit_one_epoch(module=module, train_data=train_data, test_data=test_data, optimizer=optimizer,
                      loss_dice=loss_1, is_cuda=True, val=val, epoch=epoch)
        if epoch > 0 and epoch % 10 == 0:
            save_model(module, epoch, args.weight_dir)
