import matplotlib.pyplot as plt
import numpy as np


# 传入一个数据集默认只显示一个
def show(dataset, num=1):
    if num > len(dataset) or num < 0:
        print("******想要显示的数据数量过大或过小******")
    else:
        for x1, x2, x1_label, x2_label, change in dataset[0:num]:
            x1 = np.squeeze(x1)
            x2 = np.squeeze(x2)
            x1_label = np.squeeze(x1_label)
            x2_label = np.squeeze(x2_label)
            change = np.squeeze(change)
            # 可视化
            plt.subplot(1, 5, 1)
            plt.imshow(x1)
            plt.axis("off")
            plt.title("x1")
            plt.subplot(1, 5, 2)
            plt.imshow(x2)
            plt.axis("off")
            plt.title("x2")
            plt.subplot(1, 5, 3)
            plt.imshow(x1_label)
            plt.axis("off")
            plt.title("x1_label")
            plt.subplot(1, 5, 4)
            plt.imshow(x2_label)
            plt.axis("off")
            plt.title("x2_label")
            plt.subplot(1, 5, 5)
            plt.imshow(change)
            plt.axis("off")
            plt.title("change")
            plt.show()