import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from mnist_models import Net
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from torch.autograd import Variable
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler  # 标准差标准化
from sklearn.decomposition import PCA
import time

# 设置模型超参数
EPOCHS = 50                # 训练轮次
SAVE_PATH = './models'      # 指中间以及最终模型保存的路径


def draw_result(eval_losses, eval_acces):
    # 横轴点取为 1-50，代表50次迭代
    x = range(1, EPOCHS + 1)
    fig, left_axis = plt.subplots()
    # 画出测试集Loss记录随迭代次数的变化曲线
    p1, = left_axis.plot(x, eval_losses, 'ro-')
    right_axis = left_axis.twinx()
    # 画出测试集Acc记录随迭代次数的变化曲线
    p2, = right_axis.plot(x, eval_acces, 'bo-')
    plt.xticks(x, rotation=0)

    # 设置左坐标轴以及右坐标轴的范围、精度
    left_axis.set_ylim(0, 0.5)
    left_axis.set_yticks(np.arange(0, 0.5, 0.1))
    right_axis.set_ylim(0.9, 1.01)
    right_axis.set_yticks(np.arange(0.9, 1.01, 0.02))
    # 设置坐标及标题的大小、颜色
    left_axis.set_xlabel('Labels')
    left_axis.set_ylabel('Loss', color='r')
    left_axis.tick_params(axis='y', colors='r')
    right_axis.set_ylabel('Accuracy', color='b')
    right_axis.tick_params(axis='y', colors='b')
    plt.show()


if __name__ == "__main__":
    VSM = pd.read_csv('./VSM.csv', index_col=0)
    print(VSM)
    print(VSM.shape)
    x = VSM.iloc[:, :4613].values
    y = VSM[['4614']].values
    x = np.array(x)
    y = np.array(y)
    x_train, x_test, y_train, y_test = \
        train_test_split(x, y, random_state=1, train_size=0.7, test_size=0.3)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

    # 2、数据标准化
    stdScaler = StandardScaler()
    x_train = stdScaler.fit_transform(x_train)
    x_test = stdScaler.fit_transform(x_test)
    print('数据标准化完成!')

    # 3、PCA降维
    pca = PCA()
    print("[PCA1] reducing dimensions...")
    pca.fit_transform(x_train)

    pca_K = PCA()
    print("[PCA2] reducing dimensions...")
    pca_K.fit_transform(x_test)

    # 转换为tensor
    # x_train = torch.from_numpy(x_train)
    # y_train = torch.from_numpy(y_train)
    # x_test = torch.from_numpy(x_test)
    # y_test = torch.from_numpy(y_test)
    # x_train = torch.tensor(x_train, dtype=torch.float32)
    # x_test = torch.tensor(x_test, dtype=torch.float32)
    x_train = torch.as_tensor(torch.from_numpy(x_train), dtype=torch.float32)
    x_test = torch.as_tensor(torch.from_numpy(x_test), dtype=torch.float32)
    x_train = x_train.reshape(x_train.shape[0], 1, 1, x_train.shape[1])
    x_test = x_test.reshape(x_test.shape[0], 1, x_test.shape[1])
    print(x_train.shape, x_test.shape)

    y_train = y_train.reshape(y_train.shape[0], 1)
    y_test = y_test.reshape(y_test.shape[0], 1)

    # y_train = torch.tensor(y_train, dtype=torch.long)
    # y_test = torch.tensor(y_test, dtype=torch.long)
    y_train = torch.as_tensor(torch.from_numpy(y_train), dtype=torch.long)
    y_test = torch.as_tensor(torch.from_numpy(y_test), dtype=torch.long)

    net = Net()

    # 训练集上误差和准确率变化记录
    losses = []
    acces = []
    # 测试集上误差和准确率变化记录
    eval_losses = []
    eval_acces = []
    # 定义损失函数 与 优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), 1e-2)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.0002, momentum=0.2)
    start = time.time()
    for epoch in tqdm(range(EPOCHS)):
        train_loss = 0
        train_acc = 0
        net.train()
        # running_loss = 0
        for i, input_data in enumerate(x_train, 0):
            # 取得图片数据和标签
            label = y_train[i]
            # 前向传播
            outputs = net(input_data)
            loss = criterion(outputs, label)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 记录误差
            train_loss += loss.item()
            # 计算分类的准确率
            _, pred = outputs.max(1)
            num_correct = (np.array(pred, dtype=np.int_) == np.array(label, dtype=np.int_)).sum()
            acc = num_correct / input_data.shape[0]
            train_acc += acc
        losses.append(train_loss / len(x_train))  # 累积误差值除以样本总数
        acces.append(train_acc / len(x_train))  # 累积准确率除以样本总数

        # 在测试集上检验效果
        eval_loss = 0
        eval_acc = 0
        net.eval()  # 将模型改为预测模式
        for i, input_data in enumerate(x_test, 0):
            label = y_test[i]
            # 前向传播
            outputs = net(input_data)
            loss = criterion(outputs, label)
            # 记录误差
            eval_loss += loss.item()
            # 记录准确率
            _, pred = outputs.max(1)
            num_correct = (np.array(pred, dtype=np.int_) == np.array(label, dtype=np.int_)).sum()
            acc = num_correct / input_data.shape[0]
            eval_acc += acc

        eval_losses.append(eval_loss / len(x_test))
        eval_acces.append(eval_acc / len(x_test))
    print(losses)
    print(acces)
    print(eval_losses)
    print(eval_acces)
    print('time = %2dm:%2ds' % ((time.time() - start) // 60, (time.time() - start) % 60))

    # 绘制出训练过程中测试集的Loss和ACC
    draw_result(eval_losses, eval_acces)
