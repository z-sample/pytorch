import torch
from torch import nn, optim
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt

# 月份与商品价格
x_train = np.array([[1], [2], [3], [4], [5], [6],
                    [7], [8], [9], [10], [11], [12]], dtype=np.float32)

y_train = np.array([[10.7], [20.76], [30.09], [35.19], [42.694], [38.573],
                    [49.366], [60.596], [75.53], [69.221], [94.827],
                    [90.465]], dtype=np.float32)  # 如果最后一个数字为很小的值，那么对整条线影响非常大

x_train = torch.from_numpy(x_train)

y_train = torch.from_numpy(y_train)


# Linear Regression Model (f(y) = kx + b)
class LinearRegression(nn.Module):
    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(1, 1)  # input and output is 1 dimension

    def forward(self, x):
        out = self.linear(x)
        return out


model = LinearRegression()
# 定义loss和优化函数
criterion = nn.MSELoss()  # MSE就是Mean Squared Error均方误差
optimizer = optim.SGD(model.parameters(), lr=1e-4)  # SGD就是随机梯度下降， 1e-4=0.0001

# 开始训练
num_epochs = 500
for epoch in range(num_epochs):
    inputs = Variable(x_train)
    target = Variable(y_train)

    # forward 对输入进行预测
    out = model(inputs)  # 向前传播 __call__() 会调用model.forward(), 也就是LinearRegression.forward()
    print(out)  # out 就是蒙的target
    # 计算损失/误差
    loss = criterion(out, target)  # 看看蒙的对不对，跟实际差多少
    # backward
    optimizer.zero_grad()  # 梯度归零
    loss.backward()  # 向后传播 通过误差值反向算出价格预测值的偏导数
    optimizer.step()  # 更新参数

    if (epoch + 1) % 10 == 0:
        print('Epoch: {}, loss: {:.6f}'.format(epoch + 1, loss.data[0]))

# 开始测试模型
model.eval()  # 特别注意的是需要用 model.eval()，让model变成测试模式，这主要是对dropout和batch normalization的操作在训练和测试的时候是不一样的

plt.plot(x_train.numpy(), y_train.numpy(), 'ro', label='Original data')  # 绘制点（真实值）

predict = model(Variable(x_train))  # 再次对x训练集进行预测
predict = predict.data.numpy()
plt.plot(x_train.numpy(), predict, label='Fitting Line')  # 绘制线（预测值）

# 使用
print("预测第13个月份的价格", model.forward(Variable(torch.FloatTensor([13]))))  # 使用训练好的模型来蒙

# 显示图例
plt.legend()
plt.show()

# 保存模型，以后可以load之后再forward
# torch.save(model.state_dict(), './linear.pt')

# 参考
# http://www.pytorchtutorial.com/10-minute-pytorch-1/  - 10分钟快速入门 PyTorch (1) - 线性回归
# http://blog.csdn.net/u013066730/article/details/51648876  - 深度学习笔记（二）用Torch实现线性回归
