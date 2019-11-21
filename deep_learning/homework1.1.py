import math
import numpy as np
import matplotlib.pyplot as plt

#   绘制了该函数关于x轴对称的函数从 -1到8 的曲线，发现有x=1这个极小值
plot_y = []
x = np.linspace(-1, 8)  # 在-1和8范围内产生50个点
plot_x = x.tolist()  # 转换成list方便对每一个x算指数，因为exp参数只能是一维
for px in plot_x:
    plot_y.append(-px * math.exp(-px))
y = np.array(plot_y)


#   原函数关于x轴对称的函数是一个凸函数，适用梯度下降
def pilot(a):   # 求导数值的函数
    return -(1 - a) * math.exp(-a)


def value(a):   # 求函数值的函数
    return -a * math.exp(-a)


#   梯度下降算法
initial_x = 0
learning_rate = 0.1
epsilon = 1e-8  # 一个用于迭代结束的精确度差值
history_x = []  # 记录遍历过的x值
last_x = -1
while abs(value(last_x)-value(initial_x) > epsilon):
    gradient = pilot(initial_x)
    last_x = initial_x
    initial_x -= learning_rate * gradient
    history_x.append(last_x)
plt.plot(x, y)
for i in range(len(history_x)):
    plt.plot(history_x[i], value(history_x[i]), color='r', marker='*')
print("求得的让原函数取得极大值的数值为：%f" % history_x[-1])
print("最优解为：%f" % (-value(history_x[-1])))
plt.show()
