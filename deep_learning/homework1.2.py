import numpy as np
import math


def f(x, y):
    return x * math.exp(-(x ** 2 + y ** 2))


def fx(x, y):  # 对x求偏导
    return (1 - 2 * x ** 2) * math.exp(-(x ** 2 + y ** 2))


def fy(x, y):  # 对y求偏导
    return (-2 * x * y) * math.exp(-(x ** 2 + y ** 2))


#   梯度算法
x = 0
y = 1
x_save = []
y_save = []
f_save = []
learning_rate = 0.1
f_value = f(x, y)
f_current = 999
iter_num = 0
while abs(f_current - f_value) > 1e-10 and iter_num < 100:
    f_value = f(x, y)
    iter_num += 1
    x -= learning_rate * fx(x, y)
    y -= learning_rate * fy(x, y)
    x_save.append(x)
    y_save.append(y)
    f_current = f(x, y)
    f_save.append(f_current)
print("最终经过%d轮迭代，x=%f, y=%f, 最小值为%f" % (iter_num+1, x, y, f_current))
print("x的记录为：", end="")
print(x_save)
print("y的记录为：", end="")
print(y_save)
print("函数的值记录为：", end="")
print(f_save)