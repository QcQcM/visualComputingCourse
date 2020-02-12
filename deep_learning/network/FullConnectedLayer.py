import numpy as np


class FullConnectedLayer(object):
    """
        全连接层：每一层包括输入尺寸、输出尺寸、权重数组、偏置项数组、激活函数
    """

    def __init__(self, input_size, output_size, activator):
        self.input = None
        self.delta = None
        self.W_grad = None
        self.b_grad = None
        self.input_size = input_size
        self.output_size = output_size
        #   权重数组
        self.W = np.random.uniform(-0.1, 0.1, (output_size, input_size))
        # self.W = np.ones((output_size, input_size))
        #   偏置项数组
        self.b = np.zeros((output_size, 1))
        #   输出向量数组
        self.output = np.zeros((output_size, 1))
        self.activator = activator

    def forward(self, input_array):
        """
        前项计算 :输入数据尺寸必须等于 input_size
        """
        self.input = input_array
        self.output = self.activator(np.dot(self.W, input_array) + self.b)

    def backward(self, delta_array):
        """
        delta_array: 从上一层传递过来的误差项
        """
        # print(np.dot(self.W.T, delta_array))
        # print(np.multiply(self.input, 1 - self.input))
        self.delta = np.multiply(self.input, 1 - self.input) * np.dot(self.W.T, delta_array)
        self.W_grad = np.dot(delta_array, self.input.T)
        self.b_grad = delta_array

    def update(self, rate):
        self.W += rate * self.W_grad
        self.b += rate * self.b_grad

