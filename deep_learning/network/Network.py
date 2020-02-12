import math

from deep_learning.network.Connection import Connection
from deep_learning.network.Layer import Layer
from deep_learning.network.ConstConnection import ConstConnection
import random


class Network(object):
    def __init__(self, layers, learn_rate, iteration, activator):
        self.learn_rate = learn_rate  # 超参数学习率
        self.layer_count = len(layers)
        self.layers = []  # 存储各层
        self.current_node_num = 1  # 当前节点编号，用于给新建的节点编号，整个网络一起编号
        self.connections = []  # 网络里所有的连接
        self.iteration = iteration
        self.activator = activator
        for i in range(self.layer_count):  # 新建层
            if i == self.layer_count - 1:
                self.layers.append(Layer(layers[i], self.current_node_num, True, self.activator))
                self.current_node_num += layers[i]
            else:
                self.layers.append(Layer(layers[i], self.current_node_num, False, self.activator))
                self.current_node_num += layers[i] + 1
        for i in range(self.layer_count - 1):
            # 新建连接,每一次在两层之间建立连接并且把连接加入到输入连接和输出连接
            for j in range(self.layers[i].layer_node_count):
                #   如果不是输出层，说明i+1层有一项偏置项，i层的节点不需要与该偏置项建立连接
                if not self.layers[i + 1].is_output:
                    for k in range(self.layers[i + 1].layer_node_count - 1):
                        #   i层的最后一个是i+1层偏置项，需要与下一层除偏置项之外的建立特殊连接
                        if j == self.layers[i].layer_node_count - 1:
                            conn = ConstConnection(self.layers[i].node[j], self.layers[i + 1].node[k],
                                                   random.randint(0, 1) * 0.01, self.learn_rate)
                            self.connections.append(conn)
                            self.layers[i].node[j].add_output_conn(conn)
                            self.layers[i + 1].node[k].add_input_conn(conn)
                        else:
                            #   i层除最后一个外与下一层除最后一个以外建立普通连接
                            conn = Connection(self.layers[i].node[j], self.layers[i + 1].node[k],
                                              random.randint(0, 1) * 0.1, self.learn_rate)
                            self.connections.append(conn)
                            self.layers[i].node[j].add_output_conn(conn)
                            self.layers[i + 1].node[k].add_input_conn(conn)
                else:
                    #   如果是下一层是输出层，说明下一层没有偏置项节点
                    for k in range(self.layers[i + 1].layer_node_count):
                        #   i层的最后一个是i+1层偏置项，需要与下一层所有节点建立特殊连接
                        if j == self.layers[i].layer_node_count - 1:
                            conn = ConstConnection(self.layers[i].node[j], self.layers[i + 1].node[k],
                                                   random.randint(0, 1) * 0.01, self.learn_rate)
                            self.connections.append(conn)
                            self.layers[i].node[j].add_output_conn(conn)
                            self.layers[i + 1].node[k].add_input_conn(conn)
                        else:
                            #   i层除最后一个外与下一层所有节点建立普通连接
                            conn = Connection(self.layers[i].node[j], self.layers[i + 1].node[k],
                                              random.randint(0, 1) * 0.01, self.learn_rate)
                            self.connections.append(conn)
                            self.layers[i].node[j].add_output_conn(conn)
                            self.layers[i + 1].node[k].add_input_conn(conn)

    def train(self, train_labels, train_data_set):
        for i in range(self.iteration):
            self.train_one_step(train_labels, train_data_set)

    def train_one_step(self, train_labels, train_data_set):
        for i in range(len(train_data_set)):  # 对每一个样本数据
            self.predict(train_data_set[i], True)
            k = self.layer_count - 1
            while k >= 0:
                for j in range(self.layers[k].layer_node_count):
                    self.layers[k].node[j].count_error()
                    for l in range(len(self.layers[k].node[j].input_conn)):
                        self.layers[k].node[j].input_conn[l].update_weight()
                k -= 1

    def predict(self, sample_data, is_train=False):  # 一个计算输出结果的函数
        for j in range(self.layers[0].layer_node_count - 1):  # 对第一层的所有节点
            self.layers[0].node[j].input_data = sample_data[j]  # 把该样本的各特征值分别赋值给第一层的节点
        if is_train:
            for j in range(self.layers[self.layer_count - 1].layer_node_count):  # 对最后一层的所有节点
                self.layers[self.layer_count - 1].node[j].label_output = sample_data[j]  # 把该样本的标签值作为输出层理想输出
        for k in range(self.layer_count):  # 从输入层开始到输出层，对每一层中的每一个节点，计算输入输出
            for j in range(self.layers[k].layer_node_count):
                self.layers[k].node[j].count_input()
                self.layers[k].node[j].count_output()
        result = []
        for i in range(self.layers[self.layer_count - 1].layer_node_count):
            result.append(self.layers[self.layer_count - 1].node[i].output_data)
        return result


def sigmoid(x):
    try:
        return 1 / (1 + math.exp(-x))
    except OverflowError:
        return 1 / (1 + float('inf'))

