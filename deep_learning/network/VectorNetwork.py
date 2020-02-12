from deep_learning.network.FullConnectedLayer import FullConnectedLayer

import struct

import numpy as np

from pip._vendor.distlib.compat import raw_input

from datetime import datetime


class Loader(object):
    def __init__(self, path, count):
        """
        初始化加载器
        path: 数据文件路径
        count: 文件中的样本个数
        """
        self.path = path
        self.count = count

    def get_file_content(self):
        f = open(self.path, 'rb')
        content = f.read()
        f.close()
        # print(type(content)) content的类型是bytes没错
        return content

    def to_int(self, byte):
        return struct.unpack('B', byte)[0]


class ImageLoader(Loader):
    def get_picture(self, content, index):
        start = index * 28 * 28 + 16
        picture = []
        for i in range(28):
            picture.append([])
            for j in range(28):
                picture[i].append(
                    content[start + i * 28 + j])
        return picture

    def get_one_sample(self, picture):
        sample = []
        for i in range(28):
            for j in range(28):
                sample.append(picture[i][j])
        return sample

    def load(self):
        content = self.get_file_content()
        data_set = []
        for index in range(self.count):  # 这里不用self.count而是用两个样本试一下
            data_set.append(
                self.get_one_sample(
                    self.get_picture(content, index)))
        return data_set


# 标签数据加载器
class LabelLoader(Loader):
    def load(self):
        content = self.get_file_content()
        labels = []
        for index in range(self.count):
            labels.append(self.norm(content[index + 8]))
        return labels

    def norm(self, label):
        label_vec = []
        label_value = label
        for i in range(10):
            if i == label_value:
                label_vec.append(1)
            else:
                label_vec.append(0)
        return label_vec


def get_result(vec):
    max_value_index = 0
    max_value = 0
    x = vec.shape[0]    # x应该是10
    y = vec.shape[1]    # y应该是1
    for i in range(x):
        if vec[i][0] > max_value:
            max_value = vec[i][0]
            max_value_index = i
    return max_value_index


def get_training_data_set():
    image_loader = ImageLoader('train-images.idx3-ubyte', 60000)
    label_loader = LabelLoader('train-labels.idx1-ubyte', 60000)
    return image_loader.load(), label_loader.load()


def get_test_data_set():
    image_loader = ImageLoader('t10k-images.idx3-ubyte', 10000)
    label_loader = LabelLoader('t10k-labels.idx1-ubyte', 10000)
    return image_loader.load(), label_loader.load()


def evaluate(network, test_data_set, test_labels):
    error = 0
    total = len(test_data_set)
    for i in range(total):
        label = get_result(np.array(test_labels[i]).reshape(10, 1))
        predict = get_result(network.predict(test_data_set[i]))
        if label != predict:
            error += 1
    return float(error) / float(total)


def train_and_evaluate():
    last_error_ratio = 1.0
    epoch = 0
    train_data_set, train_labels = get_training_data_set()
    test_data, test_labels = get_test_data_set()
    net_size = list(raw_input('请输入网络尺寸'))
    iteration = int(raw_input('请输入训练轮数'))
    rate = float(raw_input('请输入初始学习率'))
    decay_rate = float(raw_input('请输入学习率衰减率'))
    # iteration = 100
    # decay_rate = 0.999
    # rate = 0.5
    # net_size = [784, 300, 10]
    network = VectorNetwork(net_size)
    for i in range(iteration):
        learn_rate = rate * decay_rate**(i/iteration)
        network.train(train_data_set, train_labels, learn_rate)
        error_ratio = evaluate(network, test_data, test_labels)
        epoch += 1
        print('%s after epoch %d, error ratio is %f' % (datetime.now(), epoch, error_ratio))


def sigmoid(inx):
    x = inx.shape[0]
    y = inx.shape[1]
    result = []
    for i in range(x):
        for j in range(y):
            if inx[i][j] >= 0:
                result.append(1.0 / (1.0 + np.exp(-inx[i][j])))
            else:
                result.append(1.0 - 1.0 / (1.0 + np.exp(inx[i][j])))
    return np.array(result).reshape(x, y)


class VectorNetwork(object):
    def __init__(self, layers):  # 输入的是代表每一层节点数的列表 如[784, 300, 10]
        self.layer_count = len(layers)
        self.layers = []
        for i in range(self.layer_count - 1):
            self.layers.append(FullConnectedLayer(layers[i], layers[i + 1], sigmoid))

    def train(self, train_data, train_label, rate):
        """
        训练函数
        :训练数据 train_data:
        :训练数据标签 train_label:
        :学习率 rate:
        :迭代次数 epoch:
        """
        for d in range(len(train_data)):
            data = np.array(train_data[d]).reshape(784, 1)/255
            label = np.array(train_label[d]).reshape(10, 1)
            self.train_one_step(data, label, rate)

    def train_one_step(self, train_data, train_label, rate):
        self.predict(train_data, True)
        self.calc_gradient(train_label)
        self.update_weight(rate)

    def predict(self, train_data, train=False):
        if not train:
            output = np.array(train_data).reshape(784, 1)/255
        else:
            output = np.array(train_data).reshape(784, 1)
        for i in range(self.layer_count-1):
            self.layers[i].forward(output)
            output = self.layers[i].output
        return output

    def calc_gradient(self, train_label):
        delta = np.multiply(np.multiply(self.layers[-1].output, 1 - self.layers[-1].output), train_label - self.layers[-1].output)
        for layer in self.layers[::-1]:
            layer.backward(delta)
            delta = layer.delta
        return delta

    def update_weight(self, rate):
        for layer in self.layers:
            layer.update(rate)


if __name__ == '__main__':
    train_and_evaluate()
