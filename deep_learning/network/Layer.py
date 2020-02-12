from deep_learning.network.Node import Node
from deep_learning.network.ConstNode import ConstNode


class Layer(object):
    def __init__(self, layer_node_count, current_node_num, is_output, activator):
        # 传入的一个是当前层节点数，一个是神经网络中现在节点编号
        self.node = []
        self.is_output = is_output
        self.activator = activator
        self.current_node_num = current_node_num
        for i in range(layer_node_count):
            self.node.append(Node(self.current_node_num, is_output, activator))
            self.current_node_num += 1
        if not is_output:
            self.node.append(ConstNode(self.current_node_num, is_output, None))
            self.current_node_num += 1
        if not is_output:
            self.layer_node_count = layer_node_count + 1
        else:
            self.layer_node_count = layer_node_count
