from deep_learning.network.Node import Node


class ConstNode(Node):
    def count_output(self):
        self.output_data = 1

    def count_error(self):
        self.error = 0  # 偏置项节点不需要计算损失