from deep_learning.network.Connection import Connection


class ConstConnection(Connection):
    def update_weight(self):
        self.weight += self.learn_rate * self.output_node.error
