class Connection(object):

    def __init__(self, input_node, output_node, weight, learn_rate):
        self.input_node = input_node
        self.output_node = output_node
        self.weight = weight
        self.learn_rate = learn_rate

    def update_weight(self):
        self.weight += self.learn_rate * self.output_node.error * self.output_node.output_data
