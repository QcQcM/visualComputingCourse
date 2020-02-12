class Node(object):
    def __init__(self, current_node_num, is_output, activator):
        self.input_conn = []  # 输入连接
        self.output_conn = []  # 输出连接
        self.number = current_node_num  # 这个节点的标号
        self.error = 0  # 误差值
        self.is_output = is_output  # 是否是输出层节点
        self.input_data = 0
        self.output_data = 0
        self.activator = activator
        self.label_output = 0

    def count_input(self):
        for i in range(len(self.input_conn)):
            self.input_data += self.input_conn[i].input_node.output_data * self.input_conn[i].weight

    def count_output(self):
        self.output_data = self.activator(self.input_data)

    def add_input_conn(self, input_conn):
        self.input_conn.append(input_conn)

    def add_output_conn(self, output_conn):
        self.output_conn.append(output_conn)

    def count_error(self):
        if self.is_output:  # 输出层误差计算方式
            self.error = (self.label_output - self.output_data) * self.output_data * (1 - self.output_data)
        else:
            update_delta = 0
            for i in range(len(self.output_conn)):
                update_delta += self.output_conn[i].weight * self.output_conn[i].output_node.error
            self.error = self.output_data * (1 - self.output_data) * update_delta