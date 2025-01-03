import numpy as np

# Base Node class
class Node:
    def __init__(self, inputs=None):
        if inputs is None:
            inputs = []
        self.inputs = inputs
        self.outputs = []
        self.value = None
        self.gradients = {}

        for node in inputs:
            node.outputs.append(self)

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError

# Input Node
class Input(Node):
    def __init__(self):
        Node.__init__(self)

    def forward(self, value=None):
        if value is not None:
            self.value = value

    def backward(self):
        self.gradients = {self: 0}
        for n in self.outputs:
            self.gradients[self] += n.gradients[self]

# Parameter Node
class Parameter(Node):
    def __init__(self, value):
        Node.__init__(self)
        self.value = value

    def forward(self):
        pass

    def backward(self):
        self.gradients = {self: 0}
        for n in self.outputs:
            self.gradients[self] += n.gradients[self] 

class Linear(Node):
    def __init__(self, x, A, b):
        Node.__init__(self, [x, A, b])

    def forward(self):
        x, A, b = self.inputs
        self.value = np.dot(x.value , A.value) + b.value

    def backward(self):
        x, A, b = self.inputs
        self.gradients[x] = np.dot(self.outputs[0].gradients[self] , A.value.T)
        self.gradients[A] = self.outputs[0].gradients[self] * x.value
        self.gradients[b] = self.outputs[0].gradients[self]

class Multiply(Node):
    def __init__(self, x, y):
        Node.__init__(self, [x, y])

    def forward(self):
        x, y = self.inputs
        self.value = x.value * y.value

    def backward(self):
        x, y = self.inputs
        self.gradients[x] = self.outputs[0].gradients[self] * y.value
        self.gradients[y] = self.outputs[0].gradients[self] * x.value

class Addition(Node):
    def __init__(self, x, y):
        Node.__init__(self, [x, y])

    def forward(self):
        x, y = self.inputs
        self.value = x.value + y.value

    def backward(self):
        x, y = self.inputs
        self.gradients[x] = self.outputs[0].gradients[self]
        self.gradients[y] = self.outputs[0].gradients[self]

# Sigmoid Activation Node
class Sigmoid(Node):
    def __init__(self, node):
        Node.__init__(self, [node])

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self):
        input_value = self.inputs[0].value
        self.value = self._sigmoid(input_value)

    def backward(self):
        partial = self.value * (1 - self.value)
        self.gradients[self.inputs[0]] = partial * self.outputs[0].gradients[self]

class BCE(Node):
    def __init__(self, y_true, y_pred):
        Node.__init__(self, [y_true, y_pred])

    def forward(self):
        y_true, y_pred = self.inputs
        self.value = -(1 / y_true.value.shape[0]) * np.sum(y_true.value * np.log(y_pred.value) + (1 - y_true.value) * np.log(1 - y_pred.value))

    def backward(self):
        y_true, y_pred = self.inputs
        self.gradients[y_pred] = (y_pred.value - y_true.value) / (y_pred.value * (1 - y_pred.value))
        self.gradients[y_true] = -(np.log(y_pred.value) - np.log(1 - y_pred.value))


