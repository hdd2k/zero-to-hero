class Value:
    '''
    Scalar value with its gradient.
    '''

    def __init__(self, data, _children=(), _op=''):
        self.data = data
        self.grad = 0
        # internal vars for graph construction
        # _prev: list of nodes that are inputs to this node
        self._prev = set(_children)
        # _backward : function that backpropagates gradient to inputs
        self._backward = lambda: None
        # _op: operator that created this node
        self._op = _op

    # operators for Value
    def __add__(self, other):  # self + other
        # handle primitive int inputs
        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += output.grad
            other.grad += output.grad

        output._backward = _backward
        return output

    def __radd__(self, other):  # other + self
        return self + other

    def __sub__(self, other):  # self - other
        return self + (-other)

    def __rsub__(self, other):  # other - self
        return -self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += (other.data * output.grad)
            other.grad += (self.data * output.grad)
        output._backward = _backward
        return output

    def __rmul__(self, other):  # other * self
        return (self * other)

    def __pow__(self, other):  # self ** other
        assert isinstance(other.data, (int, float)
                          ), "exponent must be a number"
        output = Value(self.data ** other, (self,), _op=f'**{other}')

        def _backward():
            self.grad += (other * (self.data ** (other - 1))) * output.grad
        output._backward = _backward
        return output

    def __neg__(self):  # -self
        return (self * -1)

    def __truediv__(self, other):  # self / other
        return (self * (other ** -1))

    def __rtruediv__(self, other):  # other / self
        return (other * (self ** -1))

    # repr
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad}))"

    # activation functions
    def relu(self):
        output = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            self.grad += (output.data > 0) * output.grad
        output._backward = _backward
        return output

    def tanh(self):
        pass

    # backpropagates new gradient to ALL reachable children from "self"
    def backward(self):
        # topo sort (leaf ... chilren->curr)
        topoList = []
        visited = set()

        def topo(node, topoList, visited):
            # all children of curr
            if node not in visited:
                visited.add(node)
                for child in node._prev:
                    topo(child, topoList, visited)
                # curr appended
                topoList.append(node)

        topo(self, topoList, visited)

        # in backwards order propagate new gradients (by applying _backward)
        self.grad = 1
        for node in reversed(topoList):
            node._backward()
