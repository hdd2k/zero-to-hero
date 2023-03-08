class Value:
    '''
    Scalar value with its gradient.
    '''

    def __init__(self, data, _children=(), _op=None):
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
    def __add__(self, other):
        # handle primitive int inputs
        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data + other.data,
                       _children=(self, other), _op='+')

        def _backward():
            self.grad += output.grad
            other.grad += output.grad

        output._backward = _backward
        return output

    def __radd__(self, other):
        return other + self

    def __sub__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        output = Value(self.data - other.data,
                       _children=(self, other), _op='-')

        def _backward():
            self.grad += output.grad
            other.grad -= output.grad
        output._backward = _backward

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        pass

    def __rmul__(self, other):
        pass

    def __pow__(self, other):
        pass

    def __neg__(self):
        pass

    def __truediv__(self, other):
        pass

    def __rtruediv__(self, other):
        pass

    # repr
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad}))"

    # activation functions
    def relu(self):
        pass

    def tanh(self):
        pass

    # backpropagates new gradient to ALL reachable children from "self"
    def backward(self):
        # topo sort (leaf ... chilren->curr)
        def topo(node, topoList, visited):
            # all children of curr
            for child in node._prev:
                if child not in visited:
                    topo(child, topoList, visited)

            # curr appended
            topoList.append(node)

        topoList = []
        visited = set()
        topo(self, topoList, visited)

        # in backwards order propagate new gradients (by applying _backward)
        self.grad = 1
        for node in reversed(topoList):
            node._backward()
