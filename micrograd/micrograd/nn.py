import random
from micrograd.value import Value


class Module:
    def zero_grad(self):
        pass

    def parameters(self):
        pass


class Neuron(Module):
    def __init__(self, input_neurons, nonlin=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(input_neurons)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        acc = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)  # w * x + b
        return acc.relu() if self.nonlin else acc

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"


class Layer(Module):
    def __init__(self, input_neurons, output_neurons, **kwargs):
        self.neurons = [Neuron(input_neurons, **kwargs)
                        for _ in range(output_neurons)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        # get all parameters from all neurons in this layer
        params = []
        for n in self.neurons:
            params.extend(n.parameters())
        return params

    def __repr__(self):
        return f"Layer of [{', '.join([str(n) for n in self.neurons])}]"

# MLP = Multi-Layer Perceptron


class MLP(Module):
    def __init__(self, input_neurons, output_neurons):
        sizes = [input_neurons] + output_neurons
        self.layers = [Layer(sizes[i], sizes[i+1], nonlin=(i != len(output_neurons)-2))
                       for i in range(len(sizes)-1)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for l in self.layers:
            params.extend(l.parameters())
        return params

    def __repr__(self):
        return f"MLP of [{', '.join([str(l) for l in self.layers])}]"
