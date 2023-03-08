# perform tests by comparing results with PyTorch equivalent

import torch
from micrograd.micrograd.value import Value


def test_add():
    # micrograd
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    c.backward()
    mg_a, mg_b, mg_c = a, b, c

    # pytorch
    a = torch.Tensor([2.0]).double()
    b = torch.Tensor([3.0]).double()
    a.requires_grad = True
    b.requires_grad = True
    c = a + b
    c.retain_grad()
    c.backward()
    pt_a, pt_b, pt_c = a, b, c

    # check - forward pass (feedforward for gradient descent)
    assert mg_a.data == pt_a.item()

    # check - backward pass (backpropagation)
    assert mg_c.grad == pt_c.grad.item()


def test_sanity_check():
    # Micrograd
    x = Value(-4.0)
    z = 2*x+2+x
    q = z.relu() + z * x
    h = (z*z).relu()
    y = h + q + q * x
    y.backward()
    mg_x, mg_y = x, y

    # PyTorch
    x = torch.Tensor([-4.0]).double()
    x.requires_grad = True
    z = 2*x+2+x
    q = z.relu() + z * x
    h = (z*z).relu()
    y = h + q + q * x
    y.backward()
    pt_x, pt_y = x, y

    # check - forward pass (feedforward for gradient descent)
    assert mg_y.data == pt_y.data.item()

    # check - backward pass (backpropagation)
    assert mg_x.grad == pt_x.grad.item()


def test_more_ops():
    # micrograd
    a = Value(-4.0)
    b = Value(2.0)
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c * (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c-d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    mg_a, mg_b, mg_g = a, b, g

    # pytorch
    a = torch.Tensor([-4.0]).double()
    b = torch.Tensor([2.0]).double()
    c = a + b
    d = a * b + b**3
    c += c + 1
    c += 1 + c * (-a)
    d += d * 2 + (b + a).relu()
    d += 3 * d + (b - a).relu()
    e = c-d
    f = e**2
    g = f / 2.0
    g += 10.0 / f
    g.backward()
    pt_a, pt_b, pt_g = a, b, g

    # check - forward pass (feedforward for gradient descent)
    tol = 1e-6
    assert abs(mg_g.data - pt_g.data.item()) < tol

    # check - backward pass (backpropagation)
    assert abs(mg_a.data - pt_a.data.item()) < tol
    assert abs(mg_b.data - pt_b.data.item()) < tol


test_add()
test_sanity_check()
# test_more_ops()
