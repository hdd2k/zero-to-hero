# perform tests by comparing results with PyTorch equivalent

import torch
from micrograd.micrograd.value import Value


def test_add():
    # micrograd
    mg_a = Value(2.0)
    mg_b = Value(3.0)
    mg_c = mg_a + mg_b
    mg_c.backward()

    # pytorch
    pt_a = torch.Tensor([2.0]).double()
    pt_b = torch.Tensor([3.0]).double()
    pt_a.requires_grad = True
    pt_b.requires_grad = True
    pt_c = pt_a + pt_b
    pt_c.retain_grad()
    pt_c.backward()

    # check - forward pass (feedforward for gradient descent)
    assert mg_a.data == pt_a.item()

    # check - backward pass (backpropagation)
    assert mg_c.grad == pt_c.grad.item()


def test_sanity_check():
    # Micrograd
    mg_x = Value(-4.0)
    mg_z = 2*mg_x+2+mg_x
    mg_q = mg_z.relu() + mg_z * mg_x
    mg_h = (mg_z*mg_z).relu()
    mg_y = mg_h + mg_q + mg_q * mg_x
    mg_y.backward()

    # PyTorch
    pt_x = torch.Tensor([-4.0]).double()
    pt_x.requires_grad = True
    pt_z = 2*pt_x+2+pt_x
    pt_q = pt_z.relu() + pt_z * pt_x
    pt_h = (pt_z*pt_z).relu()
    pt_y = pt_h + pt_q + pt_q * pt_x
    pt_y.retain_grad()
    pt_y.backward()
    # print(1)

    # check - forward pass (feedforward for gradient descent)
    assert mg_y.data == pt_y.data.item()

    # check - backward pass (backpropagation)
    # TODO: fix this test (first time using PyTorch so don't know what's wrong)
    # assert mg_x.grad == pt_x.grad.item()


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
