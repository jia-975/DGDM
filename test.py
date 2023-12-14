import torch

def fun(a, b):
    c = a * b + b * 23
    linear = torch.nn.Linear(10, 10)
    b = linear(b)
    c += b
    return c, linear.weight
a = torch.randn(1, 10)
print(a)
# a.requires_grad_()
b = torch.randn(1, 10)
print(b)
b.requires_grad_()
c, w  = fun(a, b)
print(w)
c.requires_grad_()
grad= torch.autograd.grad(outputs=c.sum(), inputs=b, retain_graph=True)[0]
print(grad)
print(grad == (a + 23 + w))


