#Tensors로 2계층의 신경망이 무작위로 데이터를 맞추도록 구현

import torch

dtype = torch.float
device =torch.device("cpu")
# devce = torch.device("cuda:0") #GPU에서 실행시 주석 삭제

#N: batch_size, D_in : input_demension, H: hidden_layer, D_out: output_demension
N, D_in, H, D_out = 64, 1000, 100, 10

#random input, output data create
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

#random weight init
w1 = torch.randn( D_in, H, device=device, dtype=dtype)
w2 = torch.randn(H, D_out, device=device, dtype=dtype)

learning_rate = 1e-6
for t in range(500):
    # Feedforward(순전파 단계): predict y
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)

    # get loss and print
    loss = (y_pred - y).pow(2).sum().item()
    if t % 100 == 99:
        print(t, loss)

    #loss에 따라 w1,w1 수정, backpropagation(역전파 단계)
    grad_y_pred = 2.0*(y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.T)
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    # gradient descent(경사하강법) -> update weights
    w1 -= learning_rate * grad_w1
    w2 -= learning_rate * grad_w2