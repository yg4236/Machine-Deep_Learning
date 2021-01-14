#nupmy로 신경망 구성
import numpy as np

#N: batch_size, D_in : input_demension, H: hidden_layer, D_out: output_demension
N, D_in, H, D_out = 64, 1000, 100, 10

#random input, output data create
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

#random weight init
w1 = np.random.randn(D_in, H)
w2 = np.random.randn(H, D_out)

learning_rate = 1e-6
for t in range(500):
    #Feedforward(순전파 단계)
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)

    #get loss and print
    loss = np.square(y_pred - y).sum
    print(t, loss)

    #loss에 따라 w1,w1 수정, backpropagation(역전파 단계)
    grad_y_pred = 2.0*(y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    #update weight
    w1 -= learning_rate*grad_w1
    w2 -= learning_rate*grad_w2