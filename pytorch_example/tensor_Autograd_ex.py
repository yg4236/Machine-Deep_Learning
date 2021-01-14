#Tensor and Autograd

import torch

dtype = torch.float
device = torch.device("cpu")
# devce = torch.device("cuda:0") #GPU에서 실행시 주석 삭제

#N: batch_size, D_in : input_demension, H: hidden_layer, D_out: output_demension
N, D_in, H, D_out = 64, 1000, 100, 10

#input과 output 저장을 위해 random value를 갖는 Tensor create
# requires_grad=False로 설정해 backpropagation 중 이 Tensor들에 대한 변화도를 계산할
# 필요가 없음을 나타냄 ( 기본값이 False이므로 코드에 반영 x)
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# weight 저장 위해 random value를 갖는 Tensor create
# requires_grad=True로 설정하여 backpropagation 중 이 Tensor들에 대한 변화도를
# 계산할 필요가 있음을 나타냄
w1 = torch.randn( D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    # Feedforward : Tensor 연산으로 predict y
    # Tensor을 사용한 Feedforward 단계와 완전히 동일하지만, backpropagation 단계를
    # 별도 구현하지 않아도 되므로 중간값들에 대한 reference를 갖고 있을 필요가 없다
    y_pred = x.mm(w1).clamp(min=0).mm(w2)
    # Tensor 연산으로 get loss and print
    # loss는 (1, ) 형태의 Tensor이고, loss.item()은 loss의 스칼라 값
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())
    # autograd로 backpropagation 계산. 이는 requires_grad=True를 갖는 모든 Tensor
    # 에 대해 손실 변화도 계산
    loss.backward()
    # gradient descent로 weight 갱신 ( torch.no_grad()로 감싸는 이유는 weight들이
    # requires_grad=True이지만 autograd에서 이를 추적할 필요가 없기때문
    with torch.no_grad():
        w1 -= learning_rate*w1.grad
        w2 -= learning_rate*w2.grad
        #weight updat 후 변화도를 0으로 만듬
        w1.grad.zero_()
        w2.grad.zero_()