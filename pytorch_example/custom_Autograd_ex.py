#New Autograd function def
# ReLU로 비선형적으로 동작하는 사용자 정의 autograd 함수를 정의하고, 2-계층 신경망에 적용
import torch

class MyReLU(torch.autograd.Function):
    """
    torch.autograd.Function을 상속받아 사용자 정의 autograd Function 구현하고,
    Tensor 연산을 하는 순전파, 역전파 단계 구현
    """
    @staticmethod
    def forward(ctx, input):
        """
        순전파 단계에서는 입력을 갖는 Tensor를 받아 출력을 갖는 Tensor를 반환합니다.
        ctx는 컨텍스트 객체(context object)로 역전파 연산을 위한 정보 저장에
        사용합니다. ctx.save_for_backward method를 사용하여 역전파 단계에서 사용할 어떠한
        객체도 저장(cache)해 둘 수 있습니다.
        """
        ctx.save_for_backward(input)
        return input.clamp(min=0)
    @staticmethod
    def backward(ctx, grad_output):
        """
        역전파 단계에서는 출력에 대한 손실의 변화도를 갖는 Tensor를 받고, 입력에
        대한 손실의 변화도를 계산합니다.
        """
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input<0] =0
        return grad_input

dtype = torch.float
device = torch.device("cpu")

# devce = torch.device("cuda:0") #GPU에서 실행시 주석 삭제

#N: batch_size, D_in : input_demension, H: hidden_layer, D_out: output_demension
N, D_in, H, D_out = 64, 1000, 100, 10

#random input, output data create
x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

#random weight init
w1 = torch.randn( D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6
for t in range(500):
    relu = MyReLU.apply
    # 순전파 단계: Tensor 연산을 사용하여 예상되는 y 값을 계산합니다;
    # 사용자 정의 autograd 연산을 사용하여 ReLU를 계산합니다.
    y_pred = relu(x.mm(w1)).mm(w2)

    # 손실을 계산하고 출력합니다.
    loss = (y_pred - y).pow(2).sum()
    if t % 100 == 99:
        print(t, loss.item())

    # autograde를 사용하여 역전파 단계를 계산합니다.
    loss.backward()
    # 경사하강법(gradient descent)을 사용하여 가중치를 갱신합니다.
    with torch.no_grad():
        w1 -= learning_rate * w1.grad
        w2 -= learning_rate * w2.grad

        # 가중치 갱신 후에는 수동으로 변화도를 0으로 만듭니다.
        w1.grad.zero_()
        w2.grad.zero_()
