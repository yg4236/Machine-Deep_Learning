## nn module
import torch
#N: batch_size, D_in : input_demension, H: hidden_layer, D_out: output_demension
N, D_in, H, D_out = 64, 1000, 100, 10

#random input, output data create
x = torch.randn(N, D_in)
y = torch.randn(N, D_out)
# nn 패키지를 사용하여 모델을 순차적 계층(sequence of layers)으로 정의합니다.
# nn.Sequential은 다른 Module들을 포함하는 Module로, 그 Module들을 순차적으로
# 적용하여 출력을 생성합니다. 각각의 Linear Module은 선형 함수를 사용하여
# 입력으로부터 출력을 계산하고, 내부 Tensor에 가중치와 편향을 저장합니다.
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)

## nn 패키지에 포함되어있는 손실 함수중 MSE 을 사용
loss_fn = torch.nn.MSELoss(reduction='sum')

learning_rate = 1e-4
for t in range(500):
    # Feedfoward 단계 : model에 x를 전달하여 y_pred를 계산
    y_pred = model(x)

    # 손실을 계산하고 출력합니다. 예측한 y와 정답인 y를 갖는 Tensor들을 전달하고,
    # 손실 함수는 손실 값을 갖는 Tensor를 반환합니다.
    loss = loss_fn(y_pred, y)
    if t % 100 == 99:
        print(t, loss.item())

    # 역전파 단계를 실행하기 전에 변화도를 0으로 만듭니다.
    model.zero_grad()

    # 역전파 단계: 모델의 학습 가능한 모든 매개변수에 대해 손실의 변화도를
    # 계산합니다. 내부적으로 각 Module의 매개변수는 requires_grad=True 일 때
    # Tensor 내에 저장되므로, 이 호출은 모든 모델의 모든 학습 가능한 매개변수의
    # 변화도를 계산하게 됩니다.
    loss.backward()

    # 경사하강법(gradient descent)를 사용하여 가중치를 갱신합니다. 각 매개변수는
    # Tensor이므로 이전에 했던 것과 같이 변화도에 접근할 수 있습니다.
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad