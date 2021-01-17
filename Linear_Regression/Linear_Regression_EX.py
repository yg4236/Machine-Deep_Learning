import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#재실행해도 같은결과가 나오도록 랜덤시드를 줌
torch.manual_seed(1)

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

# 가중치와 편항를 0으로 초기화, 학습을 통해 값이 변경되는 변수임을 명시
W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# 경사 하강법(SGD) 구현
optimizer = optim.SGD([W, b], lr=0.01)

epochs = 2000
for epoch in range(epochs+1):
    # 가설 (직선의 방정식)
    hypothesis = x_train * W + b
    # 비용(손실)함수  (평균 제곱 오차)
    cost = torch.mean((hypothesis - y_train) ** 2)
    # gradient를 0으로 초기화
    optimizer.zero_grad()
    # 비용 함수를 미분하여 gradient 계산
    cost.backward()
    # W와 b를 업데이트
    optimizer.step()
    #100번마다 로그 출력
    if epoch % 100 == 0:
        print('Epoch {:4d}/{} W: {:.3f}, b: {:.3f} Cost: {:.6f}'.format(
            epoch, epochs, W.item(), b.item(), cost.item()
        ))

