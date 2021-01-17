import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader

torch.manual_seed(1)

x_train = torch.FloatTensor([[73, 80, 75],
                             [93, 88, 93],
                             [89, 91, 90],
                             [96, 98, 100],
                             [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

dataset = TensorDataset(x_train, y_train)

"""파이토치의 데이터셋을 만들었다면 데이터로더를 사용 가능합니다.
데이터로더는 기본적으로 2개의 인자를 입력받는다.
하나는 데이터셋, 미니 배치의 크기입니다.
이때 미니 배치의 크기는 통상적으로 2의 배수를 사용합니다.
(ex) 64, 128, 256...) 그리고 추가적으로 많이 사용되는 인자로 shuffle이 있습니다.
shuffle=True를 선택하면 Epoch마다 데이터셋을 섞어서 데이터가 학습되는 순서를 바꿉니다."""
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

model = nn.Linear(3, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 20
for epoch in range(nb_epochs + 1):
    for batch_idx, samples in enumerate(dataloader):
        # print(batch_idx)
        # print(samples)
        x_train, y_train = samples
        # H(x) 계산
        prediction = model(x_train)

        # cost 계산
        cost = F.mse_loss(prediction, y_train)

        # cost로 H(x) 계산
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
            epoch, nb_epochs, batch_idx + 1, len(dataloader),
            cost.item()
        ))
