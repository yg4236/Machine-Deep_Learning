import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dropout, Dense, Activation
import datetime

#데이터 셋 가져오기
data = pd.read_csv('005930.KS.csv')
#값 설정
high_prices = data['High'].values.astype(int)
low_prices = data['Low'].values.astype(int)

mid_prices = ((high_prices+low_prices)/2).astype(int)
#윈도우 만들기
seq_len = 50 #50일 단위로 끊어서 예측(50개를 보고 1개 예측)
sequence_length = seq_len+1
result = []
for index in range(len(mid_prices)-sequence_length):
    result.append(mid_prices[index: index+sequence_length])
#데이터 정규화
normalized_data = []
for window in result:
    #normalized_window = float(window)/float(result[0])-1
    #print(normalized_window)
    normalized_window = [((float(p)/float(window[0]))-1) for p in window]
    normalized_data.append(normalized_window)

result = np.array(normalized_data)

#데이터 분리(학습, 테스트)
row = int(round(result.shape[0]*0.9))
train = result[:row, :]
np.random.shuffle(train)

x_train = train[:, :-1]
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
y_train = train[:, -1]

x_test = result[row:, :-1]
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
y_test = result[row:, -1]

# x_train.shape, x_test.shape -> ((1057, 50, 1), (117, 50, 1))

#모델 빌드
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(50, 1)))
model.add(LSTM(64, return_sequences=False))
model.add(Dense(1, activation='linear'))
model.compile(loss='mse', optimizer='rmsprop')

print(model.summary())


#Training
model.fit(x_train, y_train,
          validation_data=(x_test, y_test),
          batch_size=10,
          epochs=20)
#Prediction
pred = model.predict(x_test)
fig = plt.figure(facecolor='white')
ax = fig.add_subplot(111)
ax.plot(y_test, label='True')
ax.plot(pred, label='Prediction')
ax.legend()
plt.show()