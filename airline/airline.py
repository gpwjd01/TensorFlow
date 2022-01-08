# 파이썬 패키지 가져오기
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import InputLayer, Dense
from keras.layers import LSTM

# 하이퍼 파라미터 지정
MY_PAST = 12
MY_SPLIT = 0.8 # 80%을 학습용으로 사용
MY_UNIT = 300
MY_SHAPE = (MY_PAST, 1)
MY_EPOCH = 300
MY_BATCH = 64
np.set_printoptions(precision=3) # 소수점 3자리까지 출력

# 데이터 파일 읽기
# 결과는 pandas의 데이터 프레임 형식
raw = pd.read_csv('airline.csv', header=None, usecols=[1])

# 시계열 데이터 시각화
# plt.plot(raw)
# plt.show()

# 데이터 원본 출력
print('원본 데이터 샘플 13개')
print(raw.head(13))

print('\n원본 데이터 통계')
print(raw.describe())

# MinMax 데이터 정규화
scaler = MinMaxScaler()
s_data = scaler.fit_transform(raw)
print('\nMinMax 정규화 형식:', type(s_data))

# 정규화 데이터 출력
df = pd.DataFrame(s_data)
print('\n정규화 데이터 샘플 13개')
print(df.head(13))

print('\n 정규화 데이터 통계')
print(df.describe())

# 13묶음으로 데이터 분할
# 결과는 python 리스트
bundle = []
for i in range(len(s_data) - MY_PAST):
    bundle.append(s_data[i : i+MY_PAST+1])

# 데이터 분할 결과 확인
print('\n총 13개 묶음의 수:', len(bundle))
print(bundle[0])
print(bundle[1])

# numpy로 전환
print('분할 데이터의 타입:', type(bundle))
bundle = np.array(bundle)

# 데이터를 입력과 출력으로 분할
X_data = bundle[:, 0:MY_PAST]
Y_data = bundle[:, -1]

# 데이터를 학습용과 평가용으로 분할
split = int(len(bundle) * MY_SPLIT)
X_train = X_data[: split]
X_test = X_data[split:]
Y_train = Y_data[: split]
Y_test = Y_data[split:]

# 최종 데이터 모양
print('\n학습용 입력 데이터 모양:', X_train.shape)
print('학습용 출력 데이터 모양:', Y_train.shape)
print('평가용 입력 데이터 모양:', X_test.shape)
print('평가용 출력 데이터 모양:', Y_test.shape)

##### 인공 신경망 구현 #####

# RNN 구현
# keras RNN은 2차원 입력만 허용
model = Sequential()
model.add(InputLayer(input_shape=MY_SHAPE))
model.add(LSTM(MY_UNIT)) # 출력 데이터의 수

model.add(Dense(1, activation='sigmoid')) # 출력 뉴런의 최종값
print('\nRNN 요약')
model.summary()

##### 인공 신경망 학습 #####

# 최적화 함수와 손실 함수 지정
model.compile(optimizer='rmsprop', loss='mse')
begin = time()
print('\nRNN 학습 시작')

model.fit(X_train, Y_train, epochs=MY_EPOCH, batch_size=MY_BATCH, verbose=0)
end = time()
print('총 학습 시간: {:.1f}초'.format(end - begin))

##### 인공 신경망 평가 #####

# RNN 평가
loss = model.evaluate(X_test, Y_test, verbose=1)
print('\n최종 MSE 손실값: {:.3f}'.format(loss)) # 소수점 3자리

# RNN 추측
pred = model.predict(X_test)
pred = scaler.inverse_transform(pred)
pred = pred.flatten().astype(int) # 소수를 정수로 전환
print('\n추측 결과 원본:', pred)

# 정답 역적환
truth = scaler.inverse_transform(Y_test)
truth = truth.flatten().astype(int)
print('\n정답 원본:', truth)

# line plot 구성
axes = plt.gca()
axes.set_ylim([0, 650]) #

sns.lineplot(data=pred, label='pred', color='blue')
sns.lineplot(data=truth, label='truth', color='red')
plt.show()