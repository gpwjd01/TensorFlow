# 파이썬 패키지 가져오기
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from time import time
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 하이퍼 파라미터
MY_EPOCH = 500  # 기계 학습 시 학습 데이터를 몇 번을 반복하며 학습할 지를 지정
MY_BATCH = 64   # 학습 데이터를 매번 메모리에서 몇 개씩 가져와서 계산할 지를 지정

########### 데이터 준비 ###########

# 데이터 파일 읽기
# 결과는 pandas의 데이터 프레임 형식
heading = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE',
           'DIS', 'RAD', 'TAX', 'PTRATIO', 'LSTAT', 'MEDV']
raw = pd.read_csv('housing.csv')

# 데이터 원본 출력
print('원본 데이터 샘플 10개')
print(raw.head(10))

print('원본 데이터 통계')
print(raw.describe())

# Z-점수 정규화 : 데이터의 특정 패턴을 찾음. 새로운 값을 예측 -> 기계 학습에 사용된 13개의 모든 요소들을 Z-점수화
# Z-점수 = (X-평균) / 표준편차
# 결과는 numpy의 n-차원 행렬 형식
scaler = StandardScaler()
Z_data = scaler.fit_transform(raw)

# numpy에서 pandas로 전환
# header 정보 복구 필요
Z_data = pd.DataFrame(Z_data, columns=heading) # pandas의 데이터프레임으로 재전환

# 정규화 된 데이터 출력 : -1과 1 사이의 숫자
print('정규화 된 데이터 샘플 10개')
print(Z_data.head(10))

print('정규화 된 데이터 통계') # 평균: 0에 가까운 숫자, 표준편차: 1에 가까운 숫자
print(Z_data.describe())

# 데이터를 입력과 출력으로 분리
print('\n분리 전 데이터 모양:', Z_data.shape)
X_data = Z_data.drop('MEDV', axis=1)
Y_data = Z_data['MEDV']

# 데이터를 학습용과 평가용으로 분리
X_train, X_test, Y_train, Y_test = train_test_split(X_data, Y_data, test_size=0.3)

print('\n학습용 입력 데이터 모양:', X_train.shape)    # 2차원
print('학습용 출력 데이터 모양:', Y_train.shape)
print('평가용 출력 데이터 모양:', X_test.shape)       # 2차원
print('평가용 출력 데이터 모양:', Y_test.shape)

# 상자 그림 출력
sns.set(font_scale=2)
sns.boxplot(data=Z_data, palette='dark')
plt.show()

######### 인공 신경망 구형 #########

# keras DNN 구현
model = Sequential()
input = X_train.shape[1]
model.add(Dense(200, input_dim=input, activation='relu'))  # 입력층 & 은닉층 1 추가
model.summary()                                            # 은닉층 1-None(현재 배치 데이터 표현)
model.add(Dense(1000, activation='relu'))                  # 은닉층 2 추가
model.add(Dense(1))
# 출력층 추가
print('\nDNN 요약')
model.summary()                                            # 최종 출력층의 정보: 출력층 뉴런 1개

######### 인공 신경망 학습 #########

# SGD(경사 하강법): 지도학습에서 가중치를 보정하는 대표적인 방법
# 함수의 기울기(경사)를 구해 낮은 쪽으로 이동

# 최적화 함수와 손실 함수 지정
model.compile(optimizer='sgd', loss='mse')
print('\nDNN 학습 시작')
begin = time()

# verbose=0 : 학습 과정 출력 X
# verbose=0 : 학습 과정 출력
model.fit(X_train, Y_train, epochs=MY_EPOCH, batch_size=MY_BATCH, verbose=0)
end = time()
print('총 학습 시간: {:.1f}초'.format(end - begin))

######### 인공 신경망 평가 및 활용 ##########

# 신경망 평가 및 손실값 계산
loss = model.evaluate(X_test, Y_test, verbose=0)
print('\nDNN 평균 제곱 오차 (MSE): {:.2f}'.format(loss))      # 임의의 숫자로 인해 MSE의 값이 바뀜

# 신경망 활용 및 산포도 출력
pred = model.predict(X_test)    # 예측값
sns.regplot(x=Y_test, y=pred)   # 산포도

plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.show()

#### DNN 학습 최적화 ####
# 실습 1: Epoch 조절 - MY_EPOCH = 0
# MSE 값 증가, 학습 시간 거의 0, 산포도: 추측값과 실제값의 상관관계가 거의 없음
# MY_EPOCH = 2000 : 학습 시간 증가, MSE 값 차이 없음

# 실습 2: Batch 조절 - MY_BATCH = 16
# 학습 시간 증가, 손실 값 영향 없음
# MY_BATCH = 354 : 학습 시간 증가, 손실값 차이 없음

#### DNN 구조 최적화 ####
# 실습 3: 은닉층 추가
# 은닉층 3: model.add(Dense(500, activation='relu')
# 은닉층 4: model.add(Dense(500, activation='relu')
# 파라미터 수 증가, 손실값 감소, 산포도는 추측값과 실제값의 상관관계도 개선, 학습시간 증가

#### Dataset 최적화 ####
# 실습 4: Z-점수 정규화 생략 - Z_data = raw
# 평균과 표준편차가 0과 1이 아님, MSE 결과는 NaN (계산 불가), 학습 시간 영향 없음

# 실습 5: 학습용 데이터 수 조절 (학습 데이터 감소 시키기)
# X_train = X_train.drop(X_train.index[177:])
# Y_train = Y_train.drop(Y_train.index[177:])
# MSE 증가, 학습 시간 감소

# 실습 6: 데이터 요소 제거
# raw = raw.drop('CRIM', axis=1)
# heading.pop(0)
# MSE 증가, 학습 시간 영향 미미

# 실습 7: DNN 추측 대상 교체
# X_data = Z_data.drop("AGE', axis=1)
# Y_data = Z_data['AGE']
# MSE 증가, 학습 시간 감소