# 파이썬 패키지 가져오기
import matplotlib.pyplot as plt
import numpy as np
from time import time
import os
import glob
from keras.datasets import mnist
from keras.layers import Dense, Flatten, Reshape
from keras.layers import LeakyReLU
from keras.models import Sequential

# 하이퍼 파라미터
MY_GEN = 128
MY_DIS = 128
MY_NOISE = 100 # 가짜 이미지를 생성할 때 사용
MY_SHAPE = (28, 28, 1) # 손글씨 이미지, 채널 정보:1
MY_EPOCH = 5000
MY_BATCH = 300

# 출력 이미지 폴더 생성: 가짜 이미지
MY_FOLDER = 'output/'
os.makedirs(MY_FOLDER, exist_ok=True)

for f in glob.glob(MY_FOLDER + '*'): # 파일 상관없이 모두 방문
    os.remove(f)

###### 데이터 준비 ######

# 결과는 numpy의 n-차원 행렬 형식
def read_data():
    # 학습용 입력값만 사용 (GAN은 비지도 학습)
    (X_train, _), (_, _) = mnist.load_data()
    print('데이터 모양:', X_train.shape) # 3차원
    # plt.imshow(X_train[0], cmap='gray')
    # plt.show()

    # [-1, 1] 데이터 스케일링
    X_train = X_train / 127.5 - 1.0

    # 채널 정보 추가
    X_train = np.expand_dims(X_train, axis=3)
    print('데이터 모양:', X_train.shape) # 4차원

    return X_train # 진짜 손글씨 60000만

read_data()


###### 인공 신경망 구현 ######

# 감별자를 속이는 것이 목적
# 100 -> 128(Dense) -> 128(leakyR) -> 128(Dense) -> 128(leakyR) -> 128(Dense) -> 28*28*1(Reshape)
# 생성자 설계
def build_generator():
    model = Sequential()

    # 입력층 + 은닉층 1
    model.add(Dense(MY_GEN, input_dim=MY_NOISE))
    model.add(LeakyReLU(alpha=0.01))
    # 은닉층 2
    model.add(Dense(MY_GEN))
    model.add(LeakyReLU(alpha=0.01))
    # 은닉층 3 + 출력층
    # tanh 활성화는  [-1, 1] 스케일링 때문이다.
    model.add(Dense(28*28*1, activation='tanh'))
    model.add(Reshape(MY_SHAPE))

    print('\n생성자 요약')
    model.summary()

    return model

# 생성자의 가짜 이미지를 감별하는 것이 목적
# 28*28*1 -> 784(Flatten) -> 128(Dense)- > 128(leakyR) -> 1(sigmoid)
# 감별자 설계
def build_discriminator():
    model = Sequential()

    # 입력층
    model.add(Flatten(input_shape=MY_SHAPE))
    # 은닉층 1
    model.add(Dense(MY_DIS))
    model.add(LeakyReLU(alpha=0.01))
    # 출력층
    model.add(Dense(1, activation='sigmoid'))

    print('\n감별자 요약')
    model.summary()

    return model

build_discriminator()

# 입력: 노이즈 벡터(생성자의 출력), 출력: 감별 결과(감별자의 출력), 생성자의 출력이 감별자의 입력이 됨
# DNN-GNN 구현
def build_GAN():
    model = Sequential()

    # 생성자 구현: 가짜 이미지
    generator = build_generator()

    # 감별자 구현
    # 생성자 학습시 감별자 고정
    discriminator = build_discriminator()
    discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    discriminator.trainable = False

    # GAN 구현: 생성자 먼저 추가하고 감별자
    model.add(generator)
    model.add(discriminator)

    # GAN은 정확도 무의미
    model.compile(optimizer='adam', loss='binary_crossentropy')

    # 생성자: 감별자 필요(필요 사항), 감별자 고정 필요(가중치 고정), DNN-GAN 전체(compile)
    # 감별자: 생성자 불필요(필요 사항), 생성자 고정 불필요(가중치 고정), 감별자만(compile)
    print('\nGAN 요약')
    model.summary()

    return discriminator, generator, model

###### 인공 신경망 학습 ######

# 한 batch로 학습하고, 한 번 가중치 보정
# 입력: (1) 학습용 입력 데이터, (2) 학습용 출력 데이터 / 출력: 평균 손실값(한 batch당)
# train_on_batch: batch 하나(입력 데이터), 매 batch마다 한 번(가중치 보적), batch 당 추가 정보 계산(주 목), 무의미(epoch)
# fit: 학습용 데이터 전체(입력 데이터), 매 batch마다 한 번(가중치 보정), 모든 데이터로 전체 학습 진행(주 목적), 중요한 파라미터(epoch)

# 감별자 학습 방법
def train_discriminator():
    # 진짜 이미지로 한 batch 추출
    total = X_train.shape[0]
    pick = np.random.randint(0, total, MY_BATCH)
    image = X_train[pick]

    # 숫자 1을 한 batch 생성
    all_1 = np.ones((MY_BATCH, 1))

    # 진짜 이미지(1)로 감별자 한 번 학습: 0과 1사이의 확률,
    d_loss_real = discriminator.train_on_batch(image, all_1) # [손실값, 정확도]
    # print(d_loss_real)

    # 표준 정규 분포: 표준편차 1, 평균 0
    # 생성자를 이용해 가짜 이미지 생성
    # 노이즈 벡터는 표준 정규 분포를 사용
    noise = np.random.normal(0, 1, (MY_BATCH, MY_NOISE))
    fake = generator.predict(noise)
    # print(fake.shape)

    # 숫자 0을 한 batch 생성
    all_0 = np.zeros((MY_BATCH, 1)) # 생성자가 만들어낸 가짜 이미지
    # print(all_0.shape)

    # 가짜 이미지로 감별자 한 번 학습: 300개 데이터로 학습
    d_loss_fake = discriminator.train_on_batch(fake, all_0)
    # print(d_loss_fake)

    # 평균 손실과 정확도 계산
    d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
    # print(d_loss)

    return d_loss

# 생성자 학습 방법: 감별자 속이기
def train_generator():
    # 노이즈 벡터는 표준 정규 분포를 사용
    noise = np.random.normal(0, 1, (MY_BATCH, MY_NOISE))
    # print(noise.shape)

    # 숫자 1을 한 batch 생성
    all_1 = np.ones((MY_BATCH, 1))
    # print(all_1.shape)

    # 가짜 이미지로 생성자 한 번 학습
    g_loss = gan.train_on_batch(noise, all_1)
    # print(g_loss) # 생성자가 만들어낸 손실값

    return g_loss

# 샘플 이미지 N x N 출력
def sample(epoch):
    row = col = 4

    # 노이즈 벡터 생성
    noise = np.random.normal(0, 1, (row*col, MY_NOISE))
    # print(noise.shape)

    # 생성자를 이용해 가짜 이미지 생성
    fake = generator.predict(noise)
    # print(fake.shape)

    # 채널 정보 삭제
    fake = np.squeeze(fake)
    # print(fake.shape)

    # 캔버스 만들기
    fig, spot = plt.subplots(row, col) # 전체, 위치

    # i행 j열에 가짜 이미지 추가
    cnt = 0
    for i in range(row):
        for j in range(col):
            spot[i, j].imshow(fake[cnt], cmap='gray')
            spot[i, j].axis('off')
            cnt += 1

    # 이미지를 PNG 파일로 저장
    path = os.path.join(MY_FOLDER, 'img-{}'.format((epoch)))
    plt.savefig(path)
    plt.close()

# GAN 학습
def train_GAN():
    begin = time()
    print('\nGAN 학습 시작')

    for epoch in range(MY_EPOCH + 1):
        d_loss = train_discriminator() # 손실값, 정확도
        g_loss = train_generator() # 손실값

        # 매 50번 학습 때마다 결과와 샘플 이미지 생성
        if epoch % 50 == 0:
            print('epoch:', epoch,
                  '생성자 손실: {:.3f}'.format(g_loss),
                  '감별자 손실: {:.3f}'.format(d_loss[0]),
                  '감별자 정확도: {:.1f}%'.format(d_loss[0] * 100))
            sample(epoch)
    end = time()
    print('최종 학습 시간: {:.1f}초'.format(end - begin))

###### 컨트롤 타워 ######

# 데이터 준비
X_train = read_data()

# GAN 구현
discriminator, generator, gan = build_GAN() # 빌드 간 함수 사용

# GAN 학습
train_GAN()