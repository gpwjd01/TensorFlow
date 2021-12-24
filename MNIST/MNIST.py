import tensorflow as tf

# MNIST 손글씨 데이터 package 가져오기
mnist = tf.keras.datasets.mnist

# MNIST 4분할 데이터 불러오기
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
print('학습용 입력 데이터 모양:', X_train.shape)
print('학습용 출력 데이터 모양:', Y_train.shape)
print('평가용 입력 데이터 모양:', X_test.shape)
print('평가용 출력 데이터 모양:', Y_test.shape)

# 이미지 데이터 원본 출력
import matplotlib.pyplot as plt

plt.imshow(X_train[0], cmap='gray') # 그레이 스케일
# plt.show() # 이미지 출력

# print('\n첫번째 학습용 입력 데이터 모양:', X_train[0])
# print('첫번째 학습용 출력 데이터 모양:', Y_train[0])

# 이미지 데이터 [0, 1] 스케일링: 정확도 향상
X_train = X_train / 255.0
X_test = X_test / 255.0

# 스케일링 후 데이터 확인
plt.imshow(X_train[0], cmap='gray') # 그레이 스케일
# plt.show() # 이미지 출력
# print('첫번째 학습용 입력 데이터 모양:', X_train[0])

# 인공신경망 구현
model = tf.keras.models.Sequential()
layers = tf.keras.layers

model.add(layers.Flatten(input_shape=(28, 28)))
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0, 2))
model.add(layers.Dense(10, activation='softmax'))

# 인공신경망 요약
model.summary()

# 인공신경망 학습 환경 설정
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 인공신경망 학습
model.fit(X_train, Y_train, epochs=5, verbose=0)

# 인공신경망 평가 : 정확도 출력
model.evaluate(X_test, Y_test, verbose=1)

# 인공신경망 예측
pick = X_test[0].reshape(1, 28, 28)
pred = model.predict(pick)
answer = tf.argmax(pred, axis=1)

print('\n인공신경망 추측 결과 (원본):', pred)
print('인공신경망 추측 결과 (해석):', answer)
print('정답:', Y_test[0])