# 11장

import tensorflow as tf, os

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 메모리 증가 방지 설정
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
os.putenv('TF_GPU_ALLOCATOR', 'cuda_malloc_async')


import numpy as np
import tensorflow as tf
from tensorflow import keras

keras.layers.Dense(10, activation='relu', kernel_initializer='he_normal')

he_avg_init = keras.initializers.VarianceScaling(scale=2, mode='fan_avg', distribution='uniform')
keras.layers.Dense(10, activation='sigmoid', kernel_initializer=he_avg_init);
keras.layers.Dense(10, kernel_initializer='he_normal')
keras.layers.LeakyReLU(alpha=0.2)

keras.layers.Dense(10, activation='selu', kernel_initializer='lecun_normal');
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.BatchNormalization(momentum=0.99),  # momentum: 기본 0.99 미니배치가 작을수록 소수점 뒤에 9를 넣어 1에 가깝게 만듦
    keras.layers.Dense(300, activation='elu', kernel_initializer='he_normal'),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100, kernel_initializer='he_normal', use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation('elu'),
    keras.layers.Dense(10, activation='softmax'),
])

model.summary()
optimizer = keras.optimizers.SGD(clipvalue=1.0, clipnorm=1.0)
# clipvalue=1.0: loss의 모든 편미분 값을 -1.0 ~ 1.0으로 잘라냄.
# clipnorm=1.0: 해당 값 기준으로 정규화
# 두 인자 모두 기입 시 norm을 먼저 적용
model.compile(loss='mse', optimizer=optimizer)


def split_dataset(X, y):
    y_5_or_6 = (y == 5) | (y == 6)  # sandals or shirts
    y_A = y[~y_5_or_6]
    y_A[y_A > 6] -= 2  # class indices 7, 8, 9 should be moved to 5, 6, 7
    y_B = (y[y_5_or_6] == 6).astype(np.float32)  # binary classification task: is it a shirt (class 6)?
    return ((X[~y_5_or_6], y_A),
            (X[y_5_or_6], y_B))


(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train_full = X_train_full / 255.0
X_test = X_test / 255.0
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

(X_train_A, y_train_A), (X_train_B, y_train_B) = split_dataset(X_train, y_train)
(X_valid_A, y_valid_A), (X_valid_B, y_valid_B) = split_dataset(X_valid, y_valid)
(X_test_A, y_test_A), (X_test_B, y_test_B) = split_dataset(X_test, y_test)
X_train_B = X_train_B[:200]
y_train_B = y_train_B[:200]
model_A = keras.models.Sequential()
model_A.add(keras.layers.Flatten(input_shape=[28, 28]))
for n_hidden in (300, 100, 50, 50, 50):
    model_A.add(keras.layers.Dense(n_hidden, activation="selu"))
model_A.add(keras.layers.Dense(8, activation="softmax"))
model_A.compile(loss="sparse_categorical_crossentropy",
                optimizer=keras.optimizers.SGD(learning_rate=1e-3),
                metrics=["accuracy"])
history = model_A.fit(X_train_A, y_train_A, epochs=20,
                      validation_data=(X_valid_A, y_valid_A))
model_B = keras.models.Sequential()
model_B.add(keras.layers.Flatten(input_shape=[28, 28]))
for n_hidden in (300, 100, 50, 50, 50):
    model_B.add(keras.layers.Dense(n_hidden, activation="selu"))
model_B.add(keras.layers.Dense(1, activation="sigmoid"))
model_B.compile(loss="binary_crossentropy",
                optimizer=keras.optimizers.SGD(learning_rate=1e-3),
                metrics=["accuracy"])
history = model_B.fit(X_train_B, y_train_B, epochs=20,
                      validation_data=(X_valid_B, y_valid_B))
model_B_on_A = keras.models.Sequential(model_A.layers[:-1])  # output 제외 전체 layer 반환
model_B_on_A.add(keras.layers.Dense(1, activation='sigmoid'))
model_A_clone = keras.models.clone_model(model_A)  # 모델 구조 복사, 가중치는 복제하지 않음
model_A_clone.set_weights(model_A.get_weights())  # 가중치 복제
for layer in model_B_on_A.layers[:-1]:
    layer.trainable = False  # 출력층 제외 가중치 동결
# 층을 동결하거나 동결 해제 후 새로 컴파일 필수
model_B_on_A.compile(loss='binary_crossentropy', optimizer='sgd', metrics=['accuracy'])
history = model_B_on_A.fit(X_train_B, y_train_B, epochs=4, validation_data=(X_valid_B, y_valid_B))

for layer in model_B_on_A.layers[:-1]:
    layer.trainable = True

optimizer = keras.optimizers.SGD(learning_rate=1e-4)  # 전이 학습은 학습률을 더 낮게 줌
model_B_on_A.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history = model_B_on_A.fit(X_train_B, y_train_B, epochs=16, validation_data=(X_valid_B, y_valid_B))
model_B_on_A.evaluate(X_test_B, y_test_B)
optimizer = keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
optimizer = keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=True)
optimizer = keras.optimizers.RMSprop(learning_rate=0.001, rho=0.9)
optimizer = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
# 거듭제곱법
optimizer = keras.optimizers.SGD(learning_rate=0.01, decay=1e-4)


# 지수
def exponential_decay_fn(epoch):
    return 0.01 * 0.1 ** (epoch / 20)


def exponential_decay(lr0, s):
    def exponential_decay_fn(epoch):
        return 0.01 * 0.1 ** (epoch / 20)

    return exponential_decay_fn


exponential_decay_fn = exponential_decay(lr0=0.01, s=20)
# callback 기능을 이용하기 때문에 위의 형태로 작성

lr_scheduler = keras.callbacks.LearningRateScheduler(exponential_decay_fn)
history = model.fit(X_train, y_train, batch_size=1028, epochs=200, callbacks=[lr_scheduler])


def exponential_decay_fn(epoch, lr):
    return lr * 0.1 ** (1 / 20)


# 구간별 고정
def piecewuse_constant_fn(epoch):
    if epoch < 5:
        return 0.01
    elif epoch < 15:
        return 0.005
    else:
        return 0.001


# 성능 기반
lr_scheduler = keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
# 연속 patience에폭 동안 vla_loss가 개션되지 않을 때 factor를 학습률에 곱함
import math


### 1사이클
# 최적 학습률 확인
class ExponentialLearningRate(keras.callbacks.Callback):
    def __init__(self, factor):
        self.factor = factor
        self.rates = []
        self.losses = []

    def on_epoch_begin(self, epoch):
        self.prev_loss = 0

    def on_batch_end(self, batch, logs=None):
        batch_loss = logs["loss"] * (batch + 1) - self.prev_loss * batch
        self.prev_loss = logs["loss"]
        self.rates.append(model.optimizer.lr.numpy())
        self.losses.append(batch_loss)
        self.model.optimizer.lr = self.model.optimizer.lr * self.factor


def find_learning_rate(model, X, y, epochs=1, batch_size=32, min_rate=10 ** -5, max_rate=10):
    init_weights = model.get_weights()
    iterations = math.ceil(len(X) / batch_size) * epochs
    factor = np.exp(np.log(max_rate / min_rate) / iterations)
    init_lr = model.optimizer.lr.numpy()
    model.optimizer.lr = min_rate
    exp_lr = ExponentialLearningRate(factor)
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size,
                        callbacks=[exp_lr])
    model.optimizer.lr = init_lr
    model.set_weights(init_weights)

    return exp_lr.rates, exp_lr.losses


# 1사이클 클래스
class OneCycleScheduler(keras.callbacks.Callback):
    def __init__(self, iterations, max_rate, start_rate=None, last_iterations=None, last_rate=None):
        self.total_iteration = iterations  # 총 학습률 조정 반복 횟수
        self.max_rate = max_rate  # 최대 학습률
        self.start_rate = start_rate or max_rate / 10  # 시작 학습률 (디폴트는 최대 학습률의 10%)
        self.last_iterations = last_iterations or iterations // 10 + 1  # 마지막 단계의 반복 횟수 (디폴트는 총 반복 횟수의 10%)
        self.half_iteration = (iterations - self.last_iterations) // 2  # 중간 단계 반복 횟수
        self.last_rate = last_rate or self.start_rate / 1000  # 마지막 학습률 (디폴트는 시작 학습률의 1/1000)
        self.current_iteration = 0  # 현재 반복 횟수 초기화

    def _interpolate(self, from_iter, to_iter2, from_rate, to_rate):
        # 두 지점 사이에서 선형 보간을 통해 학습률 계산하여 to_iter까지 선형적으로 rate 변화
        return ((to_rate - from_rate) * (self.current_iteration - from_iter) / (to_iter2 - from_iter) + from_rate)

    def on_batch_begin(self, batch, logs):
        if self.current_iteration < self.half_iteration:
            # 초기 상승 단계
            rate = self._interpolate(0, self.half_iteration, self.start_rate, self.max_rate)
        elif self.current_iteration < 2 * self.half_iteration:
            # 최대 학습률로 상승한 후 하락 단계
            rate = self._interpolate(self.half_iteration, 2 * self.half_iteration, self.max_rate, self.start_rate)
        else:
            # 마지막 하락 단계
            rate = self._interpolate(2 * self.half_iteration, self.total_iteration, self.start_rate, self.last_rate)
        self.current_iteration += 1  # 반복 횟수 증가
        self.model.optimizer.lr = rate  # 모델의 학습률 업데이트


# 규제 적용 방식
layer = keras.layers.Dense(100, activation='elu',
                           kernel_initializer='he_normal',
                           kernel_regularizer=keras.regularizers.l1(0.01))
layer = keras.layers.Dense(100, activation='elu',
                           kernel_initializer='he_normal',
                           kernel_regularizer=keras.regularizers.l2(0.01))
layer = keras.layers.Dense(100, activation='elu',
                           kernel_initializer='he_normal',
                           kernel_regularizer=keras.regularizers.l1_l2(0.01, 0.01))
from functools import partial

# partial: 함수의 인자 기본값을 새로 지정하여 사용할 수 있게 함.

RegularizedDense = partial(keras.layers.Dense,
                           activation='elu',
                           kernel_initializer='he_normal',
                           kernel_regularizer=keras.regularizers.l1_l2(0.01, 0.01))

model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    RegularizedDense(300),
    RegularizedDense(100, activation='relu'),
    RegularizedDense(10, activation='softmax', kernel_initializer='glorot_uniform')
])

[print(layer.activation) for layer in model.layers[1:]];
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[29, 29]),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(300, activation='elu', kernel_initializer='he_normal'),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(100, activation='elu', kernel_initializer='he_normal'),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(10, activation='softmax')
])
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import accuracy_score

# 데이터 준비 (예제 데이터 사용)
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
x_train = x_train.reshape(-1, 3072).astype('float32') / 255
x_test = x_test.reshape(-1, 3072).astype('float32') / 255

# 모델 정의
model = Sequential([
    Dense(300, activation='relu', input_shape=(3072,)),
    Dropout(0.5),  # Dropout 층 추가
    Dense(100, activation='relu'),
    Dropout(0.5),  # Dropout 층 추가
    Dense(10, activation='softmax')
])

# 모델 컴파일
model.compile(loss="sparse_categorical_crossentropy", optimizer='adam', metrics=["accuracy"])

# 모델 훈련
model.fit(x_train, y_train, batch_size=1024, epochs=100, validation_split=0.2)

# 검증
y_probas = np.stack([model(x_test, training=True)
                     for sample in range(50)])
y_proba = y_probas.mean(axis=0)
y_pred = np.argmax(y_proba, axis=1)

print(accuracy_score(y_test, np.argmax(model.predict(x_test), axis=1)))
print(accuracy_score(y_test, y_pred))


class MCDropout(keras.layers.Dropout):
    def call(self, inputs):
        return super().call(inputs, training=True)


mc_model = keras.models.Sequential([
    MCDropout(layer.rate) if isinstance(layer, keras.layers.Dropout) else layer
    for layer in model.layers
])

optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
mc_model.compile(loss="sparse_categorical_crossentropy", optimizer=optimizer, metrics=["accuracy"])
mc_model.set_weights(model.get_weights())

# y_probas = np.stack([mc_model(x_test, training=True)
#                      for sample in range(1000)])
y_probas = np.stack([mc_model.predict(x_test)
                     for sample in range(10)])

y_proba = y_probas.mean(axis=0)
y_pred = np.argmax(y_proba, axis=1)

print(accuracy_score(y_test, np.argmax(mc_model.predict(x_test), axis=1)))
print(accuracy_score(y_test, y_pred))
keras.layers.Dense(100, activation='elu', kernel_initializer='he_normal',
                   kernel_constraint=keras.constraints.max_norm(1.));
import tensorflow as tf

keras.backend.clear_session()
tf.random.set_seed(42)
np.random.seed(42)

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[32, 32, 3]))
for _ in range(20):
    model.add(keras.layers.Dense(100,
                                 activation="elu",
                                 kernel_initializer="he_normal"))
model.add(keras.layers.Dense(10, activation="softmax"))

optimizer = keras.optimizers.Nadam(learning_rate=5e-5)
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=optimizer,
              metrics=["accuracy"])

(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.cifar10.load_data()

X_train = X_train_full[2000:10000] / 255
y_train = y_train_full[2000:10000].flatten()
X_valid = X_train_full[:2000] / 255
y_valid = y_train_full[:2000].flatten()

early_stopping_cb = keras.callbacks.EarlyStopping(patience=20)
callbacks = [early_stopping_cb]

history = model.fit(X_train, y_train, epochs=200, batch_size=512, validation_data=(X_valid, y_valid), verbose=2)
## 연습문제 구현
import numpy as np, os, math
from tensorflow import keras


def get_run_logdir(idx):
    return os.path.join(os.getcwd(), 'board', f'run_{idx:02}')


(X_train_full, y_train_full), (X_test_full, y_test_full) = keras.datasets.cifar10.load_data()

X_train = X_train_full[2000:10000] / 255
y_train = y_train_full[2000:10000].flatten()
X_valid = X_train_full[:2000] / 255
y_valid = y_train_full[:2000].flatten()
X_test = X_test_full / 255
y_test = y_test_full.flatten()

early_stopping_cb = keras.callbacks.EarlyStopping(patience=20)
from tensorflow import keras
import numpy as np


class MakeModel(keras.models.Sequential):
    def __init__(self, input_num, hidden_nums, output_num, **kwargs):
        super().__init__([keras.layers.Input(shape=input_num),
                          keras.layers.Flatten(input_shape=[32, 32, 3])] +
                         [keras.layers.Dense(hidden_num, 'elu', kernel_initializer='he_normal')
                          for hidden_num in hidden_nums] +
                         [keras.layers.Dense(10, 'softmax')]
                         )

        self.compile('nadam', 'sparse_categorical_crossentropy', ['accuracy'])


h_ls = np.linspace(1028, 100, 20).astype(int).tolist()
model = MakeModel((32, 32, 3), h_ls, 10)

history = model.fit(X_train, y_train, epochs=100, batch_size=512, validation_data=(X_valid, y_valid), verbose=2,
                    callbacks=[early_stopping_cb, keras.callbacks.TensorBoard(get_run_logdir(1))])
model.evaluate(X_test, y_test)


class MakeModel(keras.models.Model):
    def __init__(self, input_num, hidden_nums, output_num, **kwargs):
        x = keras.layers.Input(shape=input_num)
        h = keras.layers.Flatten(input_shape=[32, 32, 3])(x)
        for hidden_num in hidden_nums:
            h = keras.layers.Dense(hidden_num, 'elu', kernel_initializer='he_normal')(h)
            h = keras.layers.BatchNormalization()(h)
        y = keras.layers.Dense(10, 'softmax')(h)
        super().__init__(x, y)
        self.compile('nadam', 'sparse_categorical_crossentropy', ['accuracy'])


h_ls = np.linspace(1028, 100, 20).astype(int).tolist()
model = MakeModel((32, 32, 3), h_ls, 10)

history = model.fit(X_train, y_train, epochs=100, batch_size=512, validation_data=(X_valid, y_valid), verbose=2,
                    callbacks=[early_stopping_cb, keras.callbacks.TensorBoard(get_run_logdir(2))])
model.evaluate(X_test, y_test)
pixel_means = X_train_full[2000:10000].mean(axis=0, keepdims=True)
pixel_stds = X_train_full[2000:10000].std(axis=0, keepdims=True)
X_train_scaled = (X_train_full[2000:10000] - pixel_means) / pixel_stds
X_valid_scaled = (X_train_full[:2000] - pixel_means) / pixel_stds
X_test_scaled = (X_test_full - pixel_means) / pixel_stds

x = keras.layers.Input(shape=X_train.shape[1:])
h = keras.layers.Flatten(input_shape=X_train.shape[1:])(x)
for hidden_num in h_ls:
    h = keras.layers.Dense(hidden_num, 'selu', kernel_initializer='lecun_normal')(h)
    h = keras.layers.BatchNormalization()(h)
y = keras.layers.Dense(10, 'softmax')(h)
model = keras.Model(x, y)
model.compile('nadam', 'sparse_categorical_crossentropy', ['accuracy'])

history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=512, validation_data=(X_valid_scaled, y_valid),
                    verbose=2,
                    callbacks=[early_stopping_cb, keras.callbacks.TensorBoard(get_run_logdir(3))])
model.evaluate(X_test_scaled, y_test)


class MakeModel(keras.models.Sequential):
    def __init__(self, input_num, hidden_nums, output_num, **kwargs):
        super().__init__()
        self.add(keras.layers.Input(shape=input_num))
        self.add(keras.layers.Flatten(input_shape=[32, 32, 3]))
        for hidden_num in hidden_nums:
            self.add(keras.layers.Dense(hidden_num, 'selu', kernel_initializer='lecun_normal'))
            self.add(keras.layers.BatchNormalization())
            self.add(keras.layers.AlphaDropout(0.5))
        self.add(keras.layers.Dense(10, 'softmax'))
        self.compile('nadam', 'sparse_categorical_crossentropy', ['accuracy'])


model = MakeModel((32, 32, 3), h_ls, 10)

history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=512, validation_data=(X_valid_scaled, y_valid),
                    verbose=2,
                    callbacks=[early_stopping_cb, keras.callbacks.TensorBoard(get_run_logdir(4))])
print(model.evaluate(X_test_scaled, y_test))


class MCDropout(keras.layers.AlphaDropout):
    def call(self, inputs):
        return super().call(inputs, training=True)


mc_model = keras.models.Sequential([
    MCDropout(layer.rate) if isinstance(layer, keras.layers.AlphaDropout) else layer
    for layer in model.layers
])

optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
mc_model.compile('nadam', 'sparse_categorical_crossentropy', ['accuracy'])
mc_model.set_weights(model.get_weights())
del model, model_A, model_B, model_B_on_A, model_A_clone

y_probas = []
for sample in range(100):
    y_probas += [np.array(mc_model.predict(X_test_scaled))]
y_probas = np.stack(y_probas)
# y_probas = np.stack([mc_model.predict(X_test_scaled)
#                      for sample in range(100)])

y_proba = y_probas.mean(axis=0)
y_pred = np.argmax(y_proba, axis=1)

print(accuracy_score(y_test, y_pred))


# 최적 학습률 확인
class ExponentialLearningRate(keras.callbacks.Callback):
    def __init__(self, factor):
        self.factor = factor
        self.rates = []
        self.losses = []

    def on_epoch_begin(self, epoch, logs=None):
        self.prev_loss = 0

    def on_batch_end(self, batch, logs=None):
        batch_loss = logs["loss"] * (batch + 1) - self.prev_loss * batch
        self.prev_loss = logs["loss"]
        self.rates.append(model.optimizer.lr.numpy())
        self.losses.append(batch_loss)
        self.model.optimizer.lr = self.model.optimizer.lr * self.factor


def find_learning_rate(model, X, y, epochs=1, batch_size=32, min_rate=10 ** -5, max_rate=10):
    init_weights = model.get_weights()
    iterations = math.ceil(len(X) / batch_size) * epochs
    factor = np.exp(np.log(max_rate / min_rate) / iterations)
    init_lr = model.optimizer.lr.numpy()
    model.optimizer.lr = min_rate
    exp_lr = ExponentialLearningRate(factor)
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size,
                        callbacks=[exp_lr])
    model.optimizer.lr = init_lr
    model.set_weights(init_weights)

    return exp_lr.rates, exp_lr.losses


model = MakeModel((32, 32, 3), h_ls, 10)
rates, losses = find_learning_rate(model, X_train_scaled, y_train, epochs=1, batch_size=1)
lr = rates[losses.index(min(losses)) - 10]
print(lr)


# 1사이클 클래스
class OneCycleScheduler(keras.callbacks.Callback):
    def __init__(self, iterations, max_rate, start_rate=None, last_iterations=None, last_rate=None):
        self.total_iteration = iterations  # 총 학습률 조정 반복 횟수
        self.max_rate = max_rate  # 최대 학습률
        self.start_rate = start_rate or max_rate / 10  # 시작 학습률 (디폴트는 최대 학습률의 10%)
        self.last_iterations = last_iterations or iterations // 10 + 1  # 마지막 단계의 반복 횟수 (디폴트는 총 반복 횟수의 10%)
        self.half_iteration = (iterations - self.last_iterations) // 2  # 중간 단계 반복 횟수
        self.last_rate = last_rate or self.start_rate / 1000  # 마지막 학습률 (디폴트는 시작 학습률의 1/1000)
        self.current_iteration = 0  # 현재 반복 횟수 초기화

    def _interpolate(self, from_iter, to_iter2, from_rate, to_rate):
        # 두 지점 사이에서 선형 보간을 통해 학습률 계산하여 to_iter까지 선형적으로 rate 변화
        return ((to_rate - from_rate) * (self.current_iteration - from_iter) / (to_iter2 - from_iter) + from_rate)

    def on_batch_begin(self, batch, logs):
        if self.current_iteration < self.half_iteration:
            # 초기 상승 단계
            rate = self._interpolate(0, self.half_iteration, self.start_rate, self.max_rate)
        elif self.current_iteration < 2 * self.half_iteration:
            # 최대 학습률로 상승한 후 하락 단계
            rate = self._interpolate(self.half_iteration, 2 * self.half_iteration, self.max_rate, self.start_rate)
        else:
            # 마지막 하락 단계
            rate = self._interpolate(2 * self.half_iteration, self.total_iteration, self.start_rate, self.last_rate)
        self.current_iteration += 1  # 반복 횟수 증가
        self.model.optimizer.lr = rate  # 모델의 학습률 업데이트


class MakeModel(keras.models.Sequential):
    def __init__(self, input_num, hidden_nums, output_num, **kwargs):
        super().__init__()
        self.add(keras.layers.Input(shape=input_num))
        self.add(keras.layers.Flatten(input_shape=[32, 32, 3]))
        for hidden_num in hidden_nums:
            self.add(keras.layers.Dense(hidden_num, 'selu', kernel_initializer='lecun_normal'))
            self.add(keras.layers.BatchNormalization())
        self.add(keras.layers.Dense(10, 'softmax'))
        self.compile('nadam', 'sparse_categorical_crossentropy', ['accuracy'])


model = MakeModel((32, 32, 3), h_ls, 10)

history = model.fit(X_train_scaled, y_train, epochs=100, batch_size=512, validation_data=(X_valid_scaled, y_valid),
                    verbose=2,
                    callbacks=[keras.callbacks.TensorBoard(get_run_logdir(5))])
print(model.evaluate(X_test_scaled, y_test))
