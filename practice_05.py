# 12장
import numpy as np
from tensorflow import keras
import tensorflow as tf
import os

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # 메모리 증가 방지 설정
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)
os.putenv('TF_GPU_ALLOCATOR', 'cuda_malloc_async')
t = tf.constant([[1., 2., 3.], [4., 5., 6., ]])
print(t,
      t[:, :, tf.newaxis],
      t + 10,
      tf.square(t),
      t @ tf.transpose(t),
      tf.cast(t, tf.float16),
      sep='\n\n')
v = tf.Variable([[1., 2., 3.], [4., 5., 6.]])
print(v)
v.assign(2 * v)
print(v)
v[0, 1].assign(42)
print(v)
v[:, 1].assign([0., 1.])
print(v)
v.scatter_nd_update(indices=[[0, 0], [1, 2]], updates=[100., 200.])
print(v)


def huber_fn(y_true, y_pred):
    error = y_true - y_pred
    is_small_error = tf.abs(error) < 1
    squared_loss = tf.square(error) / 2
    linear_loss = tf.abs(error) - 0.5

    return tf.where(is_small_error, squared_loss, linear_loss)


from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target.reshape(-1, 1), random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)
input_shape = X_train.shape[1:]

model = keras.models.Sequential([
    keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal",
                       input_shape=input_shape),
    keras.layers.Dense(1),
])

model.compile(loss=huber_fn, optimizer="nadam", metrics=["mae"])

model.fit(X_train_scaled, y_train, epochs=2,
          validation_data=(X_valid_scaled, y_valid))
model.save("my_model_with_a_custom_loss.h5")
model = keras.models.load_model("my_model_with_a_custom_loss.h5",
                                custom_objects={"huber_fn": huber_fn})
model.fit(X_train_scaled, y_train, epochs=2,
          validation_data=(X_valid_scaled, y_valid))


def create_huber(threshold=1.0):
    def huber_fn(y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < threshold
        squared_loss = tf.square(error) / 2
        linear_loss = threshold * tf.abs(error) - threshold ** 2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)

    return huber_fn


model.compile(loss=create_huber(2.0), optimizer='nadam', metrics=['mae'])
model.save("my_model_with_a_custom_loss.h5")
model = keras.models.load_model("my_model_with_a_custom_loss.h5",
                                custom_objects={"huber_fn": create_huber(2.0)})
model.fit(X_train_scaled, y_train, epochs=2,
          validation_data=(X_valid_scaled, y_valid))


class HuberLoss(keras.losses.Loss):
    def __init__(self, threshold=1.0, **kwargs):
        self.threshold = threshold
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < self.threshold
        squared_loss = tf.square(error) / 2
        linear_loss = self.threshold * tf.abs(error) - self.threshold ** 2 / 2

        return tf.where(is_small_error, squared_loss, linear_loss)

    def get_config(self):
        base_config = super().get_config()

        return {**base_config, 'threshold': self.threshold}


model.compile(loss=HuberLoss(2.0), optimizer='nadam', metrics=['mae'])
model.save("my_model_with_a_custom_loss.h5")
model = keras.models.load_model("my_model_with_a_custom_loss.h5",
                                custom_objects={"HuberLoss": HuberLoss})
model.fit(X_train_scaled, y_train, epochs=2,
          validation_data=(X_valid_scaled, y_valid))


def my_softplus(z):
    return tf.math.log(tf.exp(z) + 1.0)


def my_glorot_initializer(shape, dtype=tf.float32):
    stddev = tf.sqrt(2. / (shape[0] + shape[1]))
    return tf.random.normal(shape, stddev=stddev, dtype=dtype)


def my_l1_regularizer(weights):
    return tf.reduce_sum(tf.abs(0.01 * weights))


def my_positive_weights(weights):
    return tf.where(weights < 0., tf.zeros_like(weights), weights)


class MyL1Regularizer(keras.regularizers.Regularizer):
    # 부모 클래스에 생성자와 get_config가 정의돼 있지 않아 호출(super) 불요.
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, weights):
        # loss, layer, model의 경우 call / regularizer, initializer, constraint의 경우 __call__
        return tf.reduce_sum(tf.abs(self.factor * weights))

    def get_config(self):
        return {'factor': self.factor}


layer = keras.layers.Dense(30, activation=my_softplus,
                           kernel_initializer=my_glorot_initializer,
                           kernel_regularizer=my_l1_regularizer,
                           kernel_constraint=my_positive_weights,
                           input_shape=input_shape)

model = keras.models.Sequential([
    layer,
    keras.layers.Dense(1, activation=my_softplus,
                       kernel_regularizer=MyL1Regularizer(0.01),
                       kernel_constraint=my_positive_weights,
                       kernel_initializer=my_glorot_initializer),
])
model.compile(loss="mse", optimizer="nadam", metrics=["mae"])
model.save("my_model_with_many_custom_parts.h5")
model = keras.models.load_model(
    "my_model_with_many_custom_parts.h5",
    custom_objects={
        "my_l1_regularizer": my_l1_regularizer,
        "my_positive_weights": my_positive_weights,
        "my_glorot_initializer": my_glorot_initializer,
        "my_softplus": my_softplus,
        'MyL1Regularizer': MyL1Regularizer
    })
model.compile(loss='mse', optimizer='nadam', metrics=[create_huber(2.0)])
model.fit(X_train_scaled, y_train, epochs=2,
          validation_data=(X_valid_scaled, y_valid))
precision = keras.metrics.Precision()
print(precision([0, 1, 1, 1, 0, 1, 0, 1], [1, 1, 0, 1, 0, 1, 0, 1]))
print(precision([0, 1, 0, 0, 1, 0, 1, 1], [1, 0, 1, 1, 0, 0, 0, 0]))
print(precision.result())
[print(_) for _ in precision.variables]
precision.reset_states()
print(precision.result())


class HuberMetric(keras.metrics.Metric):
    def __init__(self, threshold=1.0, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.huber_fn = create_huber(threshold)
        self.total = self.add_weight('total', initializer='zeros')
        self.count = self.add_weight('count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        metric = self.huber_fn(y_true, y_pred)
        # tf.Variable 객체이므로 assign_add 사용
        self.total.assign_add(tf.reduce_sum(metric))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        return self.total / self.count

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, 'threshold': self.threshold}


model.compile(loss=create_huber(2.0), optimizer="nadam", metrics=[HuberMetric(2.0)])
model.fit(X_train_scaled.astype(np.float32), y_train.astype(np.float32), epochs=2)
exponential_layer = keras.layers.Lambda(lambda x: tf.exp(x))  # 지수 함수


class MyDense(keras.layers.Layer):
    def __init__(self, units, activation=None, **kwargs):
        super().__init__(**kwargs)  # 부모 클래스의 초기화 메서드를 호출합니다.
        self.units = units  # 출력 유닛 수를 저장합니다.
        self.activation = keras.activations.get(activation)  # 활성화 함수를 가져옵니다.

    def build(self, batch_input_shape):
        # 커널 가중치: 입력 차원과 유닛 차원을 가지며, 'glorot_normal' 초기화 방법을 사용합니다.
        self.kernel = self.add_weight(
            name='kernel', shape=[batch_input_shape[-1], self.units],
            initializer='glorot_normal'
        )
        # 바이어스: 유닛 수에 해당하는 크기를 가지며, 0으로 초기화됩니다.
        self.bias = self.add_weight(
            name='bias', shape=[self.units], initializer='zeros'
        )
        super().build(batch_input_shape)  # 부모 클래스의 build 메서드를 호출하여 추가적인 작업을 수행합니다.

    def call(self, X):
        # X와 kernel의 행렬 곱에 bias를 더하고 활성화 함수를 적용합니다.
        return self.activation(X @ self.kernel + self.bias)

    def compute_output_shape(self, batch_input_shape):
        # 입력 모양에서 마지막 차원을 유닛 수로 대체하여 출력 모양을 계산합니다.
        return tf.TensorShape(batch_input_shape.as_list()[:-1] + [self.units])

    def get_config(self):
        base_config = super().get_config()  # 부모 클래스의 구성 정보를 가져옵니다.
        return {**base_config, 'units': self.units,
                'activation': keras.activations.serialize(self.activation)}  # 추가된 구성 정보를 포함합니다.


# 다중 입력 및 출력 층 생성
class MyMultilayer(keras.layers.Layer):
    def call(self, X):
        X1, X2 = X
        return [X1 + X2, X1 * X2, X1 / X2]

    def compute_output_shape(self, batch_input_shape):
        b1, b2 = batch_input_shape
        return [b1, b1, b1]  # 맞게 브로드캐스팅 돼야 함.


# 훈련에서만 동작하는 층(ex: Dropout, BatchNormalization)
class MyGaussianNoise(keras.layers.Layer):
    def __init__(self, stddev, **kwargs):
        super().__init__(**kwargs)
        self.stddev = stddev

    def call(self, X, training=None):
        if training:
            noise = tf.random.normal(tf.shape(X), stddev=self.stddev)
            return X + noise
        else:
            return X

    def compute_output_shape(self, batch_input_shape):
        return batch_input_shape


model = keras.models.Sequential([
    MyGaussianNoise(stddev=1.0),
    keras.layers.Dense(30, activation="selu"),
    keras.layers.Dense(1)
])
model.compile(loss="mse", optimizer="nadam")
model.fit(X_train_scaled, y_train, epochs=2,
          validation_data=(X_valid_scaled, y_valid))
model.evaluate(X_test_scaled, y_test)


# 잔차 블록(resudual block) 층(layer) 정의
class ResidualBlock(keras.layers.Layer):
    def __init__(self, n_layers, n_neurons, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [keras.layers.Dense(n_neurons, activation='elu', kernel_initializer='he_normal')
                       for _ in range(n_layers)]

    def call(self, inputs):
        Z = inputs
        for layer in self.hidden:
            Z = layer(Z)
        return inputs + Z


# 모델 정의
class ResidualRegressor(keras.Model):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden1 = keras.layers.Dense(30, activation='elu', kernel_initializer='he_normal')
        self.block1 = ResidualBlock(2, 30)
        self.block2 = ResidualBlock(2, 30)
        self.out = keras.layers.Dense(output_dim)

    def call(self, inputs):
        Z = self.hidden1(inputs)
        for _ in range(1 + 3):
            Z = self.block1(Z)
        Z = self.block2(Z)
        return self.out(Z)


model = ResidualRegressor(1)
model.compile(loss="mse", optimizer="nadam")
history = model.fit(X_train_scaled, y_train, epochs=5)
score = model.evaluate(X_test_scaled, y_test)
# y_pred = model.predict(X_test_scaled)
print(score)
model.save("my_custom_model.ckpt")
model = keras.models.load_model("my_custom_model.ckpt")
history = model.fit(X_train_scaled, y_train, epochs=5)


# 재구성 손실(reconstruction loss): 보조 출력에 연결된 손실로 일반화 성능을 향상 시킬 수 있음
class ReconstructingRegressor(keras.Model):
    def __init__(self, outpur_dim, **kwargs):
        super().__init__(**kwargs)
        self.hidden = [keras.layers.Dense(30, activation='selu', kernel_initializer='lecun_normal')
                       for _ in range(5)]
        self.out = keras.layers.Dense(outpur_dim)

    def build(self, batch_input_shape):
        n_inputs = batch_input_shape[-1]
        self.reconstruct = keras.layers.Dense(n_inputs)
        super().build(batch_input_shape)

    def call(self, inputs):
        Z = inputs
        for layer in self.hidden:
            Z = layer(Z)
        reconstruction = self.reconstruct(Z)
        recon_loss = tf.reduce_mean(tf.square(reconstruction - inputs))
        self.add_loss(0.05 * recon_loss)
        return self.out(Z)


# 책에 나와 있는대로 하면 아래의 오류 발생
# InaccessibleTensorError: <tf.Tensor 'mul:0' shape=() dtype=float32> is out of scope and cannot be used here. Use return values, explicit Python locals or TensorFlow collections to access it.
# The tensor <tf.Tensor 'mul:0' shape=() dtype=float32> cannot be accessed from FuncGraph(name=train_function, id=1925012499472), because it was defined in FuncGraph(name=build_graph, id=1925012545696), which is out of scope.

import tensorflow as tf
from tensorflow import keras


class ReconstructingRegressor(keras.Model):
    def __init__(self, output_dim, **kwargs):
        super().__init__(**kwargs)
        # 5개의 Dense 레이어를 정의합니다. 각 레이어는 30개의 유닛을 가지고, 'selu' 활성화 함수와 'lecun_normal' 초기화 방법을 사용합니다.
        self.hidden = [
            keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal")
            for _ in range(5)
        ]
        # 최종 출력 레이어입니다. 출력 차원은 모델의 출력 차원(output_dim)입니다.
        self.out = keras.layers.Dense(output_dim)
        # 재구성 손실을 저장하는 메트릭입니다.
        self.reconstruction_mean = keras.metrics.Mean(name="reconstruction_error")

    def build(self, batch_input_shape):
        n_inputs = batch_input_shape[-1]  # 입력 차원 수를 추출합니다.
        # 입력 차원 수와 동일한 출력 차원을 가진 Dense 레이어를 정의합니다.
        self.reconstruct = keras.layers.Dense(n_inputs)
        # 부모 클래스의 build 메서드를 호출하여 추가적인 초기화 작업을 수행합니다.
        super().build(batch_input_shape)

    def call(self, inputs, training=None):
        Z = inputs
        # 모든 Dense 레이어를 입력에 순차적으로 적용합니다.
        for layer in self.hidden:
            Z = layer(Z)
        # 마지막 Dense 레이어를 통해 입력의 재구성을 수행합니다.
        reconstruction = self.reconstruct(Z)
        # 재구성 손실을 계산합니다. 재구성 결과와 입력 간의 평균 제곱 오차를 계산합니다.
        self.recon_loss = 0.05 * tf.reduce_mean(tf.square(reconstruction - inputs))

        # 모델이 학습 중인 경우, 재구성 손실을 메트릭에 추가합니다.
        if training:
            result = self.reconstruction_mean(self.recon_loss)
            self.add_metric(result)

        # 최종 출력 레이어를 적용하여 예측 결과를 반환합니다.
        return self.out(Z)

    def train_step(self, data):
        x, y = data  # 훈련 데이터에서 입력 x와 타겟 y를 추출합니다.

        with tf.GradientTape() as tape:
            y_pred = self(x)  # 모델을 사용하여 예측값을 계산합니다.
            # 손실을 계산합니다. 재구성 손실을 정규화 손실로 추가합니다.
            loss = self.compiled_loss(y, y_pred, regularization_losses=[self.recon_loss])

        # 그래디언트를 계산합니다.
        gradients = tape.gradient(loss, self.trainable_variables)
        # 그래디언트를 적용하여 모델의 가중치를 업데이트합니다.
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # 현재 배치에 대한 메트릭 결과를 딕셔너리 형태로 반환합니다.
        return {m.name: m.result() for m in self.metrics}


model = ReconstructingRegressor(1)
model.compile(loss="mse", optimizer="nadam")
history = model.fit(X_train_scaled, y_train, epochs=2)
y_pred = model.predict(X_test_scaled)


def f(w1, w2):
    return 3 * w1 ** 2 + 2 * w1 * w2


w1, w2 = 5, 3
eps = 1e-6
print((f(w1 + eps, w2) - f(w1, w2)) / eps)
print((f(w1, w2 + eps) - f(w1, w2)) / eps)
w1, w2 = tf.Variable(5.), tf.Variable(3.)
with tf.GradientTape() as tape:
    # 자동 미분
    z = f(w1, w2)
gradients = tape.gradient(z, [w1, w2])
# tape.gradient(): 호출 즉시 tape 자동 제거(ex: list.pop)
print(gradients)
with tf.GradientTape(persistent=True) as tape:
    # 자동 제거 해제
    z = f(w1, w2)

dz_dw1 = tape.gradient(z, w1)
dz_dw2 = tape.gradient(z, w2)
print(dz_dw1, dz_dw2)
print(tape.gradient(z, w1), tape.gradient(z, w2))
del tape
c1, c2 = tf.constant(5.), tf.constant(3.)
with tf.GradientTape() as tape:
    z = f(c1, c2)
gradients = tape.gradient(z, [c1, c2])
print(gradients)  # Variable이 아닌 constant는 None 반환

with tf.GradientTape() as tape:
    tape.watch(c1)
    # 연산 강제
    tape.watch(c2)
    z = f(c1, c2)
gradients = tape.gradient(z, [c1, c2])
print(gradients)


# 역전파 미이행
def f(w1, w2):
    return 3 * w1 ** 2 + tf.stop_gradient(2 * w1 * w2)


with tf.GradientTape() as tape:
    z = f(w1, w2)

gradients = tape.gradient(z, [w1, w2])
print(gradients)
# 부동소수점 정밀도 오류로 인한 무한 나누기 무한으로 nan 반환
x = tf.Variable([100.])
with tf.GradientTape() as tape:
    z = my_softplus(x)

gradients = tape.gradient(z, [x])
print(gradients)


@tf.custom_gradient  # 사용자 정의 연산 기울기 지정할 때 사용
def my_better_softplus(z):
    exp = tf.exp(z)

    def my_softplus_gradients(grad):
        return grad / (1 + 1 / exp)

    return tf.math.log(exp + 1), my_softplus_gradients


x = tf.Variable([10.])
with tf.GradientTape() as tape:
    z = my_better_softplus(x)

z, tape.gradient(z, [x])
# L2 정규화(가중치 감소) 설정
l2_reg = keras.regularizers.l2(0.05)

# Sequential 모델 정의
model = keras.models.Sequential([
    keras.layers.Dense(30, activation='elu', kernel_initializer='he_normal',
                       kernel_regularizer=l2_reg),
    keras.layers.Dense(1, kernel_regularizer=l2_reg)
])


# 데이터 배치 추출 함수 정의
def random_batch(X, y, batch_size=32):
    idx = np.random.randint(len(X), size=batch_size)  # 데이터에서 랜덤 인덱스 생성
    return X[idx], y[idx]  # 선택된 데이터와 레이블 반환


# 학습 상태를 출력하는 함수 정의
def print_status_bar(iteration, total, loss, metrics=None):
    # 손실과 메트릭을 포맷하여 문자열로 변환
    metrics = ' - '.join(['{}: {:.4f}'.format(m.name, m.result())
                          for m in [loss] + (metrics or [])])
    # 출력 문자열 구성, 진행 중일 때는 줄바꿈 없이 출력
    end = '' if iteration < total else '\n'
    print('\r{}/{} - '.format(iteration, total) + metrics, end=end)


# L2 정규화(가중치 감소) 설정
l2_reg = keras.regularizers.l2(0.05)

# Sequential 모델 정의
model = keras.models.Sequential([
    keras.layers.Dense(30, activation='elu', kernel_initializer='he_normal',
                       kernel_regularizer=l2_reg),
    keras.layers.Dense(1, kernel_regularizer=l2_reg)
])


# 데이터 배치 추출 함수 정의
def random_batch(X, y, batch_size=32):
    idx = np.random.randint(len(X), size=batch_size)  # 데이터에서 랜덤 인덱스 생성
    return X[idx], y[idx]  # 선택된 데이터와 레이블 반환


# 학습 상태를 출력하는 함수 정의
def print_status_bar(iteration, total, loss, metrics=None):
    # 손실과 메트릭을 포맷하여 문자열로 변환
    metrics = ' - '.join(['{}: {:.4f}'.format(m.name, m.result())
                          for m in [loss] + (metrics or [])])
    # 출력 문자열 구성, 진행 중일 때는 줄바꿈 없이 출력
    end = '' if iteration < total else '\n'
    print('\r{}/{} - '.format(iteration, total) + metrics, end=end)


# 하이퍼파라미터 및 설정 정의
n_epochs = 5
batch_size = 32
n_steps = len(X_train) // batch_size  # 한 에포크에서의 배치 수 계산
optimizer = keras.optimizers.Nadam(learning_rate=0.01)  # 옵티마이저 설정
loss_fn = keras.losses.mean_squared_error  # 손실 함수 설정
mean_loss = keras.metrics.Mean()  # 평균 손실 메트릭 초기화
metrics = [keras.metrics.MeanAbsoluteError()]  # 추가 메트릭 설정

# 학습 루프 시작
for epoch in range(1, n_epochs + 1):
    print('epoch {}/{}'.format(epoch, n_epochs))  # 현재 에포크 출력
    for step in range(1, n_steps + 1):
        # 배치 데이터 추출
        X_batch, y_batch = random_batch(X_train_scaled, y_train)

        # GradientTape를 사용한 기울기 계산 및 업데이트
        with tf.GradientTape() as tape:
            y_pred = model(X_batch, training=True)  # 모델을 통한 예측
            main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))  # 주요 손실 계산
            loss = tf.add_n([main_loss] + model.losses)  # 총 손실 (정규화 손실 포함)

        gradients = tape.gradient(loss, model.trainable_variables)  # 기울기 계산
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))  # 가중치 업데이트
        mean_loss(loss)  # 평균 손실 업데이트
        for metric in metrics:
            metric(y_batch, y_pred)  # 추가 메트릭 업데이트

        # 학습 상태 출력
        print_status_bar(step * batch_size, len(y_train), mean_loss, metrics)

    # 에포크 끝난 후 상태 출력
    print_status_bar(len(y_train), len(y_train), mean_loss, metrics)

    # 메트릭 상태 리셋
    for metric in [mean_loss] + metrics:
        metric.reset_states()


def cube(x):
    return x ** 3


print(cube(2))
print(cube(tf.constant(2.0)))

tf_cube = tf.function(cube)  # 텐서플로 함수로 변환(텐서 객체로 반환)
print(cube)
print(tf_cube)
print(tf_cube(2))
print(tf_cube(tf.constant(2.0)))


@tf.function  # 텐서플로 함수로 선언하는 데코레이터
def tf_cube2(x):
    return x ** 3


print(tf_cube2(2))
print(tf_cube2.python_function(2))  # 파이썬 함수로 사용하는 메서드
# keras는 자동으로 텐서플로 함수로 변환하지만 dynamic=True를 주거나 compile 시 run_eagerly=True 지정하면 불변환

# 텐서플로 함수의 소스코드 출력
print(tf.autograph.to_code(cube))
print(tf.autograph.to_code(tf_cube.python_function))


## 연습문제
class MyLayerNormalization(keras.layers.Layer):
    def __init__(self, eps=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def build(self, batch_input_shape):
        self.alpha = self.add_weight(
            name="alpha", shape=batch_input_shape[-1:],
            initializer="ones")
        self.beta = self.add_weight(
            name="beta", shape=batch_input_shape[-1:],
            initializer="zeros")
        super().build(batch_input_shape)  # 반드시 끝에 와야 합니다

    def call(self, X):
        mean, variance = tf.nn.moments(X, axes=-1, keepdims=True)
        return self.alpha * (X - mean) / tf.sqrt(variance + self.eps) + self.beta

    def compute_output_shape(self, batch_input_shape):
        return batch_input_shape

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "eps": self.eps}


X = X_train.astype(np.float32)

custom_layer_norm = MyLayerNormalization()
keras_layer_norm = keras.layers.LayerNormalization()

tf.reduce_mean(keras.losses.mean_absolute_error(
    keras_layer_norm(X), custom_layer_norm(X)))
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train_full = X_train_full.astype(np.float32) / 255.
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test.astype(np.float32) / 255.
keras.backend.clear_session()
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(1024, activation="relu", kernel_initializer='he_normal'),
    keras.layers.Dense(512, activation="relu", kernel_initializer='he_normal'),
    keras.layers.Dense(256, activation="relu", kernel_initializer='he_normal'),
    keras.layers.Dense(128, activation="relu", kernel_initializer='he_normal'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(64, activation="relu", kernel_initializer='he_normal'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(32, activation="relu", kernel_initializer='he_normal'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(10, activation="softmax"),
])


# 데이터 배치 추출 함수 정의
def random_batch(X, y, batch_size=32):
    idx = np.random.randint(len(X), size=batch_size)  # 데이터에서 랜덤 인덱스 생성
    return X[idx], y[idx]  # 선택된 데이터와 레이블 반환


# 학습 상태를 출력하는 함수 정의
def print_status_bar(iteration, total, loss, metrics=None):
    # 손실과 메트릭을 포맷하여 문자열로 변환
    metrics = ' - '.join(['{}: {:.4f}'.format(m.name, m.result())
                          for m in [loss] + (metrics or [])])
    # 출력 문자열 구성, 진행 중일 때는 줄바꿈 없이 출력
    end = '' if iteration < total else '\n'
    print('\r{}/{} - '.format(iteration, total) + metrics, end=end)


# 하이퍼파라미터 및 설정 정의
n_epochs = 5
batch_size = 32
n_steps = len(X_train) // batch_size  # 한 에포크에서의 배치 수 계산
optimizer = keras.optimizers.Nadam(learning_rate=0.01)  # 옵티마이저 설정
loss_fn = keras.losses.sparse_categorical_crossentropy  # 손실 함수 설정
mean_loss = keras.metrics.Mean()  # 평균 손실 메트릭 초기화
metrics = [keras.metrics.SparseCategoricalAccuracy()]  # 추가 메트릭 설정

# 학습 루프 시작
for epoch in range(1, n_epochs + 1):
    print('epoch {}/{}'.format(epoch, n_epochs))  # 현재 에포크 출력
    for step in range(1, n_steps + 1):
        # 배치 데이터 추출
        X_batch, y_batch = random_batch(X_train, y_train)

        # GradientTape를 사용한 기울기 계산 및 업데이트
        with tf.GradientTape() as tape:
            # y_pred = np.argmax(model(X_batch, training=True), axis=1).astype(np.float32)  # 모델을 통한 예측
            y_pred = model(X_batch, training=True)  # 모델을 통한 예측
            main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))  # 주요 손실 계산
            loss = tf.add_n([main_loss] + model.losses)  # 총 손실 (정규화 손실 포함)

        gradients = tape.gradient(loss, model.trainable_variables)  # 기울기 계산
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))  # 가중치 업데이트
        mean_loss(loss)  # 평균 손실 업데이트
        status = {"loss": mean_loss.result().numpy()}
        for metric in metrics:
            metric(y_batch, y_pred)  # 추가 메트릭 업데이트
            status[metric.name] = metric.result().numpy()

        # 학습 상태 출력
        print_status_bar(step * batch_size, len(y_train), mean_loss, metrics)

    # 에포크 끝난 후 상태 출력
    print_status_bar(len(y_train), len(y_train), mean_loss, metrics)
    y_pred = model(X_valid)
    status["val_loss"] = np.mean(loss_fn(y_valid, y_pred))
    status["val_accuracy"] = np.mean(keras.metrics.sparse_categorical_accuracy(
        tf.constant(y_valid, dtype=np.float32), y_pred))

    # 메트릭 상태 리셋
    for metric in [mean_loss] + metrics:
        metric.reset_states()
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax"),
])

n_epochs = 5
batch_size = 1024
n_steps = len(X_train) // batch_size
optimizer = keras.optimizers.Nadam(learning_rate=0.01)
loss_fn = keras.losses.sparse_categorical_crossentropy
mean_loss = keras.metrics.Mean()
metrics = [keras.metrics.SparseCategoricalAccuracy()]
from tqdm import trange

with trange(1, n_epochs + 1, desc="All epochs", position=1) as epochs:
    for epoch in epochs:
        with trange(1, n_steps + 1, desc="Epoch {}/{}".format(epoch, n_epochs), leave=False, position=0) as steps:
            for step in steps:
                X_batch, y_batch = random_batch(X_train, y_train)
                with tf.GradientTape() as tape:
                    y_pred = model(X_batch)
                    main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
                    loss = tf.add_n([main_loss] + model.losses)
                gradients = tape.gradient(loss, model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model.trainable_variables))
                for variable in model.variables:
                    if variable.constraint is not None:
                        variable.assign(variable.constraint(variable))
                status = {}
                mean_loss(loss)
                status["loss"] = mean_loss.result().numpy()
                for metric in metrics:
                    metric(y_batch, y_pred)
                    status[metric.name] = metric.result().numpy()
                steps.set_postfix(status)
            y_pred = model(X_valid)
            status["val_loss"] = np.mean(loss_fn(y_valid, y_pred))
            status["val_accuracy"] = np.mean(keras.metrics.sparse_categorical_accuracy(
                tf.constant(y_valid, dtype=np.float32), y_pred))
            steps.set_postfix(status)
        for metric in [mean_loss] + metrics:
            metric.reset_states()
lower_layers = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(100, activation="relu"),
])
upper_layers = keras.models.Sequential([
    keras.layers.Dense(10, activation="softmax"),
])
model = keras.models.Sequential([
    lower_layers, upper_layers
])
lower_optimizer = keras.optimizers.SGD(learning_rate=1e-4)
upper_optimizer = keras.optimizers.Nadam(learning_rate=1e-3)
with trange(1, n_epochs + 1, desc="All epochs", position=1) as epochs:
    for epoch in epochs:
        with trange(1, n_steps + 1, desc="Epoch {}/{}".format(epoch, n_epochs), leave=False, position=0) as steps:
            for step in steps:
                X_batch, y_batch = random_batch(X_train, y_train)
                with tf.GradientTape(persistent=True) as tape:
                    y_pred = model(X_batch)
                    main_loss = tf.reduce_mean(loss_fn(y_batch, y_pred))
                    loss = tf.add_n([main_loss] + model.losses)
                for layers, optimizer in ((lower_layers, lower_optimizer),
                                          (upper_layers, upper_optimizer)):
                    gradients = tape.gradient(loss, layers.trainable_variables)
                    optimizer.apply_gradients(zip(gradients, layers.trainable_variables))
                del tape
                for variable in model.variables:
                    if variable.constraint is not None:
                        variable.assign(variable.constraint(variable))
                status = {}
                mean_loss(loss)
                status["loss"] = mean_loss.result().numpy()
                for metric in metrics:
                    metric(y_batch, y_pred)
                    status[metric.name] = metric.result().numpy()
                steps.set_postfix(status)
            y_pred = model(X_valid)
            status["val_loss"] = np.mean(loss_fn(y_valid, y_pred))
            status["val_accuracy"] = np.mean(keras.metrics.sparse_categorical_accuracy(
                tf.constant(y_valid, dtype=np.float32), y_pred))
            steps.set_postfix(status)
        for metric in [mean_loss] + metrics:
            metric.reset_states()