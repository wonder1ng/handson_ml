# 14장
import tensorflow as tf, os
import warnings

warnings.filterwarnings('ignore')

tf.get_logger().setLevel('ERROR')

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
from sklearn.datasets import load_sample_image
import matplotlib.pyplot as plt

china = load_sample_image('china.jpg') / 255
flower = load_sample_image('flower.jpg') / 255
images = np.array([china, flower])
batch_size, height, width, channels = images.shape

filters = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
filters[:, 3, :, 0] = 1  # 수직선
filters[3, :, :, 1] = 1  # 수평선

outputs = tf.nn.conv2d(images, filters, strides=1, padding='SAME')

for image_index in (0, 1):
    plt.figure(figsize=(10, 10))
    for channel in range(channels):
        plt.subplot(3, 2, channel + 1, title=f'channel: {channel}')
        plt.imshow(images[image_index, :, :, channel], cmap='gray')
        plt.axis("off")

    plt.subplot(3, 2, 4, title='channel: mean')
    plt.imshow(images[image_index].mean(axis=2).reshape(-1, 640, 1), cmap='gray')
    plt.axis("off")

    plt.subplot(3, 2, 5, title='output: 0')
    plt.imshow(outputs[image_index, :, :, 0], cmap='gray')
    plt.axis("off")

    plt.subplot(3, 2, 6, title='output: 1')
    plt.imshow(outputs[image_index, :, :, 1], cmap='gray')
    plt.axis("off")

    plt.show(block=False)
    plt.pause(2)
    plt.close()
conv = keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', activation='relu')
print(outputs.shape)

conv2 = keras.layers.Conv2D(filters=2, kernel_size=7, strides=1,
                            padding="SAME", activation="relu", input_shape=outputs.shape)

conv_outputs = conv(images)
conv_outputs2 = conv2(images)
print(conv_outputs.shape, conv_outputs2.shape)

with tf.device('/CPU:0'):
    output = tf.nn.max_pool(images,
                            ksize=(1, 1, 1, 3),
                            strides=(1, 1, 1, 3),
                            padding='VALID')  # 패딩 없음
depth_pool = keras.layers.Lambda(lambda X:
                                 tf.nn.max_pool(X,
                                                ksize=(1, 1, 1, 3),
                                                strides=(1, 1, 1, 3),
                                                padding='valid'))
global_avg_pool = keras.layers.GlobalAvgPool2D()
global_avg_pool = keras.layers.Lambda(lambda X: tf.reduce_mean(X, axis=[1, 2]))

(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]

X_mean = X_train.mean(axis=0, keepdims=True)
X_std = X_train.std(axis=0, keepdims=True) + 1e-7
X_train = (X_train - X_mean) / X_std
X_valid = (X_valid - X_mean) / X_std
X_test = (X_test - X_mean) / X_std

X_train = X_train[..., np.newaxis]
X_valid = X_valid[..., np.newaxis]
X_test = X_test[..., np.newaxis]
model = keras.models.Sequential([
    keras.layers.Conv2D(64, 7, activation='relu', padding='same', input_shape=[28, 28, 1]),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
    keras.layers.Conv2D(128, 3, activation='relu', padding='same'),
    keras.layers.MaxPooling2D(2),
    keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
    keras.layers.Conv2D(256, 3, activation='relu', padding='same'),
    keras.layers.MaxPooling2D(2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam", metrics=["accuracy"])

history = model.fit(X_train, y_train, batch_size=1024, epochs=10, validation_data=(X_valid, y_valid))
score = model.evaluate(X_test, y_test)
X_new = X_test[:10]  # 새로운 이미지처럼 사용합니다
y_pred = model.predict(X_new)
lenet = keras.Sequential([
    keras.layers.Conv2D(filters=6, kernel_size=5, strides=1, padding='same', activation='tanh',
                        input_shape=(28, 28, 1)),
    keras.layers.AvgPool2D(pool_size=2, strides=2),
    keras.layers.Conv2D(16, 5, 1, 'same', activation='tanh'),
    keras.layers.AvgPool2D(2, 2),
    keras.layers.Conv2D(120, 5, 1, 'same', activation='tanh'),
    keras.layers.Flatten(),
    keras.layers.Dense(84, activation='tanh'),
    keras.layers.Dense(10, activation='softmax')
])
lenet.summary()
del lenet
alexnet = keras.Sequential([
    keras.layers.experimental.preprocessing.Resizing(227, 227, input_shape=(28, 28, 3)),
    keras.layers.Conv2D(filters=96, kernel_size=11, strides=4, padding='valid', activation='relu'),
    keras.layers.Lambda(tf.nn.lrn),  # 정규화, 요즘은 BatchNormalization 사용
    keras.layers.MaxPool2D(pool_size=3, strides=2),
    keras.layers.Conv2D(256, 5, 1, 'same', activation='relu'),
    keras.layers.Lambda(tf.nn.local_response_normalization),  # lrn 풀네임
    keras.layers.MaxPool2D(pool_size=3, strides=2),
    keras.layers.Conv2D(384, 3, 1, 'same', activation='relu'),
    keras.layers.Conv2D(384, 3, 1, 'same', activation='relu'),
    keras.layers.Conv2D(256, 3, 1, 'same', activation='relu'),
    keras.layers.MaxPool2D(pool_size=3, strides=2),
    keras.layers.Flatten(),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(4096, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')
])
alexnet.summary()
del alexnet


class ResidualUnit(keras.layers.Layer):
    def __init__(self, filters, strides=1, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.activation = keras.activations.get(activation)
        self.main_layers = [
            keras.layers.Conv2D(filters, 3, strides=strides, padding='same', use_bias=False),
            keras.layers.BatchNormalization(),
            self.activation,
            keras.layers.Conv2D(filters, 3, strides=1, padding='same', use_bias=False),
            keras.layers.BatchNormalization()
        ]

        self.skip_layers = []
        if strides > 1:
            self.skip_layers = [
                keras.layers.Conv2D(filters, 1, strides=strides, padding='same', use_bias=False),
                keras.layers.BatchNormalization()
            ]

    def call(self, inputs):
        Z = inputs
        for layer in self.main_layers:
            Z = layer(Z)
        skip_Z = inputs
        for layer in self.skip_layers:
            skip_Z = layer(skip_Z)

        return self.activation(Z + skip_Z)


model = keras.models.Sequential()
model.add(keras.layers.Conv2D(64, 7, strides=2, input_shape=[224, 224, 3], padding='same', use_bias=False))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D(pool_size=3, strides=2, padding='same'))
prev_filters = 64
for filters in [64] * 3 + [128] * 4 + [256] * 6 + [512] * 3:
    strides = 1 if filters == prev_filters else 2
    model.add(ResidualUnit(filters, strides=strides))
    prev_filters = filters
model.add(keras.layers.GlobalAvgPool2D())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(10, activation='softmax'))
model.summary()

model = keras.applications.resnet50.ResNet50(weights='imagenet')
images_resized = tf.image.resize(images, [224, 224])
inputs = keras.applications.resnet50.preprocess_input(images_resized * 255)
Y_proba = model.predict(inputs)

top_K = keras.applications.resnet50.decode_predictions(Y_proba, top=3)
for image_index in range(len(images)):
    print('이미지 #{}'.format(image_index))
    for class_id, name, y_proba in top_K[image_index]:
        print('  {} - {:12s} {:.2f}%'.format(class_id, name, y_proba * 100))
    print()
import tensorflow_datasets as tfds

dataset, info = tfds.load('tf_flowers', as_supervised=True, with_info=True)
dataset_size = info.splits['train'].num_examples
class_names = info.features['label'].names
n_classes = info.features['label'].num_classes
test_set, valid_set, train_set = tfds.load("tf_flowers",
                                           split=["train[:10%]", "train[10%:25%]", "train[25%:]"],
                                           as_supervised=True)


def preprocess(image, label):
    resized_image = tf.image.resize(image, [224, 224])
    final_image = keras.applications.xception.preprocess_input(resized_image)

    return final_image, label


batch_size = 32
train_set = train_set.shuffle(1000)
train_set = train_set.map(preprocess).batch(batch_size).prefetch(1)
valid_set = valid_set.map(preprocess).batch(batch_size).prefetch(1)
test_set = test_set.map(preprocess).batch(batch_size).prefetch(1)
base_model = keras.applications.xception.Xception(weights='imagenet', include_top=False)
avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
output = keras.layers.Dense(n_classes, activation='softmax')(avg)
model = keras.Model(inputs=base_model.input, outputs=output)

print(len(base_model.trainable_variables), len(model.trainable_variables))
for layer in base_model.layers:
    layer.trainable = False
print(len(base_model.trainable_variables), len(model.trainable_variables))

base_model.trainable = True
print(len(base_model.trainable_variables), len(model.trainable_variables))
base_model.trainable = False
print(len(base_model.trainable_variables), len(model.trainable_variables))
optimizer = keras.optimizers.SGD(learning_rate=0.2, momentum=0.9)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history = model.fit(train_set, epochs=5, validation_data=valid_set)
print(len(base_model.trainable_variables), len(model.trainable_variables))
base_model.trainable = True
print(len(base_model.trainable_variables), len(model.trainable_variables))

optimizer = keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history = model.fit(train_set, epochs=3, validation_data=valid_set)

base_model = keras.applications.xception.Xception(weights="imagenet",
                                                  include_top=False)
avg = keras.layers.GlobalAveragePooling2D()(base_model.output)
class_output = keras.layers.Dense(n_classes, activation="softmax")(avg)
loc_output = keras.layers.Dense(4)(avg)
model = keras.models.Model(inputs=base_model.input,
                           outputs=[class_output, loc_output])
model.compile(loss=["sparse_categorical_crossentropy", "mse"],
              loss_weights=[0.8, 0.2],  # 어떤 것을 중요하게 생각하느냐에 따라
              optimizer=optimizer, metrics=["accuracy"])
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train_full = X_train_full / 255.
X_test = X_test / 255.
X_train, X_valid = X_train_full[:-5000], X_train_full[-5000:]
y_train, y_valid = y_train_full[:-5000], y_train_full[-5000:]

X_train = X_train[..., np.newaxis]
X_valid = X_valid[..., np.newaxis]
X_test = X_test[..., np.newaxis]
model = keras.models.Sequential([
    keras.layers.Conv2D(32, kernel_size=3, padding="same", activation="relu"),
    keras.layers.Conv2D(64, kernel_size=3, padding="same", activation="relu"),
    keras.layers.MaxPool2D(),
    keras.layers.Flatten(),
    keras.layers.Dropout(0.25),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation="softmax")
])
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam",
              metrics=["accuracy"])

model.fit(X_train, y_train, epochs=10, validation_data=(X_valid, y_valid))
model.evaluate(X_test, y_test)