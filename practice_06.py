# 13장
import tensorflow as tf

# TensorFlow의 로깅 수준을 설정
tf.get_logger().setLevel('ERROR')
import tensorflow as tf
import numpy as np

X = tf.range(10)
dataset = tf.data.Dataset.from_tensor_slices(X)  # 각 원소를 아이템으로 변환
print(dataset)
for item in dataset:
    print(item)
print('=' * 10)

dataset2 = dataset.repeat(3).batch(7)  # 3번 반복, 배치 크기 7
for item in dataset2:
    print(item)
print('=' * 10)

dataset2 = dataset.map(lambda x: x * 2)  # 데이터에 함수 적용
dataset2 = dataset2.repeat(3).batch(7, drop_remainder=True)  # 배치 크기보다 작은 데이터 버림
for item in dataset2:
    print(item)
print('=' * 10)

dataset2 = dataset2.apply(tf.data.experimental.unbatch())  # 배치 해제(배치 크기 1)
for item in dataset2:
    print(item)
print('=' * 10)

dataset2 = dataset2.filter(lambda x: x < 10)  # 필터
for item in dataset2:
    print(item)
print('=' * 10)

for item in dataset2.take(3):  # 인자 값의 길이만큼만 반환.(One-based indexing)
    print(item)
dataset = tf.data.Dataset.range(10).repeat(3)
dataset = dataset.shuffle(buffer_size=5, seed=42).batch(7)
# buffer_size만큼 순서대로 추출 후 임의값 1개 추출, 다음 값으로 buffer 채움. 반복.
for item in dataset:
    print(item)
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

housing = fetch_california_housing()
X_train_full, X_test, y_train_full, y_test = train_test_split(
    housing.data, housing.target.reshape(-1, 1), random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_train_full, y_train_full, random_state=42)
import csv, os
from glob import glob

os.makedirs('dataset_hosing', exist_ok=True)
header_cols = housing.feature_names + ["MedianHouseValue"]


def make_data_files(data_using_type: str, X: np.array, y: np.array, split_nums: int = 10) -> list:
    for idx, data in enumerate(np.array_split(np.column_stack((X, y)), split_nums)):
        with open(f'dataset_hosing/{data_using_type}_{idx:0>2}.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerows(np.row_stack((header_cols, data)))

    return glob(f'dataset_hosing/{data_using_type}*.csv')


train_filepaths = make_data_files("train", X_train, y_train, 20)
valid_filepaths = make_data_files("valid", X_valid, y_valid)
test_filepaths = make_data_files("test", X_test, y_test)
filepath_dataset = tf.data.Dataset.list_files(train_filepaths, seed=42)
# tf.data.Dataset.list_files: 파일들의 경로 리스트로 dataset 생성

n_readers = 5
dataset = filepath_dataset.interleave(  # tf.data.Dataset.interleave: dataset에 함수를 적용하여 병렬로 데이터 읽음.
    lambda filepath: tf.data.TextLineDataset(filepath).skip(1),  # 첫 행 스킵(컬럼명)
    cycle_length=n_readers  # 병렬로 읽을 dataset 수
)
# tf.data.TextLineDataset: 텍스트 파일 데이터를 한 줄씩 읽음.

for line in dataset.take(5):
    print(line.numpy())
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train)
X_mean = scaler.mean_
X_std = scaler.scale_
n_inputs = 8


@tf.function
def preprocess(line):
    defs = [0.] * n_inputs + [tf.constant([], dtype=tf.float32)]
    # defs=기본값 지정: 각 input에 0. 기본값 + y는 tf.float32 타입의 빈 배열(실수지만 기본값 없음: 값 누락 시 오류 발생)
    fields = tf.io.decode_csv(line, record_defaults=defs)
    # csv 형식의 문자열을 단일 값인 0차원 스칼라 텐서로 변환
    x = tf.stack(fields[:-1])
    y = tf.stack(fields[-1:])

    return (x - X_mean) / X_std, y


print(preprocess(line.numpy()))


def csv_reader_dataset(filepaths, repeat=1, n_readers=5,
                       n_read_threads=None, shuffle_buffer_size=10000,
                       n_parse_threads=5, batch_size=32):
    dataset = tf.data.Dataset.list_files(filepaths).repeat(repeat)
    dataset = dataset.interleave(
        lambda filepath: tf.data.TextLineDataset(filepath).skip(1),
        cycle_length=n_readers, num_parallel_calls=n_read_threads)
    dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(preprocess, num_parallel_calls=n_parse_threads)

    return dataset.batch(batch_size).prefetch(1)
    # 항상 1개 배치 미리 준비


train_set = csv_reader_dataset(train_filepaths, repeat=None)
valid_set = csv_reader_dataset(valid_filepaths)
test_set = csv_reader_dataset(test_filepaths)
from tensorflow import keras

model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=X_train.shape[1:]),
    keras.layers.Dense(1),
])
model.compile(loss="mse", optimizer=keras.optimizers.SGD(learning_rate=1e-3))

batch_size = 128
model.fit(train_set, steps_per_epoch=len(X_train) // batch_size, epochs=10,
          validation_data=valid_set)
model.evaluate(test_set, steps=len(X_test) // batch_size)
# `Dataset` 클래스에 있는 메서드의 간략한 설명입니다:
for m in dir(tf.data.Dataset):
    if not (m.startswith("_") or m.endswith("_")):
        func = getattr(tf.data.Dataset, m)
        if hasattr(func, "__doc__"):
            print("● {:21s}{}".format(m + "()", func.__doc__.split("\n")[0]))

with tf.io.TFRecordWriter('my_data.tfrecord') as f:
    f.write(b'This is the first record')
    f.write(b'and this is the second record')
    # 현재는 바이트 문자열이 아닌 그냥 문자열로 입력해도 알아서 변환됨.

filepaths = ['my_data.tfrecord']
dataset = tf.data.TFRecordDataset(filepaths)
for item in dataset:
    print(item)
options = tf.io.TFRecordOptions(compression_type="GZIP")
with tf.io.TFRecordWriter("my_compressed.tfrecord", options) as f:
    f.write(b"This is the first record")
    f.write(b"And this is the second record")

dataset = tf.data.TFRecordDataset(["my_compressed.tfrecord"],
                                  compression_type="GZIP")
for item in dataset:
    print(item)
from tensorflow.train import BytesList, FloatList, Int64List
from tensorflow.train import Feature, Features, Example

person_example = Example(
    features=Features(
        feature={
            "name": Feature(bytes_list=BytesList(value=[b"Alice"])),
            "id": Feature(int64_list=Int64List(value=[123])),
            "emails": Feature(bytes_list=BytesList(value=[b"a@b.com",
                                                          b"c@d.com"]))
        }
    )
)

with tf.io.TFRecordWriter("my_contacts.tfrecord") as f:
    f.write(person_example.SerializeToString())
feature_description = {
    "name": tf.io.FixedLenFeature([], tf.string, default_value=""),
    "id": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    "emails": tf.io.VarLenFeature(tf.string)
}

for serialized_example in tf.data.TFRecordDataset(["my_contacts.tfrecord"]):
    parsed_example = tf.io.parse_single_example(serialized_example, feature_description)
    # tf.io.parse_single_example: 하나씩 파싱, 딕셔너리로 반환
print(tf.sparse.to_dense(parsed_example["emails"], default_value=b""))
# tf.sparse.to_dense: 희소 텐서를 밀집 텐서로 변환
print(parsed_example["emails"].values)
print(parsed_example)

for serialized_example in tf.data.TFRecordDataset(["my_contacts.tfrecord"]).repeat(13).batch(10):
    parsed_example = tf.io.parse_example(serialized_example, feature_description)
    # tf.io.parse_example: 배치 단위 파싱, 딕셔너리로 반환
print(parsed_example["name"].numpy())  # SparseTensor가 아니라 .numpy() 사용
print(parsed_example)  # 반복문에서 할당이기에 마지막 배치를 할당하여 호출


from sklearn.datasets import load_sample_images
import matplotlib.pyplot as plt

img = load_sample_images()["images"][0]
plt.imshow(img)
plt.axis("off")
plt.title("Original Image")
plt.show(block=False)
plt.pause(2)
plt.close()
data = tf.io.encode_jpeg(img)
example_with_image = Example(features=Features(feature={
    "image": Feature(bytes_list=BytesList(value=[data.numpy()]))}))
serialized_example = example_with_image.SerializeToString()
# then save to TFRecord

feature_description = {"image": tf.io.VarLenFeature(tf.string)}
example_with_image = tf.io.parse_single_example(serialized_example, feature_description)
decoded_img = tf.io.decode_jpeg(example_with_image["image"].values[0])

# tf.io.decode_image(): BMP, GIF, JPEG, PNG 포맷을 지원.
decoded_img = tf.io.decode_image(example_with_image["image"].values[0])
plt.imshow(decoded_img)
plt.title("Decoded Image")
plt.axis("off")
plt.show(block=False)
plt.pause(2)
plt.close()
from tensorflow.train import FeatureList, FeatureLists, SequenceExample

context = Features(feature={
    "author_id": Feature(int64_list=Int64List(value=[123])),
    "title": Feature(bytes_list=BytesList(value=[b"A", b"desert", b"place", b"."])),
    "pub_date": Feature(int64_list=Int64List(value=[1623, 12, 25]))
})

content = [["When", "shall", "we", "three", "meet", "again", "?"],
           ["In", "thunder", ",", "lightning", ",", "or", "in", "rain", "?"]]
comments = [["When", "the", "hurlyburly", "'s", "done", "."],
            ["When", "the", "battle", "'s", "lost", "and", "won", "."]]


def words_to_feature(words):
    return Feature(bytes_list=BytesList(value=[word.encode("utf-8")
                                               for word in words]))


content_features = [words_to_feature(sentence) for sentence in content]
comments_features = [words_to_feature(comment) for comment in comments]

sequence_example = SequenceExample(
    context=context,
    feature_lists=FeatureLists(feature_list={
        "content": FeatureList(feature=content_features),
        "comments": FeatureList(feature=comments_features)
    }))

print(type(sequence_example))
serialized_sequence_example = sequence_example.SerializeToString()
context_feature_descriptions = {
    "author_id": tf.io.FixedLenFeature([], tf.int64, default_value=0),
    "title": tf.io.VarLenFeature(tf.string),
    "pub_date": tf.io.FixedLenFeature([3], tf.int64, default_value=[0, 0, 0]),
}
sequence_feature_descriptions = {
    "content": tf.io.VarLenFeature(tf.string),
    "comments": tf.io.VarLenFeature(tf.string),
}
parsed_context, parsed_feature_lists = tf.io.parse_single_sequence_example(
    serialized_sequence_example, context_feature_descriptions, sequence_feature_descriptions)
parsed_content = tf.RaggedTensor.from_sparse(parsed_feature_lists["content"])
print(parsed_content)
print(parsed_feature_lists["content"].values)
means = np.mean(X_train, axis=0, keepdims=True)
stds = np.std(X_train, axis=0, keepdims=True)
eps = keras.backend.epsilon()

model = keras.models.Sequential([
    keras.layers.Lambda(lambda inputs: (inputs - means) / (stds + eps)),
    keras.layers.Dense(30, activation="relu"),
    keras.layers.Dense(1),
])


class Standardization(keras.layers.Layer):
    def adapt(self, data_sample):
        self.means_ = np.mean(data_sample, axis=0, keepdims=True)
        self.stds_ = np.std(data_sample, axis=0, keepdims=True)

    def call(self, inputs):
        return (inputs - self.means_) / self.stds_ + keras.backend.epsilon()


std_layer = Standardization()
# std_layer.adapt(data_sample)
keras.layers.Normalization()
vocab = ["<1H OCEAN", "INLAND", "NEAR_OCCEAN", 'NEAR BAY', 'ICELAND']
indices = tf.range(len(vocab), dtype=tf.int64)
table_init = tf.lookup.KeyValueTensorInitializer(vocab, indices)
# 초기화 객체 생성
num_oov_buckets = 2  # oov(out-of-vocabulary) 갯수 지정
# 어휘 사전에 없는 값이 들어올 경우 oov에 할당
table = tf.lookup.StaticVocabularyTable(table_init, num_oov_buckets)

categories = tf.constant(["NEAR BAY", "DESERT", "INLAND", "INLAND"])
cat_indeices = table.lookup(categories)
print(cat_indeices)

cat_one_hot = tf.one_hot(cat_indeices, depth=len(vocab) + num_oov_buckets)
print(cat_one_hot)
embedding_dim = 2
embed_init = tf.random.uniform([len(vocab) + num_oov_buckets, embedding_dim])
embedding_matrix = tf.Variable(embed_init)
print(embedding_matrix)

categories = tf.constant(["NEAR BAY", "DESERT", "INLAND", "INLAND"])
cat_indeices = table.lookup(categories)
print(cat_indeices)

cat_embed = tf.nn.embedding_lookup(embedding_matrix, cat_indeices)
print(cat_embed)
regular_inputs = keras.layers.Input(shape=[8])
categories = keras.layers.Input(shape=[], dtype=tf.string)
cat_indeices = keras.layers.Lambda(lambda cats: table.lookup(cats), output_shape=(1,))(categories)
cat_embed = keras.layers.Embedding(input_dim=6, output_dim=2)(cat_indeices)
encoded_inputs = keras.layers.concatenate([regular_inputs, cat_embed])
outputs = keras.layers.Dense(1)(encoded_inputs)
model = keras.models.Model(inputs=[regular_inputs, categories], outputs=[outputs])
# 전처리층을 포함한 모델
norm_layer = tf.keras.layers.Normalization()
model = tf.keras.models.Sequential([
    norm_layer,
    tf.keras.layers.Dense(1)
])
model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(learning_rate=2e-3))

norm_layer.adapt(X_train)  # 모든 특성의 평균과 분산을 계산합니다.
# 모델에서 정규화
model.fit(X_train, y_train, validation_data=(X_valid, y_valid), epochs=5)

# 전처리를 별도로 하는 모델
norm_layer = tf.keras.layers.Normalization()
norm_layer.adapt(X_train)
# input 전처리
X_train_scaled = norm_layer(X_train)
X_valid_scaled = norm_layer(X_valid)
# 전처리층 없이 모델 구성
model = tf.keras.models.Sequential([tf.keras.layers.Dense(1)])
model.compile(loss="mse", optimizer=tf.keras.optimizers.SGD(learning_rate=2e-3))
model.fit(X_train_scaled, y_train, epochs=5,
          validation_data=(X_valid_scaled, y_valid));
age = tf.constant([[10.], [93.], [57.], [18.], [37.], [5.]])
discretize_layer = tf.keras.layers.Discretization(bin_boundaries=[18., 50.])
age_categories = discretize_layer(age)
print(age_categories)

discretize_layer = tf.keras.layers.Discretization(num_bins=3)
discretize_layer.adapt(age)
age_categories = discretize_layer(age)
print(age_categories)
hashing_layer = tf.keras.layers.Hashing(num_bins=2)
print(hashing_layer([["Paris"], ["Tokyo"], ["Auckland"], ["Montreal"]]))
onehot_layer = tf.keras.layers.CategoryEncoding(num_tokens=3)  # output_mode="multi_hot"(default)
print(onehot_layer(age_categories))

two_age_categories = np.array([[1, 0], [2, 2], [2, 0], [1, 0]])
print(onehot_layer(two_age_categories))
onehot_layer = tf.keras.layers.CategoryEncoding(num_tokens=3, output_mode="count")
print(onehot_layer(two_age_categories))

onehot_layer = tf.keras.layers.CategoryEncoding(num_tokens=3, output_mode="one_hot")
print([onehot_layer(cat) for cat in tf.transpose(two_age_categories)])
cities = ["Auckland", "Paris", "Paris", "San Francisco"]
str_lookup_layer = tf.keras.layers.StringLookup()
str_lookup_layer.adapt(cities)
print(str_lookup_layer([["Paris"], ["Auckland"], ["Auckland"], ["Montreal"]]))

str_lookup_layer = tf.keras.layers.StringLookup(num_oov_indices=5)
str_lookup_layer.adapt(cities)
print(str_lookup_layer([["Paris"], ["Auckland"], ["Foo"], ["Bar"], ["Baz"]]))

str_lookup_layer = tf.keras.layers.StringLookup(output_mode="one_hot")
str_lookup_layer.adapt(cities)
print(str_lookup_layer([["Paris"], ["Auckland"], ["Auckland"], ["Montreal"]]))

# 추가 코드 - IntegerLookup 층을 사용한 예제
ids = [123, 456, 789]
int_lookup_layer = tf.keras.layers.IntegerLookup()
int_lookup_layer.adapt(ids)
print(int_lookup_layer([[123], [456], [123], [111]]))
import tensorflow_transform as tft


def preprocess(inputs):  # inputs is a batch of input features
    median_age = inputs["housing_median_age"]
    ocean_proximity = inputs["ocean_proximity"]
    standardized_age = tft.scale_to_z_score(median_age - tft.mean(median_age))
    ocean_proximity_id = tft.compute_and_apply_vocabulary(ocean_proximity)
    return {
        "standardized_median_age": standardized_age,
        "ocean_proximity_id": ocean_proximity_id
    }


import tensorflow_datasets as tfds

datasets = tfds.load(name="mnist")
mnist_train, mnist_test = datasets["train"], datasets["test"]
print(tfds.list_builders())
plt.figure(figsize=(6, 3))
mnist_train = mnist_train.repeat(5).batch(32).prefetch(1)
for item in mnist_train:
    images = item["image"]
    labels = item["label"]
    for index in range(5):
        plt.subplot(1, 5, index + 1)
        image = images[index, ..., 0]
        label = labels[index].numpy()
        plt.imshow(image, cmap="binary")
        plt.title(label)
        plt.axis("off")
    break  # just showing part of the first batch
datasets = tfds.load(name="mnist")
mnist_train, mnist_test = datasets["train"], datasets["test"]
mnist_train = mnist_train.repeat(5).batch(32)
mnist_train = mnist_train.map(lambda items: (items["image"], items["label"]))
mnist_train = mnist_train.prefetch(1)
for images, labels in mnist_train.take(1):
    print(images.shape)
    print(labels.numpy())
keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)
datasets = tfds.load(name="mnist", batch_size=32, as_supervised=True)
mnist_train = datasets["train"].repeat().prefetch(1)
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28, 1]),
    keras.layers.Lambda(lambda images: tf.cast(images, tf.float32)),
    keras.layers.Dense(10, activation="softmax")])
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=keras.optimizers.SGD(learning_rate=1e-3),
              metrics=["accuracy"])
model.fit(mnist_train, steps_per_epoch=60000 // 32, epochs=5);
(X_train_full, y_train_full), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
X_valid, X_train = X_train_full[:5000], X_train_full[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]

train_set = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_set = train_set.shuffle(len(X_train))
valid_set = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
valid_set = valid_set.shuffle(len(X_valid))
test_set = tf.data.Dataset.from_tensor_slices((X_test, y_test))
# test_set = test_set.shuffle(len(X_test))
for X, y in train_set:
    print(type(X), X.shape)
    print(y)
    break


def create_example(image, label):
    image_data = tf.io.serialize_tensor(image)
    # image_data = tf.io.encode_jpeg(image[..., np.newaxis])
    return Example(
        features=Features(
            feature={
                "image": Feature(bytes_list=BytesList(value=[image_data.numpy()])),  # int array를 byte로 받음
                "label": Feature(int64_list=Int64List(value=[label])),
            }))


for image, label in valid_set.take(1):
    print(create_example(image, label))
import os
from contextlib import ExitStack

# 여러 컨텍스트 관리자(예: `with` 문에서 사용하는 객체)를 동적으로 관리하거나,
# 필요에 따라 컨텍스트 관리자를 나중에 추가하고 제거할 수 있게 해주는 클래스

os.makedirs('./my_mnist', exist_ok=True)


def write_tfrecords(name, dataset, n_shards=10):
    paths = ["my_mnist/{}.tfrecord-{:05d}-of-{:05d}".format(name, index, n_shards)
             for index in range(n_shards)]
    # 저장할 파일명 지정
    with ExitStack() as stack:
        writers = [stack.enter_context(tf.io.TFRecordWriter(path))
                   for path in paths]
        # 각 파일의 writer 생성
        for index, (image, label) in dataset.enumerate():
            shard = index % n_shards
            example = create_example(image, label)
            writers[shard].write(example.SerializeToString())
            # 객체를 string으로 저장
    return paths


train_filepaths = write_tfrecords("my_fashion_mnist.train", train_set)
valid_filepaths = write_tfrecords("my_fashion_mnist.valid", valid_set)
test_filepaths = write_tfrecords("my_fashion_mnist.test", test_set)


def preprocess(tfrecord):
    feature_descriptions = {
        "image": tf.io.FixedLenFeature([], tf.string, default_value=""),
        "label": tf.io.FixedLenFeature([], tf.int64, default_value=-1)
    }
    example = tf.io.parse_single_example(tfrecord, feature_descriptions)
    image = tf.io.parse_tensor(example["image"], out_type=tf.uint8)
    image = tf.reshape(image, shape=[28, 28])

    return image, example["label"]


def mnist_dataset(filepaths, n_read_threads=5, shuffle_buffer_size=None,
                  n_parse_threads=5, batch_size=32, cache=True):
    dataset = tf.data.TFRecordDataset(filepaths,
                                      num_parallel_reads=n_read_threads)
    if cache:
        dataset = dataset.cache()
    if shuffle_buffer_size:
        dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.map(preprocess, num_parallel_calls=n_parse_threads)
    dataset = dataset.batch(batch_size)

    return dataset.prefetch(1)


train_set = mnist_dataset(train_filepaths, shuffle_buffer_size=60000)
valid_set = mnist_dataset(valid_filepaths)
test_set = mnist_dataset(test_filepaths)
for X, y in train_set.take(1):
    print(X.shape)
    for i in range(5):
        plt.subplot(1, 5, i + 1)
        plt.imshow(X[i].numpy(), cmap="binary")
        plt.axis("off")
        plt.title(str(y[i].numpy()))
standardization = keras.layers.Normalization(input_shape=[28, 28])

sample_image_batches = train_set.take(100).map(lambda image, label: image)
sample_images = np.concatenate(list(sample_image_batches.as_numpy_iterator()),
                               axis=0).astype(np.float32)
standardization.adapt(sample_images)

model = keras.models.Sequential([
    standardization,
    keras.layers.Flatten(),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="nadam", metrics=["accuracy"])
from datetime import datetime

logs = os.path.join("my_logs", "run_" + datetime.now().strftime("%Y%m%d_%H%M%S"))

tensorboard_cb = tf.keras.callbacks.TensorBoard(
    log_dir=logs, histogram_freq=1, profile_batch=10)

model.fit(train_set, epochs=5, validation_data=valid_set,
          callbacks=[tensorboard_cb]);
from pathlib import Path

root = "https://ai.stanford.edu/~amaas/data/sentiment/"
filename = "aclImdb_v1.tar.gz"
filepath = tf.keras.utils.get_file(filename, root + filename, extract=True,
                                   cache_dir=".")
path = Path(filepath).with_name("aclImdb")
path


def review_paths(dirpath):
    return [str(path) for path in dirpath.glob("*.txt")]


train_pos = review_paths(path / "train" / "pos")
train_neg = review_paths(path / "train" / "neg")
test_valid_pos = review_paths(path / "test" / "pos")
test_valid_neg = review_paths(path / "test" / "neg")

len(train_pos), len(train_neg), len(test_valid_pos), len(test_valid_neg)
np.random.shuffle(test_valid_pos)
np.random.shuffle(test_valid_neg)

test_pos = test_valid_pos[:5000]
test_neg = test_valid_neg[:5000]
valid_pos = test_valid_pos[5000:]
valid_neg = test_valid_neg[5000:]


def imdb_dataset(filepaths_positive, filepaths_negative):
    reviews = []
    labels = []
    for filepaths, label in ((filepaths_negative, 0), (filepaths_positive, 1)):
        for filepath in filepaths:
            with open(filepath, encoding='utf_8') as review_file:
                reviews.append(review_file.read())
            labels.append(label)
    return tf.data.Dataset.from_tensor_slices(
        (tf.constant(reviews), tf.constant(labels)))


batch_size = 32

train_set = imdb_dataset(train_pos, train_neg).shuffle(25000, seed=42)
train_set = train_set.batch(batch_size).prefetch(1)
valid_set = imdb_dataset(valid_pos, valid_neg).batch(batch_size).prefetch(1)
test_set = imdb_dataset(test_pos, test_neg).batch(batch_size).prefetch(1)
max_tokens = 1000
sample_reviews = train_set.map(lambda review, label: review)
text_vectorization = tf.keras.layers.TextVectorization(
    max_tokens=max_tokens, output_mode="tf_idf")
text_vectorization.adapt(sample_reviews)

text_vectorization.get_vocabulary()[:10]
model = tf.keras.Sequential([
    text_vectorization,
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])
model.compile(loss="binary_crossentropy", optimizer="nadam",
              metrics=["accuracy"])
model.fit(train_set, epochs=5, validation_data=valid_set)


def compute_mean_embedding(inputs):
    not_pad = tf.math.count_nonzero(inputs, axis=-1)
    n_words = tf.math.count_nonzero(not_pad, axis=-1, keepdims=True)
    sqrt_n_words = tf.math.sqrt(tf.cast(n_words, tf.float32))
    return tf.reduce_sum(inputs, axis=1) / sqrt_n_words


another_example = tf.constant([[[1., 2., 3.], [4., 5., 0.], [0., 0., 0.]],
                               [[6., 0., 0.], [0., 0., 0.], [0., 0., 0.]]])
compute_mean_embedding(another_example)
embedding_size = 20

text_vectorization = tf.keras.layers.TextVectorization(
    max_tokens=max_tokens, output_mode="int")
text_vectorization.adapt(sample_reviews)

model = tf.keras.Sequential([
    text_vectorization,
    tf.keras.layers.Embedding(input_dim=max_tokens,
                              output_dim=embedding_size,
                              mask_zero=True),  # <pad> tokens => zero vectors
    tf.keras.layers.Lambda(compute_mean_embedding),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

model.compile(loss="binary_crossentropy", optimizer="nadam",
              metrics=["accuracy"])
model.fit(train_set, epochs=5, validation_data=valid_set);
import tensorflow_datasets as tfds

datasets = tfds.load(name="imdb_reviews")
train_set, test_set = datasets["train"], datasets["test"]

for example in train_set.take(1):
    print(example["text"])
    print(example["label"])