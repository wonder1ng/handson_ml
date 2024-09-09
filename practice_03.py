# 3장
from sklearn.datasets import fetch_openml
import numpy as np

mnist = fetch_openml('mnist_784', version=1, as_frame=False)
print(mnist.keys())
X, y = mnist['data'], mnist['target'].astype(np.uint8)
print(X.shape, y.shape)

import matplotlib as mpl
import matplotlib.pyplot as plt

some_digit = X[0]
some_digit_image = some_digit.reshape(28, 28)

plt.imshow(some_digit_image, cmap='binary')
plt.axis('off')
print(y[0])
plt.show(block=False)
plt.pause(2)
plt.close()
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)


from sklearn.linear_model import SGDClassifier

sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
sgd_clf.predict([some_digit])


from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone  # 모델의 학습 결과는 복제하지 않음

skfolds = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)

for train_index, test_index in skfolds.split(X_train, y_train_5):
    clone_clf = clone(sgd_clf)
    X_train_folds = X_train[train_index]
    y_train_folds = y_train_5[train_index]
    X_test_folds = X_train[test_index]
    y_test_folds = y_train_5[test_index]

    clone_clf.fit(X_train_folds, y_train_folds)
    y_pred = clone_clf.predict(X_test_folds)
    n_correct = sum(y_pred == y_test_folds)
    print(n_correct / len(y_pred))


from sklearn.model_selection import cross_val_score

print(cross_val_score(sgd_clf, X_train, y_train_5, cv=3, scoring='accuracy'))


from sklearn.base import BaseEstimator

class Never5Classifier(BaseEstimator):
    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)


never_5_clf = Never5Classifier()
print(cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring='accuracy'))


from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
print(confusion_matrix(y_train_5, y_train_pred))


from sklearn.metrics import f1_score

print(f1_score(y_train_5, y_train_pred))
y_scores = sgd_clf.decision_function([some_digit])
print(y_scores)
threshold = 0
y_some_digit_pred = (y_scores > threshold)
y_some_digit_pred


from sklearn.metrics import precision_recall_curve

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method='decision_function')
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.figure(figsize=(8, 4))
    plt.plot(thresholds, precisions[:-1], 'b--', label='precision')
    plt.plot(thresholds, recalls[:-1], 'g--', label='recall')
    plt.legend(loc="center right", fontsize=16)  # Not shown in the book
    plt.xlabel("Threshold", fontsize=16)  # Not shown
    plt.grid(True)  # Not shown
    plt.axis([-50000, 50000, 0, 1])  # Not shown


plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show(block=False)
plt.pause(2)
plt.close()


from sklearn.metrics import precision_recall_curve

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method='decision_function')
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.figure(figsize=(8, 4))
    plt.plot(thresholds, precisions[:-1], 'b--', label='precision')
    plt.plot(thresholds, recalls[:-1], 'g--', label='recall')
    # plt.legend(loc="center right", fontsize=16) # Not shown in the book
    # plt.xlabel("Threshold", fontsize=16)        # Not shown
    # plt.grid(True)                              # Not shown
    # plt.axis([-50000, 50000, 0, 1])             # Not shown


plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
plt.show(block=False)
plt.pause(2)
plt.close()


from sklearn.metrics import precision_score, recall_score

threshold_90_precision = thresholds[np.argmax(precisions >= 0.9)]
y_train_pred_90 = (y_scores >= threshold_90_precision)
print(precision_score(y_train_5, y_train_pred_90))
print(recall_score(y_train_5, y_train_pred_90))


from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')


plot_roc_curve(fpr, tpr)
plt.show(block=False)
plt.pause(2)
plt.close()

print(roc_auc_score(y_train_5, y_scores))


from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method='predict_proba')
y_scores_forest = y_probas_forest[:, 1]
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)

plt.plot(fpr, tpr, 'b:', label='SGD')
plot_roc_curve(fpr_forest, tpr_forest, 'Random Forest')
plt.legend(loc='lower right')
plt.show(block=False)
plt.pause(2)
plt.close()

print(roc_auc_score(y_train_5, y_scores_forest))


from sklearn.svm import SVC

svm_clf = SVC()
svm_clf.fit(X_train, y_train)
print(svm_clf.predict([some_digit]))

some_digit_scores = svm_clf.decision_function([some_digit])
print(some_digit_scores)
some_digit_scores = svm_clf.decision_function([some_digit])
print(some_digit_scores)
svm_clf.classes_[np.argmax(some_digit_scores)]


from sklearn.multiclass import OneVsRestClassifier

ovr_clf = OneVsRestClassifier(SVC(), n_jobs=-1)
ovr_clf.fit(X_train, y_train)
print(ovr_clf.predict([some_digit]))
print(ovr_clf.estimators_)
print(len(ovr_clf.estimators_))
sgd_clf.fit(X_train, y_train)
print(sgd_clf.predict([some_digit]))
print(sgd_clf.decision_function([some_digit]))
print(cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring='accuracy'))


from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
print(cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring='accuracy'))
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
print(conf_mx)
plt.matshow(conf_mx, cmap=plt.cm.gray)
plt.show(block=False)
plt.pause(2)
plt.close()
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums
print(row_sums)
print(norm_conf_mx)
np.fill_diagonal(norm_conf_mx, 0)
plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
plt.show(block=False)
plt.pause(2)
plt.close()


# 숫자 그림을 위한 추가 함수
def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    # n_rows = int(len(instances) / images_per_row) 와 동일합니다:
    n_rows = (len(instances) - 1) // images_per_row + 1

    # 필요하면 그리드 끝을 채우기 위해 빈 이미지를 추가합니다:
    n_empty = n_rows * images_per_row - len(instances)
    padded_instances = np.concatenate([instances, np.zeros((n_empty, size * size))], axis=0)

    # 배열의 크기를 바꾸어 28×28 이미지를 담은 그리드로 구성합니다:
    image_grid = padded_instances.reshape((n_rows, images_per_row, size, size))

    # 축 0(이미지 그리드의 수직축)과 2(이미지의 수직축)를 합치고 축 1과 3(두 수평축)을 합칩니다.
    # 먼저 transpose()를 사용해 결합하려는 축을 옆으로 이동한 다음 합칩니다:
    big_image = image_grid.transpose(0, 2, 1, 3).reshape(n_rows * size,
                                                         images_per_row * size)
    # 하나의 큰 이미지를 얻었으므로 출력하면 됩니다:
    plt.imshow(big_image, cmap=mpl.cm.binary, **options)
    plt.axis("off")


cl_a, cl_b = 3, 5
X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]

plt.figure(figsize=(8, 8))
for idx, arr in enumerate([X_aa, X_ab, X_ba, X_bb], 221):
    plt.subplot(idx)
    plot_digits(arr[:25], images_per_row=5)
plt.show(block=False)
plt.pause(2)
plt.close()


from sklearn.neighbors import KNeighborsClassifier

y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_multilabel)
print(knn_clf.predict([some_digit]))
y_train_knn_pred = cross_val_predict(knn_clf, X_train, y_multilabel, cv=3)
print(f1_score(y_multilabel, y_train_knn_pred, average='macro'))
noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = X_test


def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap=mpl.cm.binary,
               interpolation="nearest")
    plt.axis("off")


some_index = 0

knn_clf.fit(X_train_mod, y_train_mod)
clean_digit = knn_clf.predict([X_test_mod[some_index]])
plot_digit(clean_digit)


## 연습문제 구현
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

knn_clf = KNeighborsClassifier()

params = {'n_neighbors': [3, 4, 5], 'weights': ['uniform', 'distance']}

X_train_pr = X_train[:len(X_train) // 10]
y_train_pr = y_train[:len(y_train) // 10]

grid_search = GridSearchCV(knn_clf, params, cv=5,
                           scoring='accuracy', n_jobs=-1,
                           return_train_score=True, error_score='raise')

grid_search.fit(X_train_pr, y_train_pr)
del X_train_pr, y_train_pr
params = {k: [v] for k, v in grid_search.best_params_.items()}

grid_search = GridSearchCV(knn_clf, params, cv=5,
                           scoring='accuracy', n_jobs=-1,
                           return_train_score=True, error_score='raise')

grid_search.fit(X_train, y_train)



from sklearn.metrics import accuracy_score

print(grid_search.best_params_, grid_search.best_score_)
y_pred = grid_search.predict(X_test)
print(accuracy_score(y_test, y_pred))


def set_expansion(X_data: np.array) -> np.array:
    X_data = X_data.reshape(-1, 28, 28)
    res = [arr[1:] + arr[:1] for arr in map(list, X_data)]
    res += [arr[-1:] + arr[:-1] for arr in map(list, X_data)]
    res += [np.array(arr[1:] + arr[:1]).T for arr in map(list, map(np.transpose, X_data))]
    res += [np.array(arr[-1:] + arr[:-1]).T for arr in map(list, map(np.transpose, X_data))]
    return np.array(res).reshape(-1, 28 * 28)


X_train_exp = set_expansion(X_train)
y_train_exp = np.concatenate([y_train, y_train, y_train, y_train])
image = X_train[0]

plt.figure(figsize=(12, 3))
plt.subplot(131)
plt.title("Original", fontsize=14)
plt.imshow(image.reshape(28, 28), interpolation="nearest", cmap="Greys")
plt.subplot(132)
plt.title("Shifted down", fontsize=14)
plt.imshow(X_train_exp[60000].reshape(28, 28), interpolation="nearest", cmap="Greys")
plt.subplot(133)
plt.title("Shifted left", fontsize=14)
plt.imshow(X_train_exp[120000].reshape(28, 28), interpolation="nearest", cmap="Greys")
plt.show(block=False)
plt.pause(2)
plt.close()


grid_search = GridSearchCV(knn_clf, params, cv=5,
                           scoring='accuracy', n_jobs=-1,
                           return_train_score=True, error_score='raise')

# grid_search.fit(X_train_exp, y_train_exp)

idx_list = list([range(i * 10000, (i + 2) * 10000) for i in [0, 6, 12, 18]])
X_train_exp_80k = X_train_exp[idx_list].reshape(-1,784)
y_train_exp_80k = y_train_exp[idx_list].flatten()
grid_search.fit(X_train_exp_80k, y_train_exp_80k)

print(grid_search.best_params_, grid_search.best_score_)
y_pred = grid_search.predict(X_test)
print(accuracy_score(y_test, y_pred))



import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import warnings

warnings.filterwarnings('ignore')

train = pd.read_csv('./original/datasets/titanic/train.csv')
test = pd.read_csv('./original/datasets/titanic/test.csv')
train.info()
use_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Fare', 'Embarked']
X_train = train[use_cols]
X_train.Age = X_train.Age.fillna(X_train.Age.mean()).apply(lambda x: int(x * 0.1))
X_test = test[use_cols]
X_test.Age = X_test.Age.fillna(X_train.Age.mean()).apply(lambda x: int(x * 0.1))

y_train = train['Survived']
cat_list = ['Age', 'Sex', 'Embarked', 'Pclass']
num_list = ['Pclass', 'SibSp', 'Fare']

cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

pipeline = ColumnTransformer([
    ('num', SimpleImputer(strategy='mean'), num_list),
    ('cat', cat_pipeline, cat_list)
])

label_encoder = LabelEncoder()
y_train_prepared = label_encoder.fit_transform(y_train)
X_train_prepared = pipeline.fit_transform(X_train)
X_test_prepared = pipeline.transform(X_test)
rfc = RandomForestClassifier()

param_grid = {'n_estimators': [3, 10, 30, 50, 100],
              'max_features': [2, 4, 6, 8, 10, 12, 15, 20, 30, 50]}

grid_search = GridSearchCV(rfc, param_grid, cv=5,
                           scoring='accuracy',
                           return_train_score=True, error_score='raise')
grid_search.fit(X_train_prepared, y_train_prepared)
print(grid_search.best_params_)
print(classification_report(y_train_prepared, grid_search.best_estimator_.predict(X_train_prepared)))



### 4번 문제는 풀이에 있는 코드를 분석하는 것으로 대체
import os
import tarfile
import urllib.request

DOWNLOAD_ROOT = "http://spamassassin.apache.org/old/publiccorpus/"
HAM_URL = DOWNLOAD_ROOT + "20030228_easy_ham.tar.bz2"
SPAM_URL = DOWNLOAD_ROOT + "20030228_spam.tar.bz2"
SPAM_PATH = os.path.join('original', "datasets", "spam")


def fetch_spam_data(ham_url=HAM_URL, spam_url=SPAM_URL, spam_path=SPAM_PATH):
    if not os.path.isdir(spam_path):
        os.makedirs(spam_path)
    for filename, url in (("ham.tar.bz2", ham_url), ("spam.tar.bz2", spam_url)):
        path = os.path.join(spam_path, filename)
        if not os.path.isfile(path):
            urllib.request.urlretrieve(url, path)
            # url의 정보를 path에 저장
        tar_bz2_file = tarfile.open(path)
        tar_bz2_file.extractall(path=spam_path)
        tar_bz2_file.close()


fetch_spam_data()

HAM_DIR = os.path.join(SPAM_PATH, "easy_ham")
SPAM_DIR = os.path.join(SPAM_PATH, "spam")
ham_filenames = [name for name in sorted(os.listdir(HAM_DIR)) if len(name) > 20]
spam_filenames = [name for name in sorted(os.listdir(SPAM_DIR)) if len(name) > 20]


import email
import email.policy


def load_email(is_spam, filename, spam_path=SPAM_PATH):
    directory = "spam" if is_spam else "easy_ham"
    with open(os.path.join(spam_path, directory, filename), "rb") as f:
        return email.parser.BytesParser(policy=email.policy.default).parse(f)
        # 바이트 형식의 이메일을 파싱하여 반환


ham_emails = [load_email(is_spam=False, filename=name) for name in ham_filenames]
spam_emails = [load_email(is_spam=True, filename=name) for name in spam_filenames]


def get_email_structure(email):
    if isinstance(email, str):
        return email
    payload = email.get_payload()
    if isinstance(payload, list):
        return "multipart({})".format(", ".join([
            get_email_structure(sub_email)
            for sub_email in payload
        ]))
    else:
        return email.get_content_type()
    # 이메일 구성 형식 반환


from collections import Counter


def structures_counter(emails):
    structures = Counter()
    for email in emails:
        structure = get_email_structure(email)
        structures[structure] += 1
    return structures
    # 이메일 구성 형식 비율 확인용 함수
    # 이를 통해 ham과 spam의 구성 형식 비교 가능


print(structures_counter(ham_emails).most_common())
print(structures_counter(spam_emails).most_common())
for header, value in spam_emails[0].items():
    print(header, ":", value)
print(spam_emails[0]["Subject"])


import numpy as np
from sklearn.model_selection import train_test_split

X = np.array(ham_emails + spam_emails, dtype=object)
y = np.array([0] * len(ham_emails) + [1] * len(spam_emails))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


import re
from html import unescape


def html_to_plain_text(html):
    # 구성 형식이 html일 경우 태그 정보 제거하고 텍스트만 반환
    text = re.sub('<head.*?>.*?</head>', '', html, flags=re.M | re.S | re.I)
    # - re.M: 멀티라인 모드, ^와 $가 각 줄의 시작과 끝을 의미.
    # - re.S: 점(.)이 개행 문자(\n)를 포함한 모든 문자와 매칭.
    # - re.I: 대소문자를 구분하지 않음.

    text = re.sub('<a\s.*?>', ' HYPERLINK ', text, flags=re.M | re.S | re.I)
    text = re.sub('<.*?>', '', text, flags=re.M | re.S)
    text = re.sub(r'(\s*\n)+', '\n', text, flags=re.M | re.S)
    return unescape(text)  # unescape: html 이스케이프 문자를 일반 문자로 변환


html_spam_emails = [email for email in X_train[y_train == 1]
                    if get_email_structure(email) == "text/html"]
sample_html_spam = html_spam_emails[7]
print(sample_html_spam.get_content().strip()[:1000], "...")
print(html_to_plain_text(sample_html_spam.get_content())[:1000], "...")


def email_to_text(email):
    html = None
    for part in email.walk():
        ctype = part.get_content_type()
        if not ctype in ("text/plain", "text/html"):
            continue
        try:
            content = part.get_content()
        except:  # in case of encoding issues
            content = str(part.get_payload())
        if ctype == "text/plain":
            return content
        else:
            html = content

    if html:
        return html_to_plain_text(html)


print(email_to_text(sample_html_spam)[:100], "...")


import urlextract

url_extractor = urlextract.URLExtract()
print(url_extractor.find_urls("Will it detect github.com and https://youtu.be/7Pq-S557XQU?t=3m32s"))
from sklearn.base import BaseEstimator, TransformerMixin
from nltk import PorterStemmer

stemmer = PorterStemmer()


class EmailToWordCounterTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, strip_headers=True, lower_case=True, remove_punctuation=True,
                 replace_urls=True, replace_numbers=True, stemming=True):
        self.strip_headers = strip_headers
        self.lower_case = lower_case
        self.remove_punctuation = remove_punctuation
        self.replace_urls = replace_urls
        self.replace_numbers = replace_numbers
        self.stemming = stemming

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X_transformed = []
        for email in X:
            text = email_to_text(email) or ""  # 텍스트 추출
            if self.lower_case:
                text = text.lower()

            if self.replace_urls and url_extractor is not None:  # 하이퍼링크는 전부 "URL"로 처리
                urls = list(set(url_extractor.find_urls(text)))
                urls.sort(key=lambda url: len(url), reverse=True)
                for url in urls:
                    text = text.replace(url, " URL ")
            if self.replace_numbers:  # 숫자는 전부 NUMBER 처리
                text = re.sub(r'\d+(?:\.\d*)?(?:[eE][+-]?\d+)?', 'NUMBER', text)
            if self.remove_punctuation:
                text = re.sub(r'\W+', ' ', text, flags=re.M)
            word_counts = Counter(text.split())
            if self.stemming and stemmer is not None:
                stemmed_word_counts = Counter()
                for word, count in word_counts.items():
                    stemmed_word = stemmer.stem(word)
                    stemmed_word_counts[stemmed_word] += count
                word_counts = stemmed_word_counts
            X_transformed.append(word_counts)
        return np.array(X_transformed)


X_few = X_train[:3]
X_few_wordcounts = EmailToWordCounterTransformer().fit_transform(X_few)
print(X_few_wordcounts)


from scipy.sparse import csr_matrix  # 희소 행렬 함수

class WordCounterToVectorTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, vocabulary_size=1000):
        self.vocabulary_size = vocabulary_size

    def fit(self, X, y=None):
        total_count = Counter()
        for word_count in X:
            for word, count in word_count.items():
                total_count[word] += min(count, 10)
        most_common = total_count.most_common()[:self.vocabulary_size]
        self.vocabulary_ = {word: index + 1 for index, (word, count) in enumerate(most_common)}
        return self

    def transform(self, X, y=None):
        rows = []
        cols = []
        data = []
        for row, word_count in enumerate(X):
            for word, count in word_count.items():
                rows.append(row)
                cols.append(self.vocabulary_.get(word, 0))
                data.append(count)
        return csr_matrix((data, (rows, cols)), shape=(len(X), self.vocabulary_size + 1))
        # csr_matrix(값, (행좌표, 열좌표), shape)


from sklearn.pipeline import Pipeline

preprocess_pipeline = Pipeline([
    ("email_to_wordcount", EmailToWordCounterTransformer()),
    ("wordcount_to_vector", WordCounterToVectorTransformer()),
])

X_train_transformed = preprocess_pipeline.fit_transform(X_train)


from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

log_clf = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
score = cross_val_score(log_clf, X_train_transformed, y_train, cv=3, verbose=3)
print(score.mean())


from sklearn.metrics import precision_score, recall_score

X_test_transformed = preprocess_pipeline.transform(X_test)

log_clf = LogisticRegression(solver="lbfgs", max_iter=1000, random_state=42)
log_clf.fit(X_train_transformed, y_train)

y_pred = log_clf.predict(X_test_transformed)

print("정밀도: {:.2f}%".format(100 * precision_score(y_test, y_pred)))
print("재현율: {:.2f}%".format(100 * recall_score(y_test, y_pred)))