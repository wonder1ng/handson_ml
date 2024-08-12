# 1장
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import linear_model

def prepare_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]

oecd_bli = pd.read_csv("original/datasets/lifesat/oecd_bli_2015.csv", thousands=',')
gdp_per_capita = pd.read_csv("original/datasets/lifesat/gdp_per_capita.csv", thousands=',', delimiter='\t',
                             encoding='latin1', na_values='n/a')

country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)
X = np.c_[country_stats["GDP per capita"]]
y = np.c_[country_stats["Life satisfaction"]]

country_stats.plot(kind='scatter', x='GDP per capita', y='Life satisfaction')
plt.show(block=False)
plt.pause(2)
plt.close()

model = linear_model.LinearRegression()
model.fit(X, y)

X_new = [[22587]]   # 키프로스 1인당 GDP
print(model.predict(X_new))

# 2징
import numpy as np
import pandas as pd

housing = pd.read_csv('original/datasets/housing/housing.csv')
housing.info()

import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20, 15))
plt.show(block=False)
plt.pause(2)
plt.close()

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

housing.reset_index()
train_set, test_set = train_test_split(housing.reset_index(), test_size=0.2, random_state=42)

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

housing['income_cat'] = pd.cut(housing['median_income'],
                               bins=[0, 1.5, 3, 4.5, 6, np.inf],
                               labels=[1,2,3,4,5])

train_set, test_set = train_test_split(housing.reset_index(), test_size=0.2, random_state=42)
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['income_cat']):
    start_train_set = housing.loc[train_index]
    start_test_set = housing.loc[test_index]

housing = start_train_set.copy()
housing.plot(kind='scatter', x='longitude', y='latitude', alpha=0.4,
             s=housing['population']/100, label='population', figsize=(10, 7),
             c='median_house_value', cmap='jet', colorbar=True,
             sharex=False)
plt.legend()
plt.show(block=False)
plt.pause(2)
plt.close()

corr_matrix = housing.corr(numeric_only=True)
corr_matrix['median_house_value'].sort_values(ascending=False)
from pandas.plotting import scatter_matrix

attributes = ['median_house_value', 'median_income', 'total_rooms', 'housing_median_age']
scatter_matrix(housing[attributes], figsize=(12,8))
plt.show(block=False)
plt.pause(2)
plt.close()

housing.plot(kind='scatter', x='median_income', y='median_house_value', alpha=0.1)
plt.show(block=False)
plt.pause(2)
plt.close()

housing['rooms_per_household'] = housing['total_rooms']/housing['households']
housing['bedrooms_per_room'] = housing['total_bedrooms']/housing['total_rooms']
housing['population_per_household'] = housing['population']/housing['households']

corr_matrix = housing.corr(numeric_only=True)
corr_matrix['median_house_value'].sort_values(ascending=False)

housing = start_train_set.drop('median_house_value', axis=1)
housing_labels = start_train_set['median_house_value'].copy()


from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy='median')
housing_num = housing.drop('ocean_proximity', axis=1)
imputer.fit(housing_num)
imputer.statistics_ # 각 열별 중앙값

X = imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns, index=housing_num.index)


from sklearn.preprocessing import OneHotEncoder

housing_cat = housing[['ocean_proximity']]

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)

from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, housholds_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True) -> None:
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, housholds_ix]
        population_per_househod = X[:, population_ix] / X[:, housholds_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, housholds_ix]
            return np.c_[X, rooms_per_household, population_per_househod, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_househod]


attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler())
])

housing_num_tr = num_pipeline.fit_transform(housing_num)


from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ['ocean_proximity']

full_pipeline = ColumnTransformer([
    ('num', num_pipeline, num_attribs),
    ('cat', OneHotEncoder(), cat_attribs)
])

housing_prepared = full_pipeline.fit_transform(housing)


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]

some_data_prepared = full_pipeline.transform(some_data)
print("예측:", lin_reg.predict(some_data_prepared))
print('레이블:', list(some_labels))

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
print("예측:", tree_reg.predict(some_data_prepared))
print('레이블:', list(some_labels))

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print(tree_rmse)


from sklearn.model_selection import cross_val_score

def display_scores(scores):
    print("점수:", scores)
    print("평균:", scores.mean())
    print("표준편차:", scores.std())

scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring='neg_mean_squared_error', cv=10)
tree_rmse_scores = np.sqrt(-scores)
display_scores(tree_rmse_scores)

print('='*20)

scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring='neg_mean_squared_error', cv=10)
lin_rmse_scores = np.sqrt(-scores)
display_scores(lin_rmse_scores)


from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
print("예측:", forest_reg.predict(some_data_prepared))
print('레이블:', list(some_labels))

housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
print(forest_rmse)

scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring='neg_mean_squared_error', cv=10)
forest_rmse_scores = np.sqrt(-scores)
display_scores(forest_rmse_scores)


from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True, error_score='raise')

grid_search.fit(housing_prepared, housing_labels)
cvres = grid_search.cv_results_
for mean_score, parmas in sorted(zip(cvres['mean_test_score'], cvres['params']), reverse=True):
    print(np.sqrt(-mean_score), parmas)


feature_importances = grid_search.best_estimator_.feature_importances_
extra_attribs = ['rooms_per_hhold', 'pop_per_hhold', 'bedrooms_per_room']
cat_encoder = full_pipeline.named_transformers_['cat']
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)


from scipy import stats

final_model = grid_search.best_estimator_

X_test = start_test_set.drop('median_house_value', axis=1)
y_test = start_test_set['median_house_value'].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)


confidence = 0.95
squared_erros = (final_predictions - y_test)**2
print(np.sqrt(stats.t.interval(confidence, len(squared_erros)-1,
                         loc=squared_erros.mean(),
                         scale=stats.sem(squared_erros))))

from sklearn.svm import SVR
from sklearn.model_selection import RandomizedSearchCV

param_grid = {
    'kernel': ['linear', 'rbf'],
    'C': np.logspace(-3, 3, 7),
    'gamma': ['scale', 'auto'] + np.logspace(-3, 3, 7).tolist()
    }

svr = SVR()

random_search = RandomizedSearchCV(
    estimator=svr,
    param_distributions=param_grid,
    n_iter=100,
    cv=5,
    verbose=1,
    n_jobs=-1,
    random_state=42
)


random_search.fit(housing_prepared, housing_labels)

from sklearn.base import BaseEstimator, TransformerMixin


class SelectBestFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, feature_importances, k) -> None:
        self.feature_importances = feature_importances
        self.k = k

    def fit(self, X, y=None):
        self.best_feature_idces = np.argsort(self.feature_importances)[-k:]
        return self

    def transform(self, X):
        print(self.best_feature_idces)
        return X[:, self.best_feature_idces]


k = 5
select_feature_pipeline = Pipeline([
    ('preparation', full_pipeline),
    ('feature_selection', SelectBestFeatures(feature_importances, k))
])
res = select_feature_pipeline.fit_transform(housing)


select_feature_pipeline = Pipeline([
    ('preparation', full_pipeline),
    ('feature_selection', SelectBestFeatures(feature_importances, k)),
    ('svr', SVR(**random_search.best_params_))
])
select_feature_pipeline.fit(housing, housing_labels)


some_data = housing.iloc[:4]
some_labels = housing_labels.iloc[:4]

print("Predictions:\t", select_feature_pipeline.predict(some_data))
print("Labels:\t\t", list(some_labels))


full_pipeline.named_transformers_["cat"].handle_unknown = 'ignore'
# full_pipeline에 있는 'cat'이라고 명명한 OneHotEncoder의 handle_unknown 인자에 'ignore' 할당

param_grid = [{
    'preparation__num__imputer__strategy': ['mean', 'median', 'most_frequent'],
    'feature_selection__k': list(range(1, len(feature_importances) + 1))
}]

grid_search_prep = GridSearchCV(select_feature_pipeline, param_grid, cv=5,
                                scoring='neg_mean_squared_error', verbose=2)
grid_search_prep.fit(housing, housing_labels)

print(grid_search_prep.best_params_)