import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.linear_model import Lasso, LassoCV, Ridge
from sklearn.decomposition import PCA
import lightgbm
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, roc_auc_score

train = pd.read_csv("./train.csv")
test = pd.read_csv("./test.csv")
test2 = pd.read_csv("./test.csv")

train_len = train.shape[0]
target = train.target
train = train.drop('ID_code', axis=1)
train = train.drop('target', axis=1)
test = test.drop('ID_code', axis=1)

x, x_test, y, y_test = train_test_split(train, target, test_size=0.2, random_state=42, stratify=target)
train_data = lightgbm.Dataset(x, label=y)
test_data = lightgbm.Dataset(x_test, label=y_test)

parameters = {
    'application': 'binary',
    'objective': 'binary',
    'metric': 'auc',
    'is_unbalance': 'true',
    'boosting': 'gbdt',
    'num_leaves': 2,
    'feature_fraction': 0.5,
    'bagging_fraction': 0.5,
    'bagging_freq': 20,
    'learning_rate': 0.05,
    'verbose': 0
}

model = lightgbm.train(parameters,
                       train_data,
                       valid_sets=test_data,
                       num_boost_round=50000,
                       early_stopping_rounds=100)

def threshold_search(y_true, y_proba):
    best_threshold = 0
    best_score = 0
    for threshold in [i * 0.01 for i in range(100)]:
        score = f1_score(y_true=y_true, y_pred=y_proba > threshold)
        if score > best_score:
            best_threshold = threshold
            best_score = score
    search_result = {'threshold': best_threshold, 'f1': best_score}
    print(search_result)

pred = model.predict(train)

threshold_search(target, pred)

pred = model.predict(test)

p = np.squeeze(pred)
result = []

print(p)
'''
for pp in p:
    if pp >= 0.5:
        result.append(1)
    else:
        result.append(0)
'''
output = pd.DataFrame({'ID_code': test2.ID_code, 'target': p})
output.to_csv('submission.csv', index=False)
