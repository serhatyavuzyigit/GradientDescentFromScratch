import sys
import random
import math
import pandas as pd

data_file = sys.argv[1]
model_file = sys.argv[2]

data = pd.read_csv(data_file)
data_size = len(data)
column_size = len(data.columns)
train_size = int(data_size * 0.8)
test_size = data_size - train_size

target = data.columns[column_size - 1]


def take_features():
    _features = []
    for i in range(1, column_size - 1):
        _features.append(data.columns[i])
    return _features


features = take_features()


def take_train_data(_data, _size, _target):
    _TRAIN_DATA = {}
    for x in features:
        _TRAIN_DATA[x] = _data.loc[0:_size, x].values
    _TRAIN_DATA[_target] = _data.loc[0:_size, _target].values
    return _TRAIN_DATA


TRAIN_DATA = take_train_data(data, data_size * 0.8 - 1, target)


def take_test_data(_data, _size, _target):
    _TEST_DATA = {}
    for x in features:
        _TEST_DATA[x] = _data.loc[_size:, x].values
    _TEST_DATA[_target] = _data.loc[_size:, _target].values
    return _TEST_DATA


TEST_DATA = take_test_data(data, data_size * 0.8-1, target)


f = open(model_file, 'r')
poly_func_str = f.readline()
f.close()

poly_func = poly_func_str.split('+')


feature_powers = {}
feature_params = {}
for i in range(len(poly_func)):
    for j in range(len(features)):
        if features[j] in poly_func[i]:
            if '*' in poly_func[i]:
                param = poly_func[i].split('*')[0]
                power = 1
                if '^' in poly_func[i]:
                    power = int(poly_func[i].split('^')[1])
                feature_params[features[j]] = param.replace(' ', '')
                feature_powers[features[j]] = power
            else:
                param = ''
                power = 1
                if '^' in poly_func[i]:
                    power = int(poly_func[i].split('^')[1])
                feature_params[features[j]] = param.replace(' ', '')
                feature_powers[features[j]] = power

counter = 0
for i in range(len(features)):
    if features[i] in poly_func[len(poly_func) - 1]:
        counter = counter + 1
if counter == 0:
    feature_params['Beta_0'] = poly_func[len(poly_func) - 1].replace(' ', '')
else:
    feature_params['Beta_0'] = ''


if feature_params['Beta_0'] != '':
    feature_params['Beta_0'] = random.randint(0, 5)
else:
    feature_params['Beta_0'] = 0

for key in feature_params:
    if key != 'Beta_0':
        feature_params[key] = random.uniform(0, 1)


def cal_predictions():
    _predictions = []
    for i in range(int(train_size)):
        y_pred = 0
        for key in feature_params:
            if key != 'Beta_0':
                y_pred = y_pred + (feature_params[key]) * pow(TRAIN_DATA[key][i], feature_powers[key])
        y_pred = y_pred + feature_params['Beta_0']
        _predictions.append(y_pred)
    return _predictions


def cal_predictions_with_range(_from, _to):
    _predictions = []
    for i in range(_from, _to):
        y_pred = 0
        for key in feature_params:
            if key != 'Beta_0':
                y_pred = y_pred + (feature_params[key]) * pow(TRAIN_DATA[key][i], feature_powers[key])
        y_pred = y_pred + feature_params['Beta_0']
        _predictions.append(y_pred)
    return _predictions



def cost_val(_predictions):
    n = int(train_size)
    _sum = 0
    for i in range(n):
        value = TRAIN_DATA[target][i] - _predictions[i]
        value = value ** 2
        _sum = _sum + value

    return (1 / n) * _sum


def cost_val_with_range(_predictions, _from, _to):
    n = _to - _from
    _sum = 0
    j = _from
    for i in range(n):
        value = TRAIN_DATA[target][j] - _predictions[i]
        value = value ** 2
        _sum = _sum + value
        j = j + 1

    return (1 / n) * _sum


def feature_derive_sum(_feature, _predictions):
    n = int(train_size)
    _sum = 0
    for i in range(n):
        _sum = _sum + pow(TRAIN_DATA[_feature][i], feature_powers[_feature]) * (TRAIN_DATA[target][i] - _predictions[i])
    return _sum


def feature_derive_sum_with_range(_feature, _predictions, _from, _to):
    n = _to - _from
    _sum = 0
    j = _from
    for i in range(n):
        _sum = _sum + pow(TRAIN_DATA[_feature][j], feature_powers[_feature]) * (TRAIN_DATA[target][j] - _predictions[i])
        j = j + 1
    return _sum


def derive_sum(_predictions):
    n = int(train_size)
    _sum = 0
    for i in range(n):
        _sum = _sum + TRAIN_DATA[target][i] - _predictions[i]
    return _sum


def derive_sum_with_range(_predictions, _from, _to):
    n = _to - _from
    _sum = 0
    j = _from
    for i in range(n):
        _sum = _sum + TRAIN_DATA[target][j] - _predictions[i]
        j = j + 1
    return _sum


def decide_batch_size():
    x = 4
    if train_size % 10 == 0:
        x = 10
    elif train_size % 9 == 0:
        x = 9
    elif train_size % 8 == 0:
        x = 8
    elif train_size % 7 == 0:
        x = 7
    elif train_size % 6 == 0:
        x = 6

    return x


learning_rate = 0.0001
index = 0

x = decide_batch_size()

batch_size = int(train_size / x)
gradient_values = {}
for feature in feature_params:
    gradient_values[feature] = 0

gradient_status = {}
for feature in feature_params:
    gradient_status[feature] = -1

while True:
    _from = index * batch_size
    _to = (index + 1) * batch_size

    if _to > train_size:
        break

    while True:
        _cc = 0
        _n = _to - _from
        predictions = cal_predictions_with_range(_from, _to)
        cost = cost_val_with_range(predictions, _from, _to)

        for feature in feature_params:
            if feature != 'Beta_0':
                if gradient_status[feature] == -1:
                    fd = -(2 / _n) * feature_derive_sum_with_range(feature, predictions, _from, _to)
                    feature_params[feature] = feature_params[feature] - learning_rate * fd
                    if gradient_values[feature] == 0:
                        gradient_values[feature] = fd
                    else:
                        old_fd = gradient_values[feature]
                        if abs(old_fd - fd) < 0.00000001:
                            gradient_status[feature] = 1
                        else:
                            gradient_values[feature] = fd
            else:
                if gradient_status['Beta_0'] == -1:
                    bd = -(2 / _n) * derive_sum_with_range(predictions, _from, _to)
                    feature_params['Beta_0'] = feature_params['Beta_0'] - learning_rate * bd
                    if gradient_values['Beta_0'] == 0:
                        gradient_values['Beta_0'] = bd
                    else:
                        old_bd = gradient_values['Beta_0']
                        if abs(old_bd - bd) < 0.00000001:
                            gradient_status['Beta_0'] = 1
                        else:
                            gradient_values['Beta_0'] = bd

        for x in gradient_status:
            if gradient_status[x] == 1:
                _cc = _cc + 1

        if _cc == len(feature_params):
            break

    for fea in feature_params:
        gradient_status[fea] = -1
        gradient_values[fea] = 0

    index = index + 1


def cal_test_predictions():
    _test_predictions = []
    _n = test_size
    for i in range(_n):
        _pred = 0
        for _feature in feature_params:
            if _feature != 'Beta_0':
                _pred = _pred + (feature_params[_feature]) * pow(TEST_DATA[_feature][i], feature_powers[_feature])
        _pred = _pred + feature_params['Beta_0']
        _test_predictions.append(_pred)

    return _test_predictions


def cal_test_mse(_test_predictions):
    _mse_sum = 0
    _n = len(_test_predictions)
    for i in range(_n):
        value = TEST_DATA[target][i] - _test_predictions[i]
        value = value**2
        _mse_sum = _mse_sum + value
    return (1 / _n) * _mse_sum


def cal_test_rmse(_mse):
    return math.sqrt(_mse)


def cal_test_tss():
    _mean_sum = 0
    _n = len(TEST_DATA[target])
    for i in range(_n):
        _mean_sum = _mean_sum + TEST_DATA[target][i]
    _mean = _mean_sum / _n

    _tss_sum = 0
    for i in range(_n):
        value = TEST_DATA[target][i] - _mean
        value = value**2
        _tss_sum = _tss_sum + value
    return _tss_sum


def cal_test_rss(_test_predictions):
    _n = len(_test_predictions)
    _rss_sum = 0
    for i in range(_n):
        value = TEST_DATA[target][i] - _test_predictions[i]
        value = value**2
        _rss_sum = _rss_sum + value
    return _rss_sum


def cal_test_r2(_test_predictions):
    _tss = cal_test_tss()
    _rss = cal_test_rss(_test_predictions)
    _r2 = (_tss - _rss) / _tss
    return _r2

test_predictions = cal_test_predictions()
print(TEST_DATA[target])
print(test_predictions)

mse = cal_test_mse(test_predictions)
rmse = cal_test_rmse(mse)
r2 = cal_test_r2(test_predictions)

print('MSE on test data: ', mse)
print('RMSE on test data: ', rmse)
print('R2 on test data: ', r2)
