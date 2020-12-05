import csv
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn import preprocessing
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")
def getTrainData():
    X = []
    y = []
    with open('./data/train.csv', 'r') as f:
        f.readline()
        reader = csv.reader(f)
        for row in reader:
            time_ = row[1].split()
            d,m,y1 = time_[0].split('/')
            h = time_[1].split(':')
            speed = float(row[2])
            X.append([int(d), int(m), int(h[0])])
            y.append(speed)
    return X, y

X, y = getTrainData() # get the data


scaler = preprocessing.StandardScaler().fit(X)
X= scaler.transform(X) # Implement transformer from class preprocessing module to standardize features

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)#Randomly split the train size to 75% and the test size to 25%



lgb_train = lgb.Dataset(X_train, y_train)#define datalist name
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)

# specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'objective': 'regression',
    'metric': {'l2', 'l1'},
    'num_leaves': 31,
    'learning_rate': 0.1,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 3,
    'verbose': 0
}

print('Starting training...')
# train
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=20,
                valid_sets=lgb_eval,
                early_stopping_rounds=5)

print('Saving model...')
# save model to file
gbm.save_model('model.txt')

print('Starting predicting...')
# predict
y_pred = gbm.predict(X_test, num_iteration=gbm.best_iteration)


from sklearn.metrics import mean_squared_error


def write2csv(filename, datalist):#datalist 
    f = open(filename, 'w', encoding='utf-8', newline='')
    csv_writer = csv.writer(f)
    for dtl in datalist:
        csv_writer.writerow(dtl)
    f.close()


datalist = []
datalist.append(['id', 'speed'])
with open('./data/test.csv', 'r') as f:
    f.readline()
    reader = csv.reader(f)
    for row in reader:
        time_ = row[1].split()
        d, m, y1 = time_[0].split('/')
        h = time_[1].split(':')
        feather = scaler.transform(np.array([[int(d), int(m), int(h[0])]]))
        datalist.append([row[0], gbm.predict(feather, num_iteration=gbm.best_iteration)[0]])
write2csv('Submission.csv', datalist)
print('Finished')
