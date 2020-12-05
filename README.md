# MSBD5001-Kaggle

## Programming Language & Development Tool
1. The main programming language is **Python**.
2. The development tool is PyCharm.

## Required Packages
1. lightgbm
2. numpy
3. pandas
4. sklearn

## How does it work
1. The data is stored under ./data/
2. First read train.csv, get the day, month, hour and speed. I use the day, month, and hour as features, and speed as the regression target.
3. I use StandardScaler in sklearn to normalize the mean and variance, and then use train_test_split to randomly divide the training set into the testing dataset, and the division ratio is 80% for training and 20% for testing
4. I use the lightgbm model to train and fit the data, and then do the same processing on the features of test.csv. Then, I use the trained model to predict test.csv and write it into result.csv
