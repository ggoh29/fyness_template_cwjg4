# This file contains code for supporting addressing questions in the data

"""# Here are some of the imports we might expect 
import sklearn.model_selection  as ms
import sklearn.linear_model as lm
import sklearn.svm as svm
import sklearn.naive_bayes as naive_bayes
import sklearn.tree as tree

import GPy
import torch
import tensorflow as tf

# Or if it's a statistical analysis
import scipy.stats"""
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.inspection import permutation_importance
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import r2_score
import random
import pandas as pd

"""Address a particular question that arises from the data"""


def get_basic_linear_regressor():
	return LinearRegression()


def generate_train_test_split(df, test_size=0.1):
	return train_test_split(df, test_size=test_size)


def split_data_into_x_and_y(df, y_col='price'):
	lst = list(df.columns)
	lst.remove(y_col)
	return df[lst], df[y_col]


def resample(df, size=1000):
	l = len(df)
	if l >= size:
		return df
	cols = list(df.columns)
	rows = list(df.to_records(index=False))
	resampled_df = [rows[random.randint(0, l - 1)] for _ in range(size)]
	resampled_df = [*zip(*resampled_df)]
	dct = {}
	for index, col in zip(cols, resampled_df):
		dct[index] = col
	return pd.DataFrame(dct)


def feature_importance(x, y, model):
	features = x.columns
	model.fit(x, y)

	r = permutation_importance(model, x, y, n_repeats=10, random_state=0)

	for i in r.importances_mean.argsort()[::-1]:
		if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
			print(f"{features[i]:<8}"
						f"{r.importances_mean[i]:.3f}"
						f" +/- {r.importances_std[i]:.3f}")


def generate_performance_of_model(df, model):
	train, test = generate_train_test_split(df)
	x_train, y_train = split_data_into_x_and_y(train)
	x_test, y_test = split_data_into_x_and_y(test)
	model.fit(x_train, y_train)
	prediction = model.predict(x_test)
	print(f"Model has an R2 score of {r2_score(y_test, prediction)}")


def train_and_predict(model, train, input):
	x_train, y_train = split_data_into_x_and_y(train)
	model.fit(x_train, y_train)
	prediction = model.predict(input)
	print(f"Model has predicted an output of {prediction}")
	return prediction
