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
"""Address a particular question that arises from the data"""

def generate_train_test_split(df, test_size=0.1):
	return train_test_split(df, test_size=test_size)

def split_data_into_x_and_y(df, y_col = 'price'):
	lst = list(df.columns)
	lst.remove(y_col)
	return df[lst], df[y_col]


def feature_importance(x, y, model):
	features = x.columns
	model.fit(x, y)

	r = permutation_importance(model, x, y, n_repeats=30, random_state=0)

	for i in r.importances_mean.argsort()[::-1]:
		if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
			print(f"{features[i]:<8}"
						f"{r.importances_mean[i]:.3f}"
						f" +/- {r.importances_std[i]:.3f}")

def get_basic_linear_regressor():
	return LinearRegression()


