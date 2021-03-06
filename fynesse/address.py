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
from sklearn.metrics import r2_score
import random
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.utils import resample
import numpy as np

"""Address a particular question that arises from the data"""


def get_basic_linear_regressor():
	return LinearRegression()


"""Functions that deal with the dataset"""

def generate_train_test_split(df, test_size=0.1):
	return train_test_split(df, test_size=test_size)


def split_data_into_x_and_y(df, y_col='price'):
	lst = list(df.columns)
	lst.remove(y_col)
	return df[lst], df[y_col]


def upsample(df, size=5000):
	l = len(df)
	if l >= size:
		return df
	df = resample(df, random_state=0, n_samples=size, replace = True)
	return df

"""Functions related to PCA"""


def scale_and_reduce(df):
	"""For PCA, some columns need to be scaled"""
	cols_to_keep = list(df.columns)
	cols_to_keep.remove('price')
	df_1 = df[cols_to_keep]
	scaled = preprocessing.scale(df_1)
	df_1_s = pd.DataFrame(scaled, columns=cols_to_keep)
	df = df.drop(cols_to_keep, axis=1)
	return df.join(df_1_s)


def dimension_reduction(df, dim=3):
	pca = PCA(n_components=dim)
	useful_cols = list(df.columns)
	useful_cols.remove('price')
	x = df[useful_cols]
	y = df['price']
	df = pd.DataFrame(pca.fit_transform(x))
	df = df.join(y)
	return df


"""Functions related to model and model performance"""


def feature_importance(x, y, model):
	features = x.columns
	model.fit(x, y)

	r = permutation_importance(model, x, y, n_repeats=20, random_state=0)

	for i in r.importances_mean.argsort()[::-1]:
		if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
			print(f"{features[i]:<8}"
						f"{r.importances_mean[i]:.3f}"
						f" +/- {r.importances_std[i]:.3f}")


def cross_validate_model(df, model, cv = 5):
	x, y = split_data_into_x_and_y(df)
	scores = cross_val_score(model, x, y, cv = cv, scoring = 'r2')
	print(f"Model has an R2 score of {np.mean(scores)}")


def calculate_r2_performance_of_model(df, model):
	train, test = generate_train_test_split(df)
	train = upsample(train)
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