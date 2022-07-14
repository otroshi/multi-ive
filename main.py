import os
import data.manage_data as manage_data
import pandas as pd
import evaluation.simple_classifier as simple_classifier
import evaluation.verification_perfromance as vp
import evaluation.utils as eval_utils
import IVE.utils as ive_util
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np


# create
if not os.path.isfile('data/train_df.csv'):
	train_df, test_df = manage_data.get_training_test_df(save_files=True)
else:
	train_df = pd.read_csv('data/train_df.csv')
	test_df = pd.read_csv('data/test_df.csv')

total_indexes = [i for i in range(len(train_df) + len(test_df))]
total_df = pd.concat([train_df, test_df], sort=False)
total_df['indexes'] = total_indexes
total_df = total_df.set_index('indexes')

# set useful variables
labels = ['sex', 'age', 'ethnicity']
# train_df contains 4 columns more than the number of features
length_embeddings = train_df.shape[1] - 4
# number of steps and number of eliminations per step have to be adjusted depending on the feature size
num_steps = 1
num_eliminations = 6
num_epochs = 85

# define classifier
n_estimators = 30
model_train = RandomForestClassifier(n_estimators=n_estimators, random_state=1)
# model_train = ExtraTreesClassifier(n_estimators=n_estimators, random_state=1)
# model_train = GradientBoostingClassifier(n_estimators=n_estimators, random_state=1)
# model_train = XGBClassifier(n_estimators=n_estimators, random_state=1)

# set parameters for method one
n_s = [num_steps, num_steps, num_steps]
n_e = [int(num_eliminations / 3), int(num_eliminations / 3), int(num_eliminations / 3)]
elimination_order = [0, 1, 2]

# get embeddings
x_total = manage_data.get_x_ready(total_df, length_embeddings)

# normalize embeddings
std_sc = StandardScaler()
# fit the scaler with only training data
std_sc.fit(x_total[:len(train_df)])
x_total = std_sc.transform(x_total)

# get labels
y = manage_data.get_y_ready(total_df, labels)

x_first = np.copy(x_total)
x_second = np.copy(x_total)
x_third = np.copy(x_total)

# helper for the evaluation of verification performance
verification_indexes = vp.get_indexes(total_df)
# to keep track of all the evaluation metrics computed
metrics = eval_utils.get_metric_dict()

for epoch in range(num_epochs):
	print('--- Epoch {} ---'.format(epoch))
	print('First size: {} - Second size: {} - Third size: {}'.format(x_first.shape[1], x_second.shape[1], x_third.shape[1]))
	for ive_method, x in enumerate([x_first, x_second, x_third]):
		# train and test sets for the evaluation of soft biometrics
		x_train = x[:len(train_df)]
		y_train = y[:len(train_df)]
		x_test = x[len(train_df):]
		y_test = y[len(train_df):]

		scores = []
		for i, label in enumerate(labels):
			# compute the scores of every sb-classifier and store them
			score = simple_classifier.train_evaluate_simple_classifier(x_train, y_train[:, i], x_test, y_test[:, i])
			scores.append(score)
		print(scores)

		eer_verification = vp.evaluate_verification(verification_indexes, x)
		print(eer_verification)
		metrics = eval_utils.store_metrics(metrics, ive_method, scores, eer_verification)

	first_ive = ive_util.first_method(x_first, y, model_train, n_e, n_s, elimination_order)
	# print('first ive training done')
	second_ive = ive_util.second_method(x_second, y, model_train, num_eliminations, num_steps)
	# print('second ive training done')
	third_ive = ive_util.third_method(x_third, y, model_train, num_eliminations, num_steps)
	# print('third ive training done')

	x_first = first_ive[2].transform(first_ive[1].transform(first_ive[0].transform(x_first)))
	x_second = second_ive.transform(x_second)
	x_third = third_ive.transform(x_third)

eval_utils.plot_metrics(metrics)