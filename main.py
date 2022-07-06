import os
import data.manage_data as manage_data
import pandas as pd
import evaluation.simple_classifier as simple_classifier
import IVE.util as ive_util
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler


# create
if not os.path.isfile('data/train_df.csv'):
	train_df, test_df = manage_data.get_training_test_df(save_files=True)
else:
	train_df = pd.read_csv('data/train_df.csv')
	test_df = pd.read_csv('data/test_df.csv')

# set useful variables
labels = ['sex', 'age', 'ethnicity']
# train_df contains 4 columns more than the number of features
length_embeddings = train_df.shape[1] - 4
# number of steps and number of eliminations per step have to be adjusted depending on the feature size
num_steps = 1
num_eliminations = 5
num_epochs = 100
# define classifier
model_train = RandomForestClassifier(n_estimators=30)
# set parameters for method one
n_s = [num_steps, num_steps, num_steps]
n_e = [int(num_eliminations / 3), int(num_eliminations / 3), int(num_eliminations / 3)]
elimination_order = [0, 1, 2]

# get embeddings
x_train = manage_data.get_x_ready(train_df, length_embeddings)
x_test = manage_data.get_x_ready(test_df, length_embeddings)

# normalize embeddings
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# get labels
y_train = manage_data.get_y_ready(train_df, labels)
y_test = manage_data.get_y_ready(test_df, labels)

x_train_first = x_train
x_train_second = x_train
x_train_third = x_train
x_test_first = x_test
x_test_second = x_test
x_test_third = x_test

for epoch in range(num_epochs):
	print('--- {} ---'.format(epoch))
	print('First size: {} - Second size: {} - Third size: {}'.format(x_train_first.shape[1],
																	 x_train_second.shape[1], x_train_third.shape[1]))
	for x_train, x_test in zip([x_train_first, x_train_second, x_train_third], [x_test_first, x_test_second, x_test_third]):
		scores = []
		for i, label in enumerate(labels):
			# compute the scores of every sb-classifier and store them
			score = simple_classifier.train_evaluate_simple_classifier(x_train, y_train[:, i], x_test, y_test[:, i])
			scores.append(score)

		print(scores)

	first_ive = ive_util.first_method(x_train_first, y_train, model_train, n_e, n_s, elimination_order)
	# print('first ive training done')
	second_ive = ive_util.second_method(x_train_second, y_train, model_train, num_eliminations, num_steps)
	# print('second ive training done')
	third_ive = ive_util.third_method(x_train_third, y_train, model_train, num_eliminations, num_steps)
	# print('third ive training done')

	x_train_first = first_ive[2].transform(first_ive[1].transform(first_ive[0].transform(x_train_first)))
	x_train_second = second_ive.transform(x_train_second)
	x_train_third = third_ive.transform(x_train_third)
	x_test_first = first_ive[2].transform(first_ive[1].transform(first_ive[0].transform(x_test_first)))
	x_test_second = second_ive.transform(x_test_second)
	x_test_third = third_ive.transform(x_test_third)


