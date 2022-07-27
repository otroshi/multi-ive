import pickle as pk
import os
import numpy as np
import evaluation.load_diveface as load_diveface
import evaluation.load_UTKFace as load_UTKFace
import evaluation.utils as eval_utils
import evaluation.simple_classifier as simple_classifier
import evaluation.verification_performance as vp
import pandas as pd
import random


def zeros_pca(x_total, mask):
	# store indexes where the mask is False.
	indexes = [i for i, m in enumerate(mask) if not m]
	# The features related to those indexes need to be zero.
	for ind in indexes:
		x_total[:, ind] = np.zeros((len(x_total)))
	return x_total


def create_random_masks(seed, k, length_embedding):
	random.seed(seed)
	masks = {'1': [], '2': [], '3': []}
	num_epochs = 170
	for i in range(1, 4):
		indexes = [i for i in range(k, length_embedding)]
		random.shuffle(indexes)
		for epoch in range(num_epochs):
			ind_false = indexes[:(epoch+1)*3]
			mask = np.array([ix not in ind_false for ix in range(length_embedding)])
			masks[str(i)].append(mask)
	return masks


def fix_pca_masks(masks):
	new_masks = {'1': [], '2': [], '3': []}
	for i in range(1, 4):
		for epoch in range(len(masks[str(i)])):
			# first epoch does not need any fix
			if len(new_masks[str(i)]) == 0:
				new_masks[str(i)].append(masks[str(i)][0])

			else:
				prev_mask = new_masks[str(i)][-1]
				prev_indexes = [i for i, m in enumerate(prev_mask) if not m]
				curr_mask = masks[str(i)][epoch]
				# fix the indexes removed at the previous iteration in the new mask to store
				for ix in prev_indexes:
					curr_mask = np.insert(curr_mask, ix, False)
				new_masks[str(i)].append(curr_mask)

	return new_masks


def execute_evaluation(db, classifiers, use_pca, seed, blocked_pca_features=0):
	folder = str(seed) + ('_pca' if use_pca else '_NO_pca')

	# load scaler and PCA
	if use_pca:
		folder = folder + '_k=' + str(blocked_pca_features)
		pca = pk.load(open(os.path.join(os.path.join('results', folder), 'pca.pkl'), 'rb'))
	std_sc = pk.load(open(os.path.join(os.path.join('results', folder), 'std_sc.pkl'), 'rb'))

	# load masks
	masks = {'1': [], '2': [], '3': []}
	num_epochs = 170
	for i in range(1, 4):
		for epoch in range(num_epochs):
			filename = "{0:03}.npy".format(epoch)
			mask = np.load(os.path.join('results', folder, 'method' + str(i), filename))
			masks[str(i)].append(mask)

	if use_pca:
		masks = fix_pca_masks(masks)

	# load data for evaluation
	if db == 'diveface':
		# create
		if not os.path.isfile('data/diveface_df.csv'):
			diveface_df, length_embeddings = load_diveface.get_diveface_df(r'data\diveface_embeddings', seed,
																		   save_files=True, limit_size=True)
		else:
			diveface_df = pd.read_csv('data/diveface_df.csv')
			length_embeddings = diveface_df.shape[1] - 5

		train_indexes, test_indexes = load_diveface.get_sb_train_test_indexes(diveface_df, seed, length_embeddings)
		verification_indexes = load_diveface.get_verification_indexes(diveface_df, seed, length_embeddings)

		# get data ready to perform evaluation of SB and verification
		x_total = load_diveface.get_x_ready(diveface_df, length_embeddings)
		labels = eval_utils.get_labels(db)
		y = load_diveface.get_y_ready(diveface_df, labels)

	elif db == 'utkface':
		# create
		if not os.path.isfile('data/utkface_df.csv'):
			utkface_df, length_embeddings = load_UTKFace.get_utkface_df(r'data\utkface_embeddings', seed,
																		   save_files=True, limit_size=True)
		else:
			utkface_df = pd.read_csv('data/utkface_df.csv')
			length_embeddings = utkface_df.shape[1] - 2

		train_indexes, test_indexes = load_UTKFace.get_sb_train_test_indexes(utkface_df, seed, length_embeddings)

		# get data ready to perform evaluation of SB and verification
		x_total = load_UTKFace.get_x_ready(utkface_df, length_embeddings)
		labels = eval_utils.get_labels(db)
		y = load_UTKFace.get_y_ready(utkface_df, labels)

	x_total = std_sc.transform(x_total)
	x_first = np.copy(x_total)
	x_second = np.copy(x_total)
	x_third = np.copy(x_total)

	metrics = eval_utils.get_metric_dict(classifiers, db)

	for epoch in range(num_epochs):
		for ive_method, x in enumerate([x_first, x_second, x_third]):
			x_train = x[train_indexes]
			x_test = x[test_indexes]
			y_train = y[train_indexes]
			y_test = y[test_indexes]

			scores = []
			for i, label in enumerate(labels):
				# compute the scores of every sb-classifier and store them
				score = simple_classifier.train_evaluate_classifier(x_train, y_train[:, i], x_test, y_test[:, i],
																	classifiers, seed)
				scores.append(score)
			txt = 'Epoch ' + str(epoch) + ' (Method ' + str(ive_method) + ') --> '
			for s, l in zip(scores, labels):
				# only to have an idea of the trend
				txt += l + ': ' + str(s[classifiers[0]]) + '---'

			eer_verification = vp.evaluate_verification(verification_indexes, x, seed) if db == 'diveface' else 'na'
			txt += 'EER: ' + str(eer_verification)
			print(txt + ('' if not ive_method == 2 else '\n'))
			metrics = eval_utils.store_metrics(metrics, ive_method, scores, eer_verification, db)

		if use_pca:
			# transform in PCA
			x_first = pca.transform(x_first)
			x_second = pca.transform(x_second)
			x_third = pca.transform(x_third)

			# eliminate features in PCA
			x_first = zeros_pca(x_first, masks['1'][epoch])
			x_second = zeros_pca(x_second, masks['2'][epoch])
			x_third = zeros_pca(x_third, masks['3'][epoch])

			# transform from PCA to original domain
			x_first = pca.inverse_transform(x_first)
			x_second = pca.inverse_transform(x_second)
			x_third = pca.inverse_transform(x_third)

		else:
			x_first = x_first[:, masks['1'][epoch]]
			x_second = x_second[:, masks['2'][epoch]]
			x_third = x_third[:, masks['3'][epoch]]

	eval_utils.plot_metrics(metrics, folder, db, save_files=True)


execute_evaluation('diveface', ['mlp'], True, 0, 3)
# ['mlp', 'svm_lin', 'svm_rbf', 'rf', 'gb', 'nb', 'et', 'log_reg']
