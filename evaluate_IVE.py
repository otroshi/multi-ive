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
import evaluation.hyperparameter_finetuning as hyp_ft


def zeros_pca(x_total, mask):
	# store indexes where the mask is False.
	indexes = [i for i, m in enumerate(mask) if not m]
	# The features related to those indexes need to be zero.
	for ind in indexes:
		x_total[:, ind] = np.zeros((len(x_total)))
	return x_total


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


def execute_evaluation(db, classifiers, transform, seed, blocked_features=0):
	# fine-tune classifiers every N_EPOCHS_FT epochs, evaluate the algorithm every N_EPOCHS_EVAL epochs
	N_EPOCHS_FT = 25
	N_EPOCHS_EVAL = 5
	folder = str(seed) + ('_' + transform)
	if transform == 'pca' or transform == 'ica':
		folder = folder + '_k=' + str(blocked_features)

	# load scaler and PCA
	if transform == 'pca':
		pca = pk.load(open(os.path.join(os.path.join('results', folder), 'pca.pkl'), 'rb'))
	if transform == 'ica':
		ica = pk.load(open(os.path.join(os.path.join('results', folder), 'ica.pkl'), 'rb'))

	std_sc = pk.load(open(os.path.join(os.path.join('results', folder), 'std_sc.pkl'), 'rb'))

	# load masks
	masks = {'1': [], '2': [], '3': []}
	num_epochs = 170
	for i in range(1, 4):
		for epoch in range(num_epochs):
			filename = "{0:03}.npy".format(epoch)
			mask = np.load(os.path.join('results', folder, 'method' + str(i), filename))
			masks[str(i)].append(mask)

	if transform == 'pca' or transform == 'ica':
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

	dict_ft_classifiers = {'0': {}, '1': {}, '2': {}}

	for epoch in range(num_epochs):
		if (epoch % N_EPOCHS_EVAL) == 0:
			for ive_method, x in enumerate([x_first, x_second, x_third]):
				x_train = x[train_indexes]
				x_test = x[test_indexes]
				y_train = y[train_indexes]
				y_test = y[test_indexes]

				scores = []
				for i, label in enumerate(labels):

					# every n epochs finetune the classifiers
					if (epoch % N_EPOCHS_FT) == 0:
						all_ft_classifiers = hyp_ft.get_finetuned_classifiers(classifiers, seed, x_train, y_train[:, i])
						dict_ft_classifiers[str(ive_method)][label] = all_ft_classifiers
						print('Classifiers fine-tuned! Epoch ' + str(epoch) + ', label ' + label)

					# get the list of ft-classifiers specific for method and label
					ft_classifiers = dict_ft_classifiers[str(ive_method)][label]
					# compute the scores of every sb-classifier and store them
					score = simple_classifier.train_evaluate_classifier(x_train, y_train[:, i], x_test, y_test[:, i],
																		classifiers, ft_classifiers, seed)
					scores.append(score)
				txt = 'Epoch ' + str(epoch) + ' (Method ' + str(ive_method) + ') --> '
				for s, l in zip(scores, labels):
					# only to have an idea of the trend
					txt += l + ': ' + str(s[classifiers[0]]) + '---'

				if db == 'diveface' and ('mlp' in classifiers):
					eer_verification = vp.evaluate_verification(verification_indexes, x, seed)
				else:
					eer_verification = 'na'

				txt += 'EER: ' + str(eer_verification)
				print(txt + ('' if not ive_method == 2 else '\n'))
				metrics = eval_utils.store_metrics(metrics, ive_method, scores, eer_verification, db)

		if transform == 'pca':
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

		elif transform == 'ica':
			# transform in PCA
			x_first = ica.transform(x_first)
			x_second = ica.transform(x_second)
			x_third = ica.transform(x_third)

			# eliminate features in PCA
			x_first = zeros_pca(x_first, masks['1'][epoch])
			x_second = zeros_pca(x_second, masks['2'][epoch])
			x_third = zeros_pca(x_third, masks['3'][epoch])

			# transform from PCA to original domain
			x_first = ica.inverse_transform(x_first)
			x_second = ica.inverse_transform(x_second)
			x_third = ica.inverse_transform(x_third)

		else:
			x_first = x_first[:, masks['1'][epoch]]
			x_second = x_second[:, masks['2'][epoch]]
			x_third = x_third[:, masks['3'][epoch]]

	eval_utils.plot_metrics(metrics, folder, db, save_files=True)


execute_evaluation('diveface', ['mlp'], 'pca', 0, 0)
# for seed in range(10):
# 	execute_evaluation('utkface', ['mlp'], 'NO_pca', seed)
# 	for k in [0, 3, 5]:
# 		execute_evaluation('utkface', ['mlp'], 'pca', seed, k)
# 	execute_evaluation('utkface', ['mlp'], 'ica', seed)
#
# other_classifiers = ['svm_lin', 'et', 'log_reg']
# databases = ['diveface', 'utkface']
# for db in databases:
# 	for seed in range(10):
# 		execute_evaluation(db, other_classifiers, 'NO_pca', seed)
# 		for k in [0, 3, 5]:
# 			execute_evaluation(db, other_classifiers, 'pca', seed, k)
# 		execute_evaluation(db, other_classifiers, 'ica', seed)

# selected --> ['mlp', 'svm_lin', 'et', 'log_reg']
# all --> ['mlp', 'svm_lin', 'svm_rbf', 'rf', 'gb', 'nb', 'et', 'log_reg']

# other_classifiers = ['svm_lin', 'et', 'log_reg']
# databases = ['diveface', 'utkface']
# for seed in range(10):
# 	for db in databases:
# 		execute_evaluation(db, other_classifiers, 'NO_pca', seed)
# 		for k in [0, 3, 5]:
# 			execute_evaluation(db, other_classifiers, 'pca', seed, k)
# 		execute_evaluation(db, other_classifiers, 'ica', seed)