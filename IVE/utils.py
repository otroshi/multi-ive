import numpy as np
import os
from IVE.incremental_variable_elimination import IncrementalVariableElimination as IVE

def first_method(X, y, model_train, num_eliminations_perStep, num_steps, elimination_order, blocked_features):
	"""
		X: training data with the shape (n_samples, n_features)
			n_features is the number of features of the unprotected embedding
		y: labels of training data with the shape (n_samples, 3)
			for each sample labels are in the order: sex, age, ethnicity
		model_train: sklearn model that contains .feature_importance
		num_eliminations_perSteps: array of number of variable eliminations in each step: [n_s, n_a, n_e]
		num_steps: array of number of iterations of eliminations: [n_s, n_a, n_e]
		elimination_order: array that represent the order of variable elimination. Example: [0, 2, 1] means
		that the order of variable elimination is: sex, ethnicity, age
		blocked_features: in case of PCA, prevent the elimination of the first features

		num_steps and num_eliminations_perSteps have to be adjusted to the
		feature sizes. The order [sex, age, ethnicity] is used to set the values in those variables.
	"""

	trained_ive = []
	# e_o contains the index for y, num_eliminations_perStep, and num_steps
	for e_o in elimination_order:
		ive = IVE(model_train, num_eliminations_perStep[e_o], num_steps[e_o], blocked_features)
		ive.fit(X, y[:, e_o])
		trained_ive.append(ive)
		X = ive.transform(X)

	return trained_ive


def second_method(X, y, model_train, num_eliminations_perStep, num_steps, blocked_features):
	"""
		X: training data with the shape (n_samples, n_features)
			n_features is the number of features of the unprotected embedding
		y: labels of training data with the shape (n_samples, 3)
			for each sample labels are in the order: sex, age, ethnicity
		model_train: sklearn model that contains .feature_importance
		num_eliminations_perSteps: number of variable eliminations in each step
		num_steps: number of iterations of eliminations
		blocked_features: in case of PCA, prevent the elimination of the first features

		num_steps and num_eliminations_perSteps have to be adjusted to the
		feature sizes.
	"""

	# set a unique label given by the combination of sex, age, and ethnicity
	y = [str(yy[0]) + '_' + str(yy[1]) + '_' + str(yy[2]) for yy in y]
	ive = IVE(model_train, num_eliminations_perStep, num_steps, blocked_features)
	ive.fit(X, y)
	return ive


def third_method(X, y, model_train, num_eliminations_perStep, num_steps, blocked_features):
	ive = IVE(model_train, num_eliminations_perStep, num_steps, blocked_features)
	ive.fit(X, y, multi_var=True)  # compute feature importance for multiple variables
	return ive


def fix_first_mask(mask, method_id, epoch, save=False):
	if not method_id == 1:
		final_mask = mask[1]

	else:
		# hard coded: take the second mask from each level
		lev_1 = mask[0][1]
		lev_2 = mask[1][1]
		lev_3 = mask[2][1]

		# include the feature to remeve at lev1 in lev2
		lev_1_indexes = [i for i, m in enumerate(lev_1) if not m]
		for ind in lev_1_indexes:
			lev_2 = np.insert(lev_2, ind, False)

		# include the features to remove at lev2 in lev3
		lev_2_indexes = [i for i, m in enumerate(lev_2) if not m]
		for ind in lev_2_indexes:
			lev_3 = np.insert(lev_3, ind, False)

		final_mask = lev_3

	if save:
		folder = "results/method" + str(method_id)
		os.makedirs(folder, exist_ok=True)
		np.save(folder + "/{0:03}".format(epoch), final_mask)

	return final_mask
