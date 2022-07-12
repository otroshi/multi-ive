from IVE.incremental_variable_elimination import IncrementalVariableElimination as IVE

def first_method(X, y, model_train, num_eliminations_perStep, num_steps, elimination_order):
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

		num_steps and num_eliminations_perSteps have to be adjusted to the
		feature sizes. The order [sex, age, ethnicity] is used to set the values in those variables.
	"""

	trained_ive = []
	# e_o contains the index for y, num_eliminations_perStep, and num_steps
	for e_o in elimination_order:
		ive = IVE(model_train, num_eliminations_perStep[e_o], num_steps[e_o])
		ive.fit(X, y[:, e_o])
		trained_ive.append(ive)
		X = ive.transform(X)

	return trained_ive


def second_method(X, y, model_train, num_eliminations_perStep, num_steps):
	"""
		X: training data with the shape (n_samples, n_features)
			n_features is the number of features of the unprotected embedding
		y: labels of training data with the shape (n_samples, 3)
			for each sample labels are in the order: sex, age, ethnicity
		model_train: sklearn model that contains .feature_importance
		num_eliminations_perSteps: number of variable eliminations in each step
		num_steps: number of iterations of eliminations

		num_steps and num_eliminations_perSteps have to be adjusted to the
		feature sizes.
	"""

	# set a unique label given by the combination of sex, age, and ethnicity
	y = [str(yy[0]) + '_' + str(yy[1]) + '_' + str(yy[2]) for yy in y]
	ive = IVE(model_train, num_eliminations_perStep, num_steps)
	ive.fit(X, y)
	return ive


def third_method(X, y, model_train, num_eliminations_perStep, num_steps):
	ive = IVE(model_train, num_eliminations_perStep, num_steps)
	ive.fit(X, y, multi_var=True)  # compute feature importance for multiple variables
	return ive

