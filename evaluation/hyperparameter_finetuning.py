from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression


def finetuning_estimators(classifier, seed, x_train, y_train):
	params = {'n_estimators': (100, 500, 1000)}
	cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=seed)

	if classifier == 'rf':
		search = GridSearchCV(estimator=RandomForestClassifier(random_state=seed), param_grid=params, n_jobs=-1, cv=cv)
	if classifier == 'gb':
		search = GridSearchCV(estimator=GradientBoostingClassifier(n_iter_no_change=10, random_state=seed),
							  param_grid=params, n_jobs=-1, cv=cv)
	if classifier == 'et':
		search = GridSearchCV(estimator=ExtraTreesClassifier(random_state=seed), param_grid=params, n_jobs=-1, cv=cv)

	search.fit(x_train, y_train)
	print(search.best_score_)
	print(search.best_params_)

	n_est = search.best_params_['n_estimators']

	if classifier == 'rf':
		return RandomForestClassifier(random_state=seed, n_estimators=n_est)
	if classifier == 'gb':
		return GradientBoostingClassifier(n_iter_no_change=10, random_state=seed, n_estimators=n_est)
	if classifier == 'et':
		return ExtraTreesClassifier(random_state=seed, n_estimators=n_est)


def finetuning_hidden(seed, x_train, y_train):
	params = {'hidden_layer_sizes': ((100, ), (500, ), (1000, ))}
	cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=seed)

	search = GridSearchCV(estimator=MLPClassifier(random_state=seed, early_stopping=True, learning_rate='adaptive'),
						  param_grid=params, n_jobs=-1, cv=cv)

	search.fit(x_train, y_train)
	print(search.best_score_)
	print(search.best_params_)

	hls = search.best_params_['hidden_layer_sizes']

	return MLPClassifier(random_state=seed, early_stopping=True, learning_rate='adaptive', hidden_layer_sizes=hls)


def finetuning_C(classifier, seed, x_train, y_train):
	params = {'C': (0.1, 1.0, 2.0)}
	cv = RepeatedStratifiedKFold(n_splits=3, n_repeats=1, random_state=seed)

	if classifier == 'svm_lin':
		search = GridSearchCV(estimator=SVC(random_state=seed, kernel='linear'), param_grid=params, n_jobs=-1, cv=cv)
	if classifier == 'svm_rbf':
		search = GridSearchCV(estimator=SVC(random_state=seed, kernel='rbf'), param_grid=params, n_jobs=-1, cv=cv)
	if classifier == 'log_reg':
		search = GridSearchCV(estimator=LogisticRegression(random_state=seed, max_iter=10000),
							  param_grid=params, n_jobs=-1, cv=cv)

	search.fit(x_train, y_train)
	print(search.best_score_)
	print(search.best_params_)

	c = search.best_params_['C']

	if classifier == 'svm_lin':
		return SVC(random_state=seed, kernel='linear', C=c)
	if classifier == 'svm_rbf':
		return SVC(random_state=seed, kernel='rbf', C=c)
	if classifier == 'log_reg':
		return LogisticRegression(random_state=seed, max_iter=10000, C=c)
