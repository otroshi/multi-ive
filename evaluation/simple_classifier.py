from sklearn.neural_network import MLPClassifier


def train_evaluate_simple_classifier(x_train, y_train, x_test, y_test):
	# training
	clf = MLPClassifier(random_state=1, max_iter=300).fit(x_train, y_train)
	# evaluation
	return clf.score(x_test, y_test)
