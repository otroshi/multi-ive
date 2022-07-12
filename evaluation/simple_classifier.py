from sklearn.neural_network import MLPClassifier
import collections


def get_mask(y_labels, keep_labels, samples_for_label):
	# filter data in order to have balanced classes for training/evaluating sb-classifiers
	train_indexes = []
	for kl in keep_labels:
		# get indexes for each label
		l_indexes = [i for i, y in enumerate(y_labels) if y == kl]
		# take the first n indexes
		assert(len(l_indexes) >= samples_for_label)
		train_indexes += l_indexes[:samples_for_label]

	train_mask = [i in train_indexes for i in range(len(y_labels))]
	return train_mask


def balance_classes(x_train, y_train, x_test, y_test, MIN_SAMPLES=50):
	dict_labels = collections.Counter(y_train)
	# select labels with enough samples to keep
	keep_labels = [k for k, v in dict_labels.items() if v > MIN_SAMPLES]
	# get the smallest number of samples among the kept labels
	samples_for_label = min([dict_labels[k] for k in keep_labels])
	train_mask = get_mask(y_train, keep_labels, samples_for_label)

	dict_labels_test = collections.Counter(y_test)
	samples_for_label_test = min([dict_labels_test[k] for k in keep_labels])
	test_mask = get_mask(y_test, keep_labels, samples_for_label_test)

	return x_train[train_mask], y_train[train_mask], x_test[test_mask], y_test[test_mask]


def train_evaluate_simple_classifier(x_train, y_train, x_test, y_test):
	# balance data according to the labels
	x_train, y_train, x_test, y_test = balance_classes(x_train, y_train, x_test, y_test)
	# training
	clf = MLPClassifier(random_state=1, max_iter=300).fit(x_train, y_train)
	# evaluation
	return clf.score(x_test, y_test)
