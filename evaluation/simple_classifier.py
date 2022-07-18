from sklearn.neural_network import MLPClassifier
from sklearn import svm
import collections
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
import random


def get_sample_mask(y_labels, keep_labels, seed, samples_for_label=-1):
	random.seed(seed)
	# filter data in order to have balanced classes for training/evaluating sb-classifiers
	# samples_for_label = -1 means that all the samples for the label of interest are kept
	train_indexes = []
	for kl in keep_labels:
		# get indexes for each label
		l_indexes = [i for i, y in enumerate(y_labels) if y == kl]

		if samples_for_label == -1:
			# store all the indexes
			train_indexes += l_indexes
		else:
			# store n indexes
			assert(len(l_indexes) >= samples_for_label)
			random.shuffle(l_indexes)
			train_indexes += l_indexes[:samples_for_label]

	train_mask = [i in train_indexes for i in range(len(y_labels))]
	return train_mask


def filter_data(x_train, y_train, x_test, y_test, seed, MIN_SAMPLES=80):
	# data for training filtered to maintain all the samples of the most represented classes (at least MIN_SAMPLES)
	# data for test filtered to maintain the same number of samples for each label of interest

	dict_labels = collections.Counter(y_train)
	# select labels with enough samples to keep
	keep_labels = [k for k, v in dict_labels.items() if v > MIN_SAMPLES]
	# store all the samples for each of the selected labels
	train_mask = get_sample_mask(y_train, keep_labels, seed, -1)

	dict_labels_test = collections.Counter(y_test)
	# get the smallest number of samples among the kept labels
	samples_for_label_test = min([dict_labels_test[k] for k in keep_labels])
	# store 'sample_for_label_test' samples for each of the selected labels
	test_mask = get_sample_mask(y_test, keep_labels, seed, samples_for_label_test)

	return x_train[train_mask], y_train[train_mask], x_test[test_mask], y_test[test_mask]


def train_evaluate_classifier(x_train, y_train, x_test, y_test, classifiers, seed):
	# filter data according to the explanation of the function filter_data
	x_train, y_train, x_test, y_test = filter_data(x_train, y_train, x_test, y_test, seed)

	# balance training data
	oversample = SMOTE(random_state=seed)
	x_train, y_train = oversample.fit_resample(x_train, y_train)

	scores = {}
	for c in classifiers:
		# training
		clf = MLPClassifier(random_state=seed, max_iter=10000, learning_rate='adaptive') if c == 'mlp' else \
			svm.SVC(random_state=seed) if c == 'svm' else \
			RandomForestClassifier(n_estimators=30, random_state=seed) if c == 'rf' else None
		clf.fit(x_train, y_train)
		# clf = MLPClassifier(random_state=seed, early_stopping=True).fit(x_train, y_train)
		# clf = svm.SVC(random_state=seed).fit(x_train, y_train)
		# clf = RandomForestClassifier(n_estimators=30, random_state=seed).fit(x_train, y_train)
		# clf = GaussianNB().fit(x_train, y_train)
		# clf = LogisticRegression().fit(x_train, y_train)

		# evaluation
		score = clf.score(x_test, y_test)
		scores[c] = score

	# to simplify for the moment: if using only a classifier it returns the scores in the format accepted by the main
	return scores if len(classifiers) > 1 else scores[classifiers[0]]
