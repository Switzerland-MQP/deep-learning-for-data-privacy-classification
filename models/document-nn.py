import time
start = time.time()

from sklearn.datasets import load_files

from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

from sklearn.metrics import fbeta_score, confusion_matrix 


from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
from keras.regularizers import l1
from keras.callbacks import EarlyStopping

from keras import backend as K

import numpy as np
import pickle

from keras.utils import np_utils

documents = load_files('../../anondata/', shuffle=False)
x_train, x_test, y_train, y_test = train_test_split(
    documents.data, documents.target, test_size=0.15
)

x_test_copy = x_test.copy()

preprocessing = Pipeline([('count', HashingVectorizer(n_features=(2**12))),
												  ('tfidf', TfidfTransformer()),
													('pca', TruncatedSVD(n_components=430))])
preprocessing.fit(x_train)
f = open("./document-clf/preprocessing.pickle", "wb")
f.write(pickle.dumps(preprocessing))
f.close()
print("Finished dumping pickle")
x_train, x_test = (preprocessing.transform(x_train), preprocessing.transform(x_test))
print("Finished data preprocessing - {} elapsed".format(time.time()-start))



early_stopping_callback = EarlyStopping(monitor='val_loss',
				min_delta=0, patience=12, verbose=0, mode='auto')


input_shape = x_train.shape[1]
nn = Sequential()
nn.add(Dense(128, activation='relu', input_shape=(input_shape,)))
nn.add(Dropout(0.25))
nn.add(Dense(32, activation='relu'))
nn.add(Dropout(0.25))
nn.add(Dense(3,  activation='sigmoid', name="out_layer"))
nn.compile(loss= 'categorical_crossentropy',
           optimizer='adam',
           metrics=['mean_squared_logarithmic_error'])

print("Begin fitting network")

def recall(y_true, y_pred):
	matrix = confusion_matrix(y_true, y_pred)
	scores = []
	for i in range(len(matrix)):
			row = matrix[i]
			correct = row[i]
			total = sum(row)
			scores.append(correct/total)
	return scores 


y_train = np_utils.to_categorical(y_train)
y_test_onehot = np_utils.to_categorical(y_test)
def to_independent_categorical(y):
	y_categorical = []
	for c in y:
		if c == 2:
			y_categorical.append([1, 1, 1])
		elif c == 1:
			y_categorical.append([1, 1, 0])
		elif c == 0:
			y_categorical.append([1, 0, 0])
	return y_categorical
#y_train = np.array(to_independent_categorical(y_train))
#y_test_onehot = np.array(to_independent_categorical(y_test))


def fit(batch_size, epochs):
	global x_train, y_train, x_test, y_test_onehot
	return nn.fit(x_train, y_train,
								batch_size=batch_size,
								epochs=epochs,
								verbose=0,
								validation_data=(x_test, y_test_onehot),
								callbacks=[early_stopping_callback])
stopped = early_stopping_callback.stopped_epoch

history = fit(196, 500)



predicted_vec = nn.predict(x_test)
sensitive_probs = predicted_vec[:,2]
y_test_sensitive = np.where(y_test == 2, 1, 0)
indices = np.argsort(sensitive_probs)
import matplotlib.pyplot as plt
def show_cut_off():
	accuracies = []
	nonpersonal = []
	labels = []
	for i in range(len(sensitive_probs)):
		y_pred = np.where(sensitive_probs[indices] >= sensitive_probs[indices][i], 1, 0)
		score = fbeta_score(y_pred, y_test_sensitive, average=None, beta=2)
		#score = np.mean(y_pred == y_test_sensitive)
		accuracies.append(score[1])
		nonpersonal.append(score[0])
		labels.append(sensitive_probs[indices][i])
	plt.plot(accuracies, c='blue')
	plt.plot(nonpersonal, c='orange')
	#plt.plot([(accuracies[i] + nonpersonal[i])/2 for i in range(len(accuracies))], c='green')
	indices_ = [i for i in range(len(labels)) if i % 50 == 0]
	plt.xticks(indices_, [labels[i] for i in indices_])
	plt.show()
#show_cut_off()

predicted = np.argmax(predicted_vec, axis=1)
"""
predicted = []
for probs in predicted_vec:
	if probs[2] > 0.01:
		predicted += [2]
		continue
	if probs[1] > 0.006:
		predicted += [1]
		continue
	predicted += [0]
"""


elapsed = time.time() - start
print("Elapsed time:", elapsed)
print(f"Stopped at epoch {stopped}")

def print_results(predicted, y_test):
	print("Classifier accuracy: {}".format(np.mean(predicted == y_test)))
	f3_scores = fbeta_score(y_test, predicted, average=None, beta=3)

	print(f"F-3 scores: {f3_scores}")
	print("Confusion matrix: \n{}".format(confusion_matrix(y_test, predicted)))
	print(f"Recalls: {recall(y_test, predicted)}")

print_results(predicted, y_test)

## Organise documents by standard deviation
#Divide each number in predicted_vec by the sum 
sums = np.sum(predicted_vec, 1)
sums = np.linalg.norm(predicted_vec, 1)
divided = (predicted_vec.T/sums).T
stds = np.std(divided, 1)
indices = np.argsort(stds)

def smooth(keep, arr):
	smoothed = []
	previous = arr[0]
	for i in range(len(arr)):
		previous = keep*previous + (1-keep)*arr[i]
		smoothed.append(previous)
	return smoothed

def to_1_interval(arr):
	minimum = min(arr)
	maximum = max(arr)
	new_arr = []
	for value in arr:
		new_arr.append((value-minimum)/(maximum-minimum))
	return new_arr

	
import matplotlib.pyplot as plt

smoothed = smooth(0.96, stds[indices])
for i in range(len(smoothed)-1):
	delta = smoothed[i+1] - smoothed[i]
	if delta < 0.00001 and i > 20:
		n = i
		print(f"delta found! {delta} -- n: {n}")
		break

def show_confidence_graph():
	accuracies = []
	for i in range(len(stds[indices])):
		p = np.argmax(predicted_vec[indices][i:], 1)
		y = y_test[indices][i:]
		accuracies.append(np.mean(p == y))
	#plt.plot(to_1_interval(smooth(0.96, accuracies)), c='blue')
	plt.plot(to_1_interval(accuracies), c='blue')
	plt.plot(to_1_interval(smooth(0.85, stds[indices])), c='orange')
	#plt.scatter([n], [to_1_interval(smooth(0.90, accuracies))[n]])
	plt.show()

#show_confidence_graph()

n = 55
predicted_vec = predicted_vec[indices][n:]
y_test_high_confidence = y_test[indices][n:]

predicted_high_confidence = np.argmax(predicted_vec, 1)
print("High confidence results:")
print_results(predicted_high_confidence,  y_test_high_confidence)


####################################
	
def show_overfit_plot():
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.legend(['train','test'], loc='upper left')
	plt.show()

#show_overfit_plot()

def show_variance_plot():
	explained = preprocessing.named_steps['pca'].explained_variance_
	cumulative = [np.sum(explained[:i]) for i in range(len(explained))]
	#plt.plot(explained)
	plt.plot(cumulative)
	plt.legend(['explained variance','cumulative explained variance'], loc='upper left')
	plt.show()



### Save model configuration and weights ###
def save():
	model_json = nn.to_json()
	with open("./document-clf/model.json", "w") as json_file:
		json_file.write(model_json)
	json_file.close()
	nn.save_weights("./document-clf/model.h5")
	
save()









