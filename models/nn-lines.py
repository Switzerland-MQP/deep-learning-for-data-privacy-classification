import numpy as np
import time
start = time.time()


#  from scipy.stats import randint as sp_randint
from sklearn.decomposition import TruncatedSVD
from scipy.stats import randint as sp_randint

from sklearn.ensemble import RandomForestClassifier

from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import confusion_matrix, fbeta_score

from sklearn.pipeline import Pipeline
import utils as utils

from keras.utils import np_utils
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
from keras.regularizers import l1
import keras.backend as K
import pickle
import os

import matplotlib.pyplot as plt

def ensure_directory(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)

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


def show_overfit_plot():
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.legend(['train','test'], loc='upper left')
	plt.show()


documents = utils.load_dirs_custom([
    '../../anondata_lines/sensitive',
    '../../anondata_lines/personal',
    '../../anondata_lines/nonpersonal'
], individual=True)


documents = utils.n_gram_documents_range(documents, 8, 8)

doc_train, doc_test, = utils.document_test_train_split(
    documents, 0.05
)

print("Doc train: ", len(doc_train))
print("Doc test: ", len(doc_test))
x_train, y_train = utils.convert_docs_to_lines(doc_train)
x_test, y_test = utils.convert_docs_to_lines(doc_test)

y_train = np.where((y_train == 2) | (y_train == 1), 1, 0)
y_test = np.where((y_test == 2) | (y_test == 1), 1, 0)

print("Convert lines timing {}".format(time.time() - start))

preprocessor = Pipeline([('vect', HashingVectorizer(n_features=(2**12))),
                     ('tfidf', TfidfTransformer()),
                     ('pca', TruncatedSVD(n_components=430))])
preprocessor.fit(x_train)

ensure_directory('line-clf')
f = open("./line-clf/preprocessing.pickle", "wb")

f.write(pickle.dumps(preprocessor))
f.close()
print("Finished dumping preprocessing pickle")

x_train, x_test = (preprocessor.transform(x_train), preprocessor.transform(x_test))
print("Finished data preprocessing - {} elapsed".format(time.time()-start))
	
def create_model():
	input_shape = x_train.shape[1]

	nn = Sequential()
	nn.add(Dense(32, activation='relu', input_shape=(input_shape,)))
	nn.add(Dropout(0.25))
	#nn.add(Dense(8, activation='relu'))
	#nn.add(Dropout(0.5))
	nn.add(Dense(1,  activation='sigmoid', name="out_layer"))
	#nn.compile(loss= 'categorical_crossentropy',
	nn.compile(loss='binary_crossentropy', optimizer='adam')
	return nn

print("Begin fitting network")
nn = create_model()

#y_train = np_utils.to_categorical(y_train)
#y_test_onehot = np_utils.to_categorical(y_test)

def fit(batch_size, epochs):
	global x_train, y_train, x_test, y_test
	return nn.fit(x_train, y_train,
								batch_size=batch_size,
								epochs=epochs,
								verbose=2,
								validation_data=(x_test, y_test))

history = fit(1000, 30)



elapsed = time.time() - start
print("Elapsed time:", elapsed)

show_overfit_plot()



documents_predicted = []
documents_target = []
all_predicted_lines = []
all_target_lines = []
document_confidences = []
for doc in doc_test:
    if np.all(doc.targets == 0):
        continue
    feature_vectors = preprocessor.transform(doc.data)
    predicted_lines = nn.predict(feature_vectors)

    predicted_lines_confs = np.array([x for x in map(lambda x: x[0], list(predicted_lines))])
    document_confidence = np.mean(predicted_lines_confs)
    document_confidences.append(document_confidence)


    all_predicted_lines += list(predicted_lines)
    doc.targets = np.where((doc.targets == 2) | (doc.targets == 1), 1, 0)
    all_target_lines += list(doc.targets)


    predicted_doc = utils.classify_doc(predicted_lines)
    documents_predicted.append(predicted_doc)
    documents_target.append(doc.category)

sorted_confidences = np.sort(np.array(document_confidences))
plt.plot(sorted_confidences)
plt.show()

all_predicted_lines = np.array([x for x in map(lambda x: x[0], all_predicted_lines)])
predicted = all_predicted_lines.copy()
all_predicted_lines = np.where(all_predicted_lines >= 0.5, 1, 0)

print("Line by Line ")
print(f"Accuracy: {np.mean(all_predicted_lines == all_target_lines)}")
print(f"F2 scores: {fbeta_score(all_predicted_lines, all_target_lines, average=None, beta=2)}")

print("Confusion Matrix: \n{}".format(
    confusion_matrix(all_target_lines, all_predicted_lines)
))


confidences = (0.5 - predicted) ** 2
indices = np.argsort(confidences)

	

smoothed = smooth(0.96, confidences[indices])
for i in range(len(smoothed)-1):
	delta = smoothed[i+1] - smoothed[i]
	if delta < 0.001 and i > 5:
		n = i
		print(f"delta found! {delta} -- n: {n}")
		break


all_target_lines = np.array(all_target_lines)

accuracies = []
nonpersonal = []
for i in range(len(confidences[indices])):
	p = predicted[indices][i:]
	p = np.where(p >= 0.5, 1, 0)	

	y = all_target_lines[indices][i:]
	f2_score = fbeta_score(p, y, average=None, beta=2)
	try:
		nonpersonal.append(f2_score[0])
		accuracies.append(f2_score[1])
	except:
		pass	

n = 3500

predicted = np.where(predicted[indices][n:] >= 0.5, 1, 0)
y = all_target_lines[indices][n:]
f2 = fbeta_score(predicted, y, average=None, beta=2)
print(f"F2-scores for lines above {n}: {f2}")



plt.plot(to_1_interval(smooth(0.92, accuracies)))
plt.plot(to_1_interval(smooth(0.92, confidences[indices])))

#plt.scatter([n], [to_1_interval(smooth(0.96, confidences[indices]))[n]])
plt.show()



### Save model configuration and weights ###
def save():
	model_json = nn.to_json()
	with open("./line-clf/model.json", "w") as json_file:
		json_file.write(model_json)
	json_file.close()
	nn.save_weights("./line-clf/model.h5")

save()


