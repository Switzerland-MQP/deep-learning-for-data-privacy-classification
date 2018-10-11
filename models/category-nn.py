import time
start = time.time()

from sklearn.datasets import load_files

from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import Pipeline

from sklearn.metrics import fbeta_score, confusion_matrix, recall_score


from keras.models import Sequential, Model
from keras.layers import Dense, Dropout
from keras.regularizers import l1
from keras.callbacks import EarlyStopping
import utils

from keras import backend as K

import numpy as np
import pickle

from keras.utils import np_utils

#utils.personal_categories_dict (.inv)


documents = utils.load_dirs_custom([
    '../../anondata_lines/sensitive',
    '../../anondata_lines/personal',
    '../../anondata_lines/nonpersonal'
], individual=True)

x = []
y = []
for document in documents:
	lines = document.lines
	categories = []
	for line in lines:
		for category in line.categories:
			if category not in categories:
				categories.append(category)
	x += ['\n'.join(document.data)]
	y += [categories]

y_encoded = []
for categories in y:
    one_hot_encoded = np.zeros(len(utils.all_categories_dict.keys()))
    for category in categories:
        i = utils.all_categories_dict.inv[category]
        one_hot_encoded[i] = 1
    y_encoded += [one_hot_encoded]
y = y_encoded


x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.15, shuffle=True
)

x_train = np.array(x_train)
x_test = np.array(x_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


preprocessing = Pipeline([('count', HashingVectorizer(n_features=(2**12))),
												  ('tfidf', TfidfTransformer()),
													('pca', TruncatedSVD(n_components=430))])
preprocessing.fit(x_train)
f = open("./category-clf/preprocessing.pickle", "wb")
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
nn.add(Dense(y[0].shape[0],  activation='sigmoid', name="out_layer"))
nn.compile(loss= 'categorical_crossentropy',
           optimizer='adam',
           metrics=['acc'])

print("Begin fitting network")

def fit(batch_size, epochs):
	global x_train, y_train, x_test, y_test
	return nn.fit(x_train, y_train,
								batch_size=batch_size,
								epochs=epochs,
								verbose=0,
								validation_data=(x_test, y_test),
								callbacks=[early_stopping_callback])

print("Fitting network")
history = fit(196, 500)
stopped = early_stopping_callback.stopped_epoch
print(f"Stopped at epoch {stopped}")


predicted = nn.predict(x_test)

import matplotlib.pyplot as plt

def cutoff_graph(predicted_category, y_category):
	#cutoffs = np.linspace(min(predicted_category), max(predicted_category), 10)
	indices = np.argsort(predicted_category)

	negatives = []
	positives = []
	for cutoff in predicted_category[indices]:
		predicted = np.where(predicted_category > cutoff, 1, 0)
		score = recall_score(predicted, y_category, average=None)
		try:
			negatives.append(score[0])
			positives.append(score[1])
			if score[1] > 0.5:
				return cutoff
		except:
			print("sdfsdfsdfds")		

	#plt.plot(negatives, c='blue')
	#plt.plot(positives, c='yellow')
	cutoffs = predicted_category[indices]
	#plt.xticks([i for i in range(len(cutoffs)) if i % 50 == 0], [cutoffs[i] for i in range(len(cutoffs)) if i % 50 == 0])
	#plt.show()
	
	return cutoffs[5]


cutoff_dict = {}
for i in range(len(utils.all_categories_dict.keys()) -1):
	category = utils.all_categories_dict[i+1]
	predicted_category = predicted[:,i+1]
	y_category = y_test[:,i+1]

	cutoff = cutoff_graph(predicted_category, y_category)
	cutoff_dict[category] = cutoff

	predicted_category = np.where(predicted_category > cutoff, 1, 0)
	#score = fbeta_score(predicted_category, y_category, average=None, beta=3)
	score = np.mean(y_category == predicted_category)
	f2 = fbeta_score(y_category, predicted_category, average=None, beta=2)
	print(f"Accuracy for category: {category} : {score} -- f2: {f2}")
	print(confusion_matrix(y_category, predicted_category))	

f = open("./category-clf/cutoffs.pickle", "wb")
f.write(pickle.dumps(cutoff_dict))
f.close()
print("Finished dumping cutoff pickle")



elapsed = time.time() - start
print("Elapsed time:", elapsed)
print(f"Stopped at epoch {stopped}")

def print_results(predicted, y_test):
	f2_scores = fbeta_score(y_test, predicted, average=None, beta=2)

	print("F-2 scores: {}  | Average: {}".format(f2_scores, np.mean(f2_scores)))

	print("Confusion matrix: \n{}".format(confusion_matrix(y_test, predicted)))

#print_results(predicted_vec, y_test)

def show_overfit_plot():
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.legend(['train','test'], loc='upper left')
	plt.show()

#show_overfit_plot()



### Save model configuration and weights ###
def save():
	model_json = nn.to_json()
	with open("./category-clf/model.json", "w") as json_file:
		json_file.write(model_json)
	json_file.close()
	nn.save_weights("./category-clf/model.h5")

save()

