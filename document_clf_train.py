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

def print_results(predicted, y_test):
	print("Classifier accuracy: {}".format(np.mean(predicted == y_test)))
	f3_scores = fbeta_score(y_test, predicted, average=None, beta=3)

	print(f"F-3 scores: {f3_scores}")
	print("Confusion matrix: \n{}".format(confusion_matrix(y_test, predicted)))
	print(f"Recalls: {recall(y_test, predicted)}")


def recall(y_true, y_pred):
    matrix = confusion_matrix(y_true, y_pred)
    scores = []
    for i in range(len(matrix)):
        row = matrix[i]
        correct = row[i]
        total = sum(row)
        scores.append(correct/total)
    return scores 

def fit(batch_size, epochs, callback, x_train, y_train, x_test, y_test_onehot, nn):
	return nn.fit(x_train, y_train,
								batch_size=batch_size,
								epochs=epochs,
								verbose=0,
								validation_data=(x_test, y_test_onehot),
								callbacks=[callback])

def document_clf(filepath):
    documents = load_files(filepath, shuffle=False)
    x_train, x_test, y_train, y_test = train_test_split(
        documents.data, documents.target, test_size=0.15
    )


    preprocessing = Pipeline([('count', HashingVectorizer(n_features=(2**12))),
							('tfidf', TfidfTransformer()),
						    ('pca', TruncatedSVD(n_components=430))])
    preprocessing.fit(x_train)
    f = open("./models/document-clf/preprocessing.pickle", "wb")
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

    y_train = np_utils.to_categorical(y_train)
    y_test_onehot = np_utils.to_categorical(y_test)


    stopped = early_stopping_callback.stopped_epoch

    history = fit(196, 500, early_stopping_callback, x_train, y_train, x_test, y_test_onehot, nn)



    predicted_vec = nn.predict(x_test)
    sensitive_probs = predicted_vec[:,2]
    y_test_sensitive = np.where(y_test == 2, 1, 0)
    indices = np.argsort(sensitive_probs)


    predicted = np.argmax(predicted_vec, axis=1)


    elapsed = time.time() - start
    print("Elapsed time:", elapsed)
    print(f"Stopped at epoch {stopped}")
    
    print_results(predicted, y_test)

    

    ## Organise documents by standard deviation
    #Divide each number in predicted_vec by the sum 
    sums = np.sum(predicted_vec, 1)
    sums = np.linalg.norm(predicted_vec, 1)
    divided = (predicted_vec.T/sums).T
    stds = np.std(divided, 1)
    indices = np.argsort(stds)

    save(nn)


import matplotlib.pyplot as plt
def show_confidence_graph():
	accuracies = []
	for i in range(len(stds[indices])):
		p = np.argmax(predicted_vec[indices][i:], 1)
		y = y_test[indices][i:]
		accuracies.append(np.mean(p == y))
	plt.plot(to_1_interval(accuracies), c='blue')
	plt.plot(to_1_interval(smooth(0.85, stds[indices])), c='orange')
	plt.show()
#show_confidence_graph()

	
def show_overfit_plot():
	plt.plot(history.history['loss'])
	plt.plot(history.history['val_loss'])
	plt.legend(['train','test'], loc='upper left')
	plt.show()

#show_overfit_plot()


### Save model configuration and weights ###
def save(nn):
	model_json = nn.to_json()
	with open("./models/document-clf/model.json", "w") as json_file:
		json_file.write(model_json)
	json_file.close()
	nn.save_weights("./models/document-clf/model.h5")
	










