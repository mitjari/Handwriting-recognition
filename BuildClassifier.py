from __future__ import division, print_function, absolute_import
from pathlib import Path

#Import TFLEARN
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression

import tflearn.datasets.mnist as mnist

#Download MNIST dataset
#Pretvori v tensor primerne oblike -> slike so velikosti 28x28 x1(crno/bele)
X, Y, testX, testY = mnist.load_data(one_hot=True)
X = X.reshape([-1, 28, 28, 1])
testX = testX.reshape([-1, 28, 28, 1])

#Zgradi konvolucijsko mrezo
network = input_data(shape=[None, 28, 28, 1], name='input')					#Pripravi vhod (batch, height(32), width(32), chanals(1))
network = conv_2d(network, 32, 5, activation='relu', regularizer="L2")		#1. konvolucijska plast (32 filtrov size 5) -> 32 znacilnic
network = max_pool_2d(network, 2)											#2x downsample

network = conv_2d(network, 64, 5, activation='relu', regularizer="L2")		#2. konvolucijska plast -> 64 znacilnic
network = max_pool_2d(network, 2)											#2x downsample

network = fully_connected(network, 1024, activation='tanh')					#Fully connected 1024 nevronov
network = dropout(network, 0.4)												#Dropout -> zmanjsaj overfitting
network = fully_connected(network, 10, activation='softmax')				#Izhodna plast -> v 10 razredov
network = regression(network, optimizer='adam', learning_rate=0.01, loss='categorical_crossentropy', name='target')

#Preveri ali model ze obstaja
model = tflearn.DNN(network, tensorboard_verbose=0)
filename= "model/myModel.tflearn"
my_file = Path(filename + ".index")
if my_file.is_file():
	#Da -> preberi model
	print("------Model ze obstaja. Berem z diska------")
	model.load(filename)
else:
	#Ne-> natreniraj novega in shrani
	print("------Model ne obstaja. Treniram------")
	model.fit({'input': X}, {'target': Y}, n_epoch=10, validation_set=({'input': testX}, {'target': testY}), show_metric=True, run_id='convnet_mnist')
	model.save(filename)

#Oceni model
print("------Ocenjujem model------")

predictions= model.predict(testX)
pravilnih=0

for i in range(0, predictions.shape[0]):
	tempOrig= testY[i].tolist()
	tempPred= predictions[i].tolist()
	
	maxOrig= tempOrig.index(max(tempOrig))
	maxPred= tempPred.index(max(tempPred))
	
	if maxOrig == maxPred:
		pravilnih= pravilnih+1;

#Izpisi rez
print("Stevilo primerov " + repr(predictions.size))
print("Pravilno klasificiranih: " + repr(pravilnih))
print("Klasifikacijska tocnost: " + repr(pravilnih / predictions.shape[0]) )
