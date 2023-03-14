import numpy as np
import pandas as pd

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger
from datetime import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, load_model

basepath = "./keras_log/"

class BaseModel:
	def __init__(self, name):
		self.RUN_NAME = name + str(datetime.utcnow())
		self.logger = keras.callbacks.TensorBoard(log_dir="logs/"+self.RUN_NAME+"/", write_graph=True)
		self.csv_log_file = basepath+self.name+"_training_log.csv"
		self.csv_logger = CSVLogger(self.csv_log_file)
		

	# Utility functions to save and load trained models
	def save(self,suffix=""):
		self.model.save(basepath+self.name+suffix+".h5")

	def load(self,suffix=""):
		self.model = load_model(basepath+self.name+suffix+".h5")

	


class MCBaseModel(BaseModel):
	def __init__(self,name,patience,monitor='val_loss',mode='min'):
		super().__init__(self.name)
		#Checkpoint for weights
		#self.bestmodelweights = name+"weights_best_ep_{epoch:02d}_{val_acc:.3f}.hdf5"
		#self.bestmodelweights = basepath+ self.name+"weights_best_ep_{epoch:02d}.hdf5"
		self.bestmodelweights = basepath+ self.name+"_weights_best.hdf5"
		self.checkpoint = ModelCheckpoint(self.bestmodelweights, monitor=monitor,verbose=1,save_best_only=True, mode=mode)
		#patience
		#stop criterion
		self.early_stopping = EarlyStopping(monitor=monitor,patience=patience)

	def loadBestWeights(self):
		print(self.checkpoint)
		self.model.load_weights(basepath+self.name+"_weights_best.hdf5")
		#self.model.compile( loss='mse', optimizer='adam' )

	def fit(self,X_train,y_train,validation_data,class_weights,epochs=100,verbose=1,batch_size=32):
		self.history = self.model.fit(X_train,y_train,validation_data=validation_data,class_weight=class_weights,epochs=epochs,verbose=verbose,batch_size=batch_size,callbacks = [self.logger, self.early_stopping, self.checkpoint, self.csv_logger])



class MLP_LARGE(MCBaseModel):
	def __init__(self,name, inputshape, n_outputs=10,lr=0.001,patience=20):
		stris = str(inputshape)
		stris = stris.replace("(","")
		stris = stris.replace(")","")
		stris = stris.replace(",","_")
		self.name = "MLP_LARGE_lr"+str(lr)+"_"+stris+"_"+name
		super().__init__(self.name,patience=patience,monitor='val_acc',mode='max')
		#Defining the model
		self.model = Sequential()
		self.model.add(layers.Input(shape=inputshape))
		self.model.add(layers.Dense(1200, activation='relu', name="layer_1"))
		self.model.add(layers.Dense(750, activation='relu', name="layer_2"))
		self.model.add(layers.Dense(125, activation='relu', name="layer_3"))
		self.model.add(layers.Dense(64, activation='relu', name="layer_4"))		
		self.model.add( layers.Dense(n_outputs,activation='softmax', name="classifier") )
		self.model.compile(optimizer=keras.optimizers.SGD(learning_rate=lr), loss='mse',metrics=['mse','acc']) 
		self.model.summary()
		self.name2layer = {}
		for layer in self.model.layers:
			self.name2layer[layer.name] = layer


class MLP_SMALL(MCBaseModel):
	def __init__(self, name,inputshape, n_outputs=10,lr=0.001,patience=20):
		stris = str(inputshape)
		stris = stris.replace("(","")
		stris = stris.replace(")","")
		stris = stris.replace(",","_")
		self.name = "MLP_SMALL_lr"+str(lr)+"_"+stris+"_"+name
		super().__init__(self.name,patience=patience,monitor='val_acc',mode='max')
		#Defining the model
		self.model = Sequential()
		self.model.add(layers.Input(shape=inputshape))
		self.model.add(layers.Dense(125, activation='relu', name="layer_1"))
		self.model.add(layers.Dense(64, activation='relu', name="layer_2"))		
		self.model.add( layers.Dense(n_outputs,activation='softmax', name="classifier") )
		self.model.compile(optimizer=keras.optimizers.SGD(learning_rate=lr), loss='mse',metrics=['mse','acc']) 
		self.model.summary()
		self.name2layer = {}
		for layer in self.model.layers:
			self.name2layer[layer.name] = layer


