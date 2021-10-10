import tensorflow as tf
from tensorflow import keras 
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
from typing import Tuple
from sklearn.metrics import confusion_matrix
import seaborn as sns


class Rhythm_Recognizer:
    def __init__(self,model_type):
        self.model_type = model_type
        self.input_train = None
        self.target_train = None
        self.input_test = None
        self.target_test = None
        self.model = None
        self.model_history = None
        self.dataset = None  
        self.train_dataset = None
        self.validation_dataset = None
        self.max_id = None
        self.tokenizer = None 
        self.label_tokenizer = None
        
    def split_dataset(self,dataset: tf.data.Dataset, 
                  dataset_size: int, 
                  train_ratio: float, 
                  validation_ratio: float) -> Tuple[tf.data.Dataset, tf.data.Dataset]:
        assert (train_ratio + validation_ratio) == 1

        train_count = int(dataset_size * train_ratio)
        validation_count = int(dataset_size * validation_ratio)
        test_count = dataset_size - (train_count + validation_count)

        dataset = dataset.shuffle(dataset_size)

        train_dataset = dataset.take(train_count)
        validation_dataset = dataset.skip(train_count).take(validation_count)


        return train_dataset, validation_dataset
    
    
    def prepare_data(self, batch_size = 5):
        with open('/content/drive/MyDrive/DSCVTrn-v2.00.txt') as file:
            X_train = file.readlines()
            X_train = np.array([line.rstrip() for line in X_train])
            
        with open('/content/drive/MyDrive/DSATrn-v2.00.txt') as file:
            y_train = file.readlines()
            y_train = np.array([line.rstrip() for line in y_train])   
            
        indexes_to_drop =[]
        for i in range(len(y_train)):
            if y_train[i] == '@#$%':
                indexes_to_drop.append(i)    

        X_train = np.delete(X_train, indexes_to_drop)
        y_train = np.delete(y_train, indexes_to_drop)
            
        with open('/content/drive/MyDrive/DSCVTst-v2.00.txt') as file:
            X_test = file.readlines()
            X_test = np.array([line.rstrip() for line in X_test])
            
        with open('/content/drive/MyDrive/DSATst-v2.00.txt') as file:
            y_test = file.readlines()
            y_test = np.array([line.rstrip() for line in y_test])    
            
        indexes_to_drop =[]
        for i in range(len(y_test)):
            if y_test[i] == '@#$%':
                indexes_to_drop.append(i)
                
        X_test = np.delete(X_test, indexes_to_drop)
        y_test = np.delete(y_test, indexes_to_drop)
        
        tokenizer = Tokenizer(char_level = True)
        tokenizer.fit_on_texts(X_train)
        tokenizer.word_index

        encoded_X_train = tokenizer.texts_to_sequences(X_train)
        encoded_X_test = tokenizer.texts_to_sequences(X_test)
        
        max_input_len = len(max(X_train, key=len))
        padded_X_train = pad_sequences(encoded_X_train, maxlen=max_input_len, padding='post')
        padded_X_test = pad_sequences(encoded_X_test, maxlen=max_input_len, padding='post')

        label_tokenizer = Tokenizer()
        label_tokenizer.fit_on_texts(y_train)
        target_train = label_tokenizer.texts_to_sequences(y_train)
        target_test = label_tokenizer.texts_to_sequences(y_test)
        self.max_id = len(label_tokenizer.word_index)
        self.label_tokenizer = label_tokenizer
        
        input_train = tf.one_hot(padded_X_train,depth=4).numpy()
        self.input_train =np.array([input_train[i] for i in range(len(input_train))])

        input_test = tf.one_hot(padded_X_test,depth=4).numpy()
        self.input_test =np.array([input_test[i] for i in range(len(input_test))])

        self.target_train = np.array([t[0]-1 for t in target_train])
        self.target_test = np.array([t[0]-1 for t in target_test])

        dataset = tf.data.Dataset.from_tensor_slices((self.input_train, self.target_train))
        self.dataset = dataset.shuffle(1000).repeat(1).batch(5)
                
        self.dataset = self.dataset.prefetch(1)
        self.train_dataset ,self.validation_dataset = self.split_dataset(dataset = self.dataset,dataset_size =len(self.dataset),train_ratio=0.9,validation_ratio=0.1)
        
    def build_layers(self):
        if self.model_type == 'GRU':
            self.model = keras.Sequential()
            self.model.add(keras.layers.GRU(128,input_shape=(None,4), return_sequences=True) )
            self.model.add(keras.layers.GRU(128))
            self.model.add(keras.layers.Dense(self.max_id,activation="softmax"))
            
        if self.model_type == 'LSTM':
            self.model = keras.Sequential()
            self.model.add(keras.layers.LSTM(128,input_shape=(None,4), return_sequences=True) )
            self.model.add(keras.layers.LSTM(128))
            self.model.add(keras.layers.Dense(self.max_id,activation="softmax"))
        
        if self.model_type == 'BILSTM':
            self.model = keras.Sequential()
            self.model.add(keras.layers.Bidirectional(keras.layers.LSTM(128, return_sequences=True),input_shape=(None,4) ))
            self.model.add(keras.layers.Bidirectional(keras.layers.LSTM(128)))
            self.model.add(keras.layers.Dense(self.max_id,activation="softmax"))
       
        return self.model.summary()

    def train(self,epochs=100):
        self.model.compile(loss='sparse_categorical_crossentropy',optimizer="adam" , metrics=['accuracy'])
        self.model_history = self.model.fit(self.dataset,validation_data = self.validation_dataset, epochs=epochs)
        
    def evaluate(self):
        loss,acc  = self.model.evaluate(self.input_test,self.target_test)
        
        print('model loss for test dataset:',loss)
        print('model accuracy for test dataset', acc)
        
    def plot_curves(self):
        plt.plot(self.model_history.history['accuracy'])
        plt.plot(self.model_history.history['val_accuracy'])
        plt.title('{} model accuracy'.format(self.model_type))
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

        plt.plot(self.model_history.history['loss'])
        plt.plot(self.model_history.history['val_loss'])
        plt.title('{} model loss'.format(self.model_type))
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()
        
    def plot_confusion_matrix(self):
        plt.figure(figsize=(14,10))
        y_pred = [np.argmax(x) for x in self.model.predict(self.input_test)] 
        y_true = [x for x in self.target_test] 
        cf = confusion_matrix(y_true, y_pred)
        labels = self.label_tokenizer.word_index.keys()
        ax = sns.heatmap(cf,cmap="gist_ncar",vmin=0.5, linewidth=0.5, annot=True, fmt="d",xticklabels =labels,yticklabels =labels)
        ax.set_title('{} model confusion matrix of rymths'.format(self.model_type))
        plt.show()
        
        
    def save_model(self,path):
        self.model.save(path)
        
    def load_model(self,path):
        self.model = keras.models.load_model(path)


## for gru model
lstm_model = Rhythm_Recognizer('GRU')
lstm_model.prepare_data()
lstm_model.build_layers()
lstm_model.train()
lstm_model.evaluate()
lstm_model.plot_curves()
lstm_model.plot_confusion_matrix()

## for lstm model
lstm_model = Rhythm_Recognizer('LSTM')
lstm_model.prepare_data()
lstm_model.build_layers()
lstm_model.train()
lstm_model.evaluate()
lstm_model.plot_curves()
lstm_model.plot_confusion_matrix()

## for bilstm model
lstm_model = Rhythm_Recognizer('BILSTM')
lstm_model.prepare_data()
lstm_model.build_layers()
lstm_model.train()
lstm_model.evaluate()
lstm_model.plot_curves()
lstm_model.plot_confusion_matrix()