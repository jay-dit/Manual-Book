import numpy as np
from progress.bar import IncrementalBar
import tensorflow as tf
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow import keras
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, CSVLogger


class Ansamble():
    def __init__(self, all_models):
        self.models = all_models
        self.loss = np.zeros((len(all_models),8),
                            dtype = 'float32')
    def culc_loss(self, X, y):
        
        bar = IncrementalBar('Countdown', max = len(self.models) + 1)
        
        for ind, model in enumerate(self.models):
            y_pred = model.predict(X)
            for i in range(len(y)):
                a1 = list(y[i])
                a2 = list(y_pred[i])
                if a1.index(max(a1)) != a2.index(max(a2)):
                    self.loss[ind, a1.index(max(a1))] += 1
            bar.next()
        self.loss = 1 - (self.loss - self.loss.min(axis = 0))/(self.loss.max(axis = 0)-self.loss.min(axis = 0))
        bar.next()
        pass

    def train(self, X, y, epochs = 12):
        from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN, CSVLogger
        reduce_learning_rate = ReduceLROnPlateau(monitor='loss',
                                         factor=0.5,
                                         patience=1,
                                         verbose=1,)
        EarlyStopping = tf.keras.callbacks.EarlyStopping(
                                    monitor='val_loss',
                                    patience=7,
                                    verbose=0,
                                    restore_best_weights=True
                                    )

        callbacks = [reduce_learning_rate, EarlyStopping]

        for model in self.models:

            model.fit(x = X,
                      y = y,
                      epochs = epochs,
                      callbacks=callbacks,
                      verbose = 1,
                      validation_split = 0.15)

    def predictions(self, X):
        y = self.models[0].predict(X) * self.loss[0,:]

        for i, model in enumerate(self.models[1:]):
            
            y += model.predict(X) * self.loss[i+1,:]
        
        return y 




