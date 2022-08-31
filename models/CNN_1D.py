from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

class CNN_1D:
    def __init__(self, len, batch_size, lr):
        self.len = len
        self.batch_size = batch_size
        self.lr = lr
    
        self.model = Sequential([
            # input layer
            Input(shape=(self.len, 1)),
            # 1D convolutional layer
            Conv1D(filters=32, kernel_size=4, strides=2, activation='relu'),
            # max pooling layer
            MaxPooling1D(pool_size=2, strides=2),
            # 1D convolutional layer
            Conv1D(filters=32, kernel_size=4, strides=2, activation='relu'),
            # max pooling layer
            MaxPooling1D(pool_size=2, strides=2),
            # three fully connected layers of sizes 128, 64 and 32
            Flatten(),
            Dense(128, activation='relu'),
            Dense(64, activation='relu'),
            Dense(3, activation='sigmoid')
        ])
        
    def fit(self, X, y, X_val=None, y_val=None):
        self.model.compile(loss='mse', optimizer=Adam(learning_rate=self.lr), metrics=['mse'])
        #self.model.fit(X, y, epochs=200, batch_size=self.batch_size, validation_split=0.2, verbose=0, callbacks=callbacks, shuffle=True)
        if X_val is None:
            callbacks = [ModelCheckpoint(filepath='./checkpoints2/checkpoint', monitor='loss', mode='min', verbose=0, save_best_only=True, save_weights_only=True)]
            self.model.fit(X, y, epochs=50, batch_size=self.batch_size, verbose=0, callbacks=callbacks, shuffle=True)
        else:
            callbacks = [ModelCheckpoint(filepath='./checkpoints2/checkpoint', monitor='val_loss', mode='min', verbose=0, save_best_only=True, save_weights_only=True)]
            self.model.fit(X, y, epochs=50, batch_size=self.batch_size, verbose=0, validation_data=(X_val,y_val), callbacks=callbacks, shuffle=True)
        self.model.load_weights('./checkpoints2/checkpoint')

    def save_model(self, path):
        self.model.save(path+'.h5')

    def load_model(self, path):
        self.model = load_model(path+'.h5')

    def predict(self, X):
        return self.model.predict(X)