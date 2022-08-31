from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Masking, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

class CNN_DTW:
    def __init__(self, size, batch_size, lr):
        self.size = size
        self.batch_size = batch_size
        self.lr = lr

        self.model = Sequential([
            # Note the input shape is the desired size of the image sizexsize with 1 color channel
            Masking(mask_value=-99.0, input_shape=(self.size, self.size, 1)),
            Conv2D(64, (3,3), activation='relu'),
            MaxPooling2D(2,2),
            Conv2D(64, (3,3), activation='relu'),
            MaxPooling2D(2,2),
            Conv2D(128, (3,3), activation='relu'),
            MaxPooling2D(2,2),
            Conv2D(128, (3,3), activation='relu'),
            MaxPooling2D(2,2),
            Flatten(),
            Dropout(0.5),
            Dense(512, activation='relu'),
            Dense(3, activation='sigmoid')
        ])
        
    def fit(self, X, y, X_val=None, y_val=None):
        self.model.compile(loss='mse', optimizer=Adam(learning_rate=self.lr), metrics=['mse'])
        if X_val is None:
            callbacks = [ModelCheckpoint(filepath='./checkpoints/checkpoint', monitor='loss', mode='min', verbose=0, save_best_only=True, save_weights_only=True)]
            self.model.fit(X, y, epochs=50, batch_size=self.batch_size, verbose=0, callbacks=callbacks, shuffle=True)
        else:
            callbacks = [ModelCheckpoint(filepath='./checkpoints/checkpoint', monitor='val_loss', mode='min', verbose=0, save_best_only=True, save_weights_only=True)]
            self.model.fit(X, y, epochs=50, batch_size=self.batch_size, validation_data=(X_val,y_val), verbose=0, callbacks=callbacks, shuffle=True)
        self.model.load_weights('./checkpoints/checkpoint')

    def save_model(self, path):
        self.model.save(path+'.h5')

    def load_model(self, path):
        self.model = load_model(path+'.h5')

    def predict(self, X):
        return self.model.predict(X)