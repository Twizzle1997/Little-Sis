from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten

class functions: 
    
    def create_model(self):
        model = Sequential()

        model.add(Conv2D(64, kernel_size=(3,3), strides=(1,1), padding='same', activation='relu', input_shape=(32, 32, 3)))
        model.add(Conv2D(128, kernel_size=(3,3), strides=(2,2), padding='same', activation='relu', input_shape=(32, 32, 3)))
        model.add(MaxPool2D(pool_size=(2,2)))
        model.add(Flatten())
        model.add(Dense(10, activation='softmax'))

        model.summary()

        return model