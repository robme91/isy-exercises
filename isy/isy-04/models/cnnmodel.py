from keras.models import Sequential
from keras.layers import Dropout, Convolution2D, MaxPooling2D, Flatten, Dense, Activation, Conv2D, Embedding, LSTM


class CNNModel:
    img_rows = 28
    img_cols = 28

    @staticmethod
    def load_inputshape():
        return CNNModel.img_rows, CNNModel.img_cols, 1

    @staticmethod
    def reshape_input_data(x_train, x_test):
        x_train = x_train.reshape(x_train.shape[0], CNNModel.img_rows, CNNModel.img_cols, 1)
        x_test = x_test.reshape(x_test.shape[0], CNNModel.img_rows, CNNModel.img_cols, 1)
        return x_train, x_test


    def configure_pregiven_model(self, model):
        '''
         configures the model like it is given in the exercise
        '''
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)))   # add input layer, with shape of images= 28x28 in gray -> 28,28,1
        model.add(Conv2D(32, (3, 3), activation='relu'))    # need kernel size, otherwise error
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        # output layer
        model.add(Dense(10, activation='softmax'))
        return model

    def configure_own_model(self, model):
        '''
            configure the model with an own architecture
        '''
        model.add(Conv2D(64, (5, 5), padding='same', activation='relu', input_shape=(28, 28, 1)))   # add input layer, with shape of images= 28x28 in gray -> 28,28,1
        model.add(Dropout(0.5))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.5))
        model.add(Conv2D(128, (5, 5), padding='same', activation='relu'))
        model.add(Dropout(0.5))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Conv2D(128, (5, 5), padding='same', activation='relu'))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        # output layer
        model.add(Dense(10, activation='softmax'))
        return model

    @staticmethod
    def load_model(classes=10, usePregivenModel=True):
        model = Sequential()    # create linear stack of layer models
        # add layers
        if usePregivenModel:
            model = CNNModel.configure_pregiven_model(CNNModel, model)
        else:
            model = CNNModel.configure_own_model(CNNModel, model)

        # compile model
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model


