from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Reshape, LSTM


def models(shape):
    def CNN1():
        print("*** Shape ***")
        print(shape[1], shape[2], shape[3])
        model = Sequential()
        model.add(Conv2D(filters=64, kernel_size=3, activation="relu", input_shape=(shape[1], shape[2], shape[3])))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        return model, "CNN2d filter 64"
    def CNN2():
        model = Sequential()
        model.add(Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation="relu",
                         input_shape=(shape[1], shape[2], shape[3])))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        return model, "CNN2d strides=(2, 2), filter 64"
    def CNN3():
        model = Sequential()
        model.add(Conv2D(filters=64, kernel_size=3, activation="relu",
                         input_shape=(shape[1], shape[2], shape[3])))
        model.add(Flatten())
        model.add(Dense(1000, activation='relu'))
        model.add(Dense(500, activation='relu'))
        model.add(Dense(80, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        return model, "CNN2d filter 64, dense: 1000,500,80,1"
    def CNN4():
        model = Sequential()
        model.add(Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation="relu",
                         input_shape=(shape[1], shape[2], shape[3])))
        model.add(Flatten())
        model.add(Dense(1000, activation='relu'))
        model.add(Dense(500, activation='relu'))
        model.add(Dense(80, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        return model, "CNN2d strides=(2, 2), filter 64, dense: 1000,500,80,1"
    def CNN_RNN1():
        if shape[1] % 2 == 1:
            new_win_size = shape[1] // 2
        else:
            new_win_size = shape[1] // 2 - 1
        model = Sequential()
        model.add(Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation="relu",
                         input_shape=(shape[1], shape[2], shape[3])))
        model.add(Reshape((new_win_size, (shape[2] // 2) * 64)))
        model.add(LSTM(new_win_size, activation='tanh', return_sequences=False))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        return model, "CNN2d + RNN strides=(2, 2), filter 64"
    def CNN_RNN2():
        model = Sequential()
        model.add(Conv2D(filters=64, kernel_size=3, activation="relu",
                         input_shape=(shape[1], shape[2], shape[3])))
        model.add(Reshape((shape[1] - 2, (shape[2] - 2) * 64)))
        model.add(LSTM(shape[1] - 2, activation='tanh', return_sequences=False))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        return model, "CNN2d + RNN, filter 64"
    def RNN():
        model = Sequential()
        model.add(Reshape((shape[1], (shape[2]) * shape[3]), input_shape=(shape[1], shape[2], shape[3])))
        model.add(LSTM(shape[1], activation='tanh', return_sequences=False))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        return model, "RNN, activation='tanh', return_sequences=False"
    return [CNN1, CNN2, CNN3, CNN4, CNN_RNN1, CNN_RNN2, RNN]
