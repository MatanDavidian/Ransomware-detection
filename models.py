from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Reshape, LSTM, Conv1D, ConvLSTM2D
from tensorflow.keras import layers


def models(shape):
    def CNN1():
        print("*** Shape ***")
        print(shape[1], shape[2], shape[3])
        model = Sequential()
        model.add(Conv2D(filters=128, kernel_size=3, activation="relu", input_shape=(shape[1], shape[2], shape[3])))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        return model, "CNN2d filter 128\n"
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
        return model, "CNN2d + RNN strides=(2, 2), filter \n"
    def CNN_RNN2():
        model = Sequential()
        model.add(Conv2D(filters=128, kernel_size=3, activation="relu",
                         input_shape=(shape[1], shape[2], shape[3])))
        model.add(Reshape((shape[1] - 2, (shape[2] - 2) * 128)))
        model.add(LSTM(shape[1] - 2, activation='tanh', return_sequences=False))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        return model, "CNN2d + RNN, filters=128, kernel_size=3\n"
    def CNN_RNN3():
        model = Sequential()
        model.add(Conv2D(filters=128, kernel_size=(1, 3), strides=(1, 2), activation="relu",
                         input_shape=(shape[1], shape[2], shape[3])))
        model.add(Reshape((shape[1], (shape[2] // 2) * 128)))
        model.add(LSTM(shape[1], activation='tanh', return_sequences=False))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        return model, "CNN2d + RNN, filters=128, kernel_size=(1, 3), strides=(1, 2)\n"
    def RNN():
        model = Sequential()
        model.add(Reshape((shape[1], (shape[2]) * shape[3]), input_shape=(shape[1], shape[2], shape[3])))
        model.add(LSTM(shape[1], activation='tanh', return_sequences=False))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        return model, "RNN, activation='tanh', return_sequences=False\n"
    def RNN2():
        model = Sequential()
        model.add(Reshape((shape[1], (shape[2]) * shape[3]), input_shape=(shape[1], shape[2], shape[3])))
        model.add(LSTM(shape[1], activation='sigmoid', return_sequences=False))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        return model, "RNN, activation='sigmoid', return_sequences=False\n"

    def convLSTM(NUM_IMAGES):
        model = Sequential()
        model.add(ConvLSTM2D(128, kernel_size=(3, 3), activation='sigmoid', padding='valid', return_sequences=False,
                             input_shape=(NUM_IMAGES, shape[1], shape[2], shape[3])))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        return model, "convLSTM,filters=128,kernel=(3,3) activation='sigmoid', return_sequences=False\n"

    return [CNN1]  # [RNN, RNN2, CNN1, CNN_RNN3, CNN_RNN2] convLSTM
