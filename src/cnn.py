from src.config import WORD_VEC_LEN

from keras import Sequential, Input, Model, regularizers
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPool2D, Concatenate


# https://stackoverflow.com/questions/43151775/how-to-have-parallel-convolutional-layers-in-keras
def get_cnn_model(input_shape, num_categories):
    kernel_sizes = {3, 4, 5}
    filters = 100
    input = Input(shape=input_shape)
    layers = []

    for k in kernel_sizes:
        layer = Conv2D(filters=filters, kernel_size=(k, WORD_VEC_LEN), padding='valid', activation='relu', batch_size=50, input_shape=input_shape, kernel_regularizer=regularizers.l2(3))(input)
        # max-over-time pooling
        # the pool shape needs to be (1, sequence_length)
        pool = MaxPool2D(pool_size=(input_shape[0] - int(k/2) - 2, 1), padding='valid')(layer)
        layers.append(pool)

    output = Concatenate()(layers)
    convolutional_layer = Model(input=input, output=output)

    convolutional_layer.summary()

    model = Sequential()
    # the model uses ReLU, filter windows (h) of 3, 4, 5 with 100 feature maps each, dropout rate (p) of 0.5,
    # l2 constraint (s) of 3, and mini-batch size of 50
    # with padding
    model.add(convolutional_layer)
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(num_categories, activation='softmax'))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

    model.summary()
    return model


if __name__ == '__main__':
    model = get_cnn_model((300, 100, 1), 2)
    model.summary()