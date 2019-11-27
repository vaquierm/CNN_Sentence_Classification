from src.config import WORD_VEC_LEN, FEATURE_MAPS, KERNEL_SIZES, REGULARIZATION_STRENGTH

import numpy as np
from keras import Input, Model, regularizers
from keras.layers import Dense, Dropout, Flatten, Embedding, Conv1D, MaxPooling1D, concatenate


def get_cnn(input_shape: tuple, num_categories: int, embedding_matrix: np.ndarray, embedding_option: str):
    """
    Get the CNN for text classification
    :param input_shape: Should be (max_words_in_sample,)
    :param num_categories: The number of classes
    :param embedding_matrix: The matrix used for word embeddings
    :param embedding_option: Weather or not the embedding is static or dynamic
    :return: Model
    """
    if len(input_shape) > 1:
        raise Exception("Something went wrong, the input shape should be 1 dimension")

    if embedding_option == "static":
        static_embedding = True
    elif embedding_option == "dynamic":
        static_embedding = False
    else:
        raise Exception("The embedding option: " + embedding_option + " is not known. (Must be 'static' or 'dynamic')")

    if REGULARIZATION_STRENGTH < 0:
        raise Exception("Regularization strength cannot be negative, it must be a small positive number")

    max_word_length = input_shape[0]

    input = Input(shape=input_shape, dtype='int32', name='input')

    # Embedding layer
    flow = Embedding(input_dim=embedding_matrix.shape[0], output_dim=WORD_VEC_LEN, input_length=max_word_length, weights=[embedding_matrix], trainable=(not static_embedding), name='embedding')(input)

    convs = []
    for kernel_size in KERNEL_SIZES:
        convs.append(__get_conv_pool_layer(flow, max_word_length, kernel_size))

    # Merge all three branches
    out = concatenate(convs, axis=-1)

    # Add the dropout layer
    out = Dropout(0.5)(out)

    out = Dense(num_categories, activation='softmax', name='output', kernel_regularizer=regularizers.l2(REGULARIZATION_STRENGTH))(out)

    model = Model(inputs=input, outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


def __get_conv_pool_layer(input, max_word_length: int, kernel_size: int):
    """
    Create a 1D convolution layer followed by a max over time pooling
    :param input: Input tensor
    :param max_word_length: Maximum number of words in text sample
    :param kernel_size: Size of window
    :return: The output tensor
    """
    out = Conv1D(filters=FEATURE_MAPS, kernel_size=kernel_size, activation='relu', name='convolution_k' + str(kernel_size), kernel_regularizer=regularizers.l2(REGULARIZATION_STRENGTH))(input)
    out = MaxPooling1D(pool_size=max_word_length - kernel_size + 1, strides=None, padding='valid',
                          name='max_pooling_k' + str(kernel_size))(out)
    out = Flatten(name='flatten_k' + str(kernel_size))(out)
    return out
