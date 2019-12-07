from src.config import WORD_VEC_LEN

import numpy as np
from keras import Input, Model, regularizers
from keras.layers import Dense, Dropout, Flatten, Embedding, Conv1D, MaxPooling1D, concatenate, Layer, InputSpec, Conv2D
import keras.backend as K
import tensorflow as tf


def get_model_config_string(kernel_sizes, dropout_rate, optimizer, feature_maps, regularization_strength):
    """
    Get a string encoding the configuration of the model used
    :return: String representation of the configuration used
    """
    return "FeatureMaps=" + str(feature_maps) + "_KernelSizes=" + str(kernel_sizes).replace(' ', '') + "Regularization=" + str(regularization_strength) + "_Dropout=" + str(dropout_rate) + "_Optimizer=" + optimizer


def get_cnn(input_shape: tuple, num_categories: int, embedding_matrix: np.ndarray, embedding_option: str, kernel_sizes, dropout_rate, optimizer, feature_maps, regularization_strength):
    """
    Get the CNN for text classification
    :param input_shape: Should be (max_words_in_sample,)
    :param num_categories: The number of classes
    :param embedding_matrix: The matrix used for word embeddings
    :param embedding_option: Weather or not the embedding is static or dynamic
    :param kernel_sizes: kernel sizes to use
    :param dropout_rate: dropout rate to use
    :param regularization_strength: regularization strength to use
    :param optimizer: optimizer to use
    :param feature_maps: feature maps to use
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

    if regularization_strength < 0:
        raise Exception("Regularization strength cannot be negative, it must be a small positive number")

    max_word_length = input_shape[0]

    input = Input(shape=input_shape, dtype='int32', name='input')

    # Embedding layer
    flow = Embedding(input_dim=embedding_matrix.shape[0], output_dim=WORD_VEC_LEN, input_length=max_word_length, weights=[embedding_matrix], trainable=(not static_embedding), name='embedding')(input)

    convs = []
    for kernel_size in kernel_sizes:
        convs.append(__get_conv_pool_layer(flow, max_word_length, kernel_size, feature_maps, regularization_strength))

    if len(convs) == 0:
        raise Exception("The model needs at least one convolution layer")
    elif len(convs) == 1:
        out = convs[0]
    else:
        # Merge all three branches
        out = concatenate(convs, axis=-1)

    # Add the dropout layer
    if ((not type(dropout_rate) == float) and (not type(dropout_rate) == int)) or dropout_rate >= 1 or dropout_rate < 0:
        raise Exception("The dropout rate must be between 0 and 1")

    out = Dropout(dropout_rate)(out)

    out = Dense(num_categories, activation='softmax', name='output', kernel_regularizer=regularizers.l2(regularization_strength))(out)

    model = Model(inputs=input, outputs=out)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    model.summary()
    return model


def __get_conv_pool_layer(input, max_word_length: int, kernel_size: int, feature_maps, regularization_strength):
    """
    Create a 1D convolution layer followed by a max over time pooling
    :param input: Input tensor
    :param max_word_length: Maximum number of words in text sample
    :param kernel_size: Size of window
    :return: The output tensor
    """
    out = Conv1D(filters=feature_maps, kernel_size=kernel_size, activation='relu', name='convolution_k' + str(kernel_size), kernel_regularizer=regularizers.l2(regularization_strength))(input)

    out = GlobalMaxPoolZeroOutNonMax()(out)

    out = Conv2D(filters=feature_maps, kernel_size=(kernel_size, WORD_VEC_LEN), activation='relu')(out)

    out = MaxPooling1D(pool_size=max_word_length - kernel_size + 1, strides=None, padding='valid',
                          name='max_pooling_k' + str(kernel_size))(out)
    out = Flatten(name='flatten_k' + str(kernel_size))(out)
    return out


class GlobalMaxPoolZeroOutNonMax(Layer):

    def __init__(self, **kwargs):
        super(GlobalMaxPoolZeroOutNonMax, self).__init__(**kwargs)
        self.input_spec = InputSpec(ndim=3)
        self.data_format = K.normalize_data_format('channels_last')

    def compute_output_shape(self, input_shape):
        return input_shape

    def call(self, inputs):
        raise NotImplemented

    def get_config(self):
        config = {'data_format': self.data_format}
        base_config = super(GlobalMaxPoolZeroOutNonMax, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
