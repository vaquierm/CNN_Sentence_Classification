# All utility functions related to converting the text data data to vector representation
from src.config import data_path, word2vec_filename, WORD_VEC_LEN

import os
import numpy as np
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize


def vectorize_texts(text_samples: list, vector_type: str):
    """
    Vectorize all text samples by converting each words to a vector representation
    :param text_samples: List of strings containing the text samples
    :param vector_type: Either word2vec or random
    :return: numpy array of shape (n, max_words, WORD_VEC_LEN, 1) where n is the number of samples, max_words is the max number of words in a review
    """
    # Build the vocabulary
    vectorizer = CountVectorizer(tokenizer=word_tokenize)
    vectorizer.fit(text_samples)
    vocabulary = list(vectorizer.vocabulary_.keys())
    del vectorizer

    # Create the word to vec mapping
    if vector_type == "random":
        word2vec_dict = __create_word_to_vec_mapping(vocabulary, random=True)
    elif vector_type == "word2vec":
        word2vec_dict = __create_word_to_vec_mapping(vocabulary, random=False)
    else:
        raise Exception("The type of word 2 vec mapping " + vector_type + " is unknown")

    matrix_form_text_samples = __transform_text_samples_to_matrices(text_samples, word2vec_dict)
    del text_samples, word2vec_dict

    return matrix_form_text_samples.reshape(matrix_form_text_samples.shape + (1,))


def __transform_text_samples_to_matrices(text_samples: list, word2vec_dict: dict):
    """
    Uses the complete word to vec dictionary to build matrices for each text sample
    :param text_samples: List of strings for N text samples
    :param word2vec_dict: Mapping from all words in the vocab to their vector representation
    :return: Numpy array of shape (N, max_words, WORD_VEC_LEN)
    """
    N = len(text_samples)
    max_words = max([len(word_tokenize(sample)) for sample in text_samples])

    # Create the buffer that will hold the data
    matrix_text_samples = np.zeros((N, max_words, WORD_VEC_LEN), dtype=np.float)

    for n in range(len(text_samples)):
        text_sample = text_samples[n]
        # Tokenize the word to create its word-wise vector matrix representation
        words = word_tokenize(text_sample)
        for j in range(len(words)):
            matrix_text_samples[n][j] = word2vec_dict[words[j]]

    return matrix_text_samples


def __create_word_to_vec_mapping(vocabulary: list, random: bool):
    """
    Creates a dictionary that maps every word in our corpus to a vector representation
    :param vocabulary: list of all words in corpus
    :param random: If random is True, create random vectors representations for all words, if False only random for unknown words in word2vec
    :return: dictionary mapping words in corpus to vector representation
    """
    if not random:
        word2vec_dict, unknown_words = __load_known_word_vectors(vocabulary)
    else:
        word2vec_dict = {}
        unknown_words = vocabulary

    del vocabulary

    # Generate random vectors for the remaining words
    __generate_random_vectors_for_unknown_words(word2vec_dict, unknown_words)

    return word2vec_dict


def __generate_random_vectors_for_unknown_words(word2vec_dict: dict, unknown_words: list):
    """
    Generate random vector representation for unknown words
    :param word2vec_dict: Current mapping of words to vector representations
    :param unknown_words: List of words to generate random vectors for
    """
    for word in unknown_words:
        word2vec_dict[word] = np.random.uniform(-0.25, 0.25, WORD_VEC_LEN) # TODO Maybe we want to actually get the std and mean


def __load_known_word_vectors(vocabulary: list):
    """
    Loads the pretrained google word2vec and create a dictionary of word to vector representation for words that
    appear in the pretrained data
    :param vocabulary: List of strings of vocabulary in the corpus
    :return: Dictionary of word to vec for known words, list of words unknown to the pretrained word2vec library, list of words that were not in the pretrained model
    """
    word2vec_filepath = os.path.join(data_path, word2vec_filename)

    if not os.path.isfile(word2vec_filepath):
        Exception("The Google pretrained word2vec file does not exist at: " + word2vec_filepath)

    # Load the word2vec model from pretrained file
    word2vec_model = KeyedVectors.load_word2vec_format(word2vec_filepath, binary=True)

    word2vec_dict = {}
    unknown_words = []
    for word in vocabulary:
        if word in word2vec_model:
            word2vec_dict[word] = word2vec_model[word]
        else:
            unknown_words.append(word)

    del word2vec_model, vocabulary

    return word2vec_dict, unknown_words
