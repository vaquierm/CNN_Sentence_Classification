# All utility functions related to converting the reviews data to vector representation
from src.config import data_path, word2vec_filename, WORD_VEC_LEN

import os
import numpy as np
from gensim.models import KeyedVectors


def vectorize_reviews(reviews: list, vector_type: str):
    """
    Vectorize all reviews by converting each words to a vector representation
    :param reviews: List of strings containing the reviews
    :param vector_type: Either word2vec or random
    :return: numpy array of shape (n, 300, max_words) where n is the number of samples, max_words is the max number of words in a review
    """
    max_words = max([len(rev.split(" ")) for rev in reviews])

    # TODO: get the word2vec mapping then transform the data to that format


def create_word_to_vec_mapping(vocabulary: list, random: bool):
    """
    Creates a dictionary that maps every word in our corpus to a vector representation
    :param vocabulary: list of all words in corpus
    :param random: If random is True, create random vectors representations for all words, if False only random for unknown words in word2vec
    :return: dictionary mapping words in corpus to vector representation
    """
    if not random:
        word2vec_dict, unknown_words = load_known_word_vectors(vocabulary)
    else:
        word2vec_dict = {}
        unknown_words = vocabulary

    del vocabulary

    # Generate random vectors for the remaining words
    generate_random_vectors_for_unknown_words(word2vec_dict, unknown_words)

    return word2vec_dict


def generate_random_vectors_for_unknown_words(word2vec_dict: dict, unknown_words: list):
    """
    Generate random vector representation for unknown words
    :param word2vec_dict: Current mapping of words to vector representations
    :param unknown_words: List of words to generate random vectors for
    """
    for word in unknown_words:
        word2vec_dict[word] = np.random.uniform(-0.25, 0.25, WORD_VEC_LEN) # TODO Maybe we want to actually get the std and mean


def load_known_word_vectors(vocabulary: list):
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


if __name__ == '__main__':
    load_known_word_vectors(["hello", "france", "njdende"])
