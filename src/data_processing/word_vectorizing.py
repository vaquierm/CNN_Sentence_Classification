# All utility functions related to converting the text data data to vector representation
from src.config import data_path, word2vec_filename, WORD_VEC_LEN

import os
import numpy as np
from gensim.models import KeyedVectors
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize


def generate_word_embeddings(text_samples: list, vector_type: str):
    """
    Generate the word embeddings with the input data, encode the input data according to that embedding
    :param text_samples: List of strings corresponding to the text samples
    :param vector_type: Either vord2vec or random
    :return: Encoded text samples (N, max_word_len), embedding matrix (vocab_size, WORD_VEC_LEN)
    """
    # Build the vocabulary
    vectorizer = CountVectorizer(tokenizer=word_tokenize)
    vectorizer.fit(text_samples)
    vocabulary_map = vectorizer.vocabulary_

    del vectorizer

    # Create the word to vec mapping
    if vector_type == "random":
        word2vec_dict = __create_word_to_vec_mapping(list(vocabulary_map.keys()), random=True)
    elif vector_type == "word2vec":
        word2vec_dict = __create_word_to_vec_mapping(list(vocabulary_map.keys()), random=False)
    else:
        raise Exception("The type of word 2 vec mapping " + vector_type + " is unknown")

    return __encode_text(text_samples, vocabulary_map), __create_word_embedding_matrix(vocabulary_map, word2vec_dict)


def __encode_text(text_samples: list, vocabulary_map: dict):
    """
    Encode each text sample into a format of a 1D array where each entry corresponds to a word.
    The value at each index is the index of that word in the vocabulary
    :param text_samples: List of strings of all text samples
    :param vocabulary_map: Mapping of each words in the vocab to its index
    :return: The newly encoded data
    """
    N = len(text_samples)
    max_words = max([len(word_tokenize(sample)) for sample in text_samples])

    encoded_sentences = np.zeros((N, max_words), dtype=np.int)

    for n in range(len(text_samples)):
        text_sample = text_samples[n]
        # Tokenize the word to create its word-wise vector matrix representation
        words = word_tokenize(text_sample)
        for j in range(len(words)):
            # The +1 is because the index 0 is reserved for the empty word token
            encoded_sentences[n][j] = vocabulary_map[words[j]] + 1

    return encoded_sentences


def __create_word_embedding_matrix(vocabulary_map: dict, word2vec_dict: dict):
    """
    Create the word embedding matrix using the index map for each words and the word to vector map
    :param vocabulary_map: Mapping from all words to the index in the vocab
    :param word2vec_dict: Mapping from all words in the vocab to their vector representation
    :return: Word embedding matrix as np array of shape (len(vocab), WORD_VEC_LEN)
    """
    if not len(list(vocabulary_map.keys())) == len(list(word2vec_dict.keys())):
        raise Exception("Something went wrong. Missing some vector representation of words...")

    # The +1 is for the empty word token
    embedding_matrix = np.empty((len(list(vocabulary_map.keys())) + 1, WORD_VEC_LEN))

    embedding_matrix[0] = np.zeros(WORD_VEC_LEN)

    for word in vocabulary_map.keys():
        index = vocabulary_map[word]
        embedding_matrix[index + 1] = word2vec_dict[word]

    return embedding_matrix


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
