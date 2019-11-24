# This file contains all data loading for all datasets
from src.config import positive_reviews_filepath, negative_reviews_filepath
from src.data_processing.data_cleaning import clean_text_samples
from src.data_processing.word_vectorizing import vectorize_texts

import os
import numpy as np
from keras.utils import to_categorical


class DataLoader:
    @classmethod
    def load_matrix_and_labels(cls, vector_type: str):
        """
        Loads the text data in matrix form with associated labels
        :param vector_type: Either word2vec or random
        :return: numpy array of shape (n, max_words, WORD_VEC_LEN, 1) containing all reviews, numpy array of shape (n, num_classes) for labels
        """

    @classmethod
    def load_raw_text_and_labels(cls):
        """
        Loads the raw text data and the labels associated
        :return: List of strings for each text sample, np array with associated label
        """


# This data loader takes care of loading all data associated with the movie reviews dataset
class MRDataLoader(DataLoader):

    @classmethod
    def load_matrix_and_labels(cls, vector_type: str):
        """
        Loads the raw reviews data and converts it to matrix format
        :param vector_type: Either word2vec or random
        :return: numpy array of shape (n, max_words, WORD_VEC_LEN, 1) containing all reviews, numpy array of shape (n, 2) for labels
        """
        # Load the reviews and associated labels
        reviews_text, reviews_lables = cls.load_raw_text_and_labels()

        # Clean them
        reviews_text = clean_text_samples(reviews_text)

        # Transform the reviews into matrix form
        reviews_matrix = vectorize_texts(reviews_text, vector_type)

        del reviews_text

        return reviews_matrix, to_categorical(reviews_lables, dtype="int")

    @classmethod
    def __load_raw_reviews(cls, sentiment: str):
        """
        Returns a list of all reviews from the sentiment category
        provided in the input
        :param sentiment: Sentiment category to load
        :return: List of strings where each string is a movie review
        """
        if sentiment == "pos":
            reviews_filepath = positive_reviews_filepath
        elif sentiment == "neg":
            reviews_filepath = negative_reviews_filepath
        else:
            raise Exception("The sentiment category " + sentiment + " is not recognized")

        if not os.path.isfile(reviews_filepath):
            raise Exception("The reviews file for sentiment " + sentiment + " at: " + reviews_filepath + " does not exist")

        with open(reviews_filepath, "r", encoding='cp1252') as f:
            reviews = list(f.readlines())

        return reviews

    @classmethod
    def load_raw_text_and_labels(cls):
        """
        Loads all movie review data and returns them as a list of string with an associated numpy array for their labels
        :return: List of strings for all movie reviews, numpy array for all labels
        """
        # Load positive examples
        pos_reviews = cls.__load_raw_reviews("pos")

        # Load negative reviews
        neg_reviews = cls.__load_raw_reviews("neg")

        return (pos_reviews + neg_reviews), np.append(np.ones(len(pos_reviews)), np.zeros(len(neg_reviews))).astype(np.int)
