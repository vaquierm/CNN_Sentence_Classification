# This file contains all data loading for all datasets
from src.config import data_path, positive_reviews_filepath, negative_reviews_filepath, objective_quotes_filepath, subjective_quotes_filepath
from src.data_processing.data_cleaning import clean_text_samples
from src.data_processing.word_vectorizing import generate_word_embeddings

import os
import numpy as np


def get_data_loader(dataset_key: str):
    """
    Get the appropriate DataLoader
    :param dataset_key: Key string of the data loader to get
    :return: Data loader
    """
    if dataset_key == "MR":
        return MRDataLoader
    if dataset_key == "SUBJ":
        return SUBJDataLoader
    else:
        raise Exception("The data set key {" + dataset_key + "} is unknown")


class DataLoader:
    @classmethod
    def load_X_Y_embeddings(cls, vector_type: str):
        """
        Loads the text data in matrix form with associated labels
        :param vector_type: Either word2vec or random
        :return: numpy array of shape (n, max_words, WORD_VEC_LEN, 1) containing all reviews, numpy array of shape (n,) for labels
        """
        pass

    @classmethod
    def load_raw_text_and_labels(cls):
        """
        Loads the raw text data and the labels associated
        :return: List of strings for each text sample, np array with associated label
        """
        pass

    @classmethod
    def get_class_labels(cls):
        """
        Get a list of strings corresponding to the class labels
        :return: List of labels
        """
        pass


# This data loader takes care of loading all data associated with the movie reviews dataset
class MRDataLoader(DataLoader):

    @classmethod
    def load_X_Y_embeddings(cls, vector_type: str):
        """
        Loads the raw reviews data and converts it the index encoding after creating an embedding matrix
        :param vector_type: Either word2vec or random
        :return: numpy array of shape (n, max_words, WORD_VEC_LEN, 1) containing all reviews, numpy array of shape (n,) for labels
        """
        # Load the reviews and associated labels
        reviews_text, reviews_lables = cls.load_raw_text_and_labels()

        # Clean them
        reviews_text = clean_text_samples(reviews_text)

        # Transform the reviews into matrix form
        encoded_reviews, embedding_matrix = generate_word_embeddings(reviews_text, vector_type)

        del reviews_text

        # Shuffle the data since we dont want it to be sorted by classes
        p = np.random.permutation(reviews_lables.shape[0])

        return encoded_reviews[p], reviews_lables[p], embedding_matrix

    @classmethod
    def __load_raw_reviews(cls, sentiment: str):
        """
        Returns a list of all reviews from the sentiment category
        provided in the input
        :param sentiment: Sentiment category to load
        :return: List of strings where each string is a movie review
        """
        if sentiment == "pos":
            reviews_filepath = os.path.join(data_path, positive_reviews_filepath)
        elif sentiment == "neg":
            reviews_filepath = os.path.join(data_path, negative_reviews_filepath)
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

    @classmethod
    def get_class_labels(cls):
        """
        Get a list of strings corresponding to the class labels
        :return: List of labels
        """
        return ["Negative", "Positive"]


# This data loader takes care of loading all data associated with the movie reviews dataset
class SUBJDataLoader(DataLoader):

    @classmethod
    def load_X_Y_embeddings(cls, vector_type: str):
        """
        Loads the quotes data and converts it the index encoding after creating an embedding matrix
        :param vector_type: Either word2vec or random
        :return: numpy array of shape (n, max_words, WORD_VEC_LEN, 1) containing all quotes, numpy array of shape (n,) for labels
        """
        # Load the reviews and associated labels
        reviews_text, reviews_lables = cls.load_raw_text_and_labels()

        # Clean them
        reviews_text = clean_text_samples(reviews_text)

        # Transform the reviews into matrix form
        encoded_reviews, embedding_matrix = generate_word_embeddings(reviews_text, vector_type)

        del reviews_text

        # Shuffle the data since we dont want it to be sorted by classes
        p = np.random.permutation(reviews_lables.shape[0])

        return encoded_reviews[p], reviews_lables[p], embedding_matrix

    @classmethod
    def __load_raw_quotes(cls, file_name: str):
        """
        Returns a list of all quotes from the file
        provided in the input
        :param file_name: Filename to load from
        :return: List of strings where each string is a quote
        """
        if not os.path.isfile(file_name):
            raise Exception("The file you are trying to load at " + file_name + " for the SUBJ dataset does not exist")

        with open(file_name, "r", encoding='cp1252') as f:
            reviews = list(f.readlines())

        return reviews

    @classmethod
    def load_raw_text_and_labels(cls):
        """
        Loads all quotes data and returns them as a list of string with an associated numpy array for their labels
        :return: List of strings for all quotes, numpy array for all labels
        """
        objective_quotes = cls.__load_raw_quotes(os.path.join(data_path, objective_quotes_filepath))

        subjective_quotes = cls.__load_raw_quotes(os.path.join(data_path, subjective_quotes_filepath))

        return (objective_quotes + subjective_quotes), np.append(np.ones(len(objective_quotes)), np.zeros(len(subjective_quotes))).astype(np.int)

    @classmethod
    def get_class_labels(cls):
        """
        Get a list of strings corresponding to the class labels
        :return: List of labels
        """
        return ["Objective", "Subjective"]
