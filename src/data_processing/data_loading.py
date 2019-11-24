# This file contains all data loading for the Movie Review Data
from src.config import positive_reviews_filepath, negative_reviews_filepath

import os
import numpy as np


def load_reviews(sentiment: str):
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


def load_all_reviews_with_labels():
    """
    Loads all movie review data and returns them as a list of string with an associated numpy array for their labels
    :return: List of strings for all movie reviews, numpy array for all labels
    """
    # Load positive examples
    pos_reviews = load_reviews("pos")

    # Load negative reviews
    neg_reviews = load_reviews("neg")

    return (pos_reviews + neg_reviews), np.append(np.ones(len(pos_reviews)), np.zeros(len(neg_reviews))).astype(np.int)
