# This file contains all data loading for the Movie Review Data
from src.config import training_data_path

import os
import glob


def load_reviews(sentiment: str):
    """
    Returns a list of all reviews from the sentiment category
    provided in the input
    :param sentiment: Sentiment category to load
    :return: List of strings where each string is a movie review
    """
    if sentiment == "pos":
        directory = os.path.join(training_data_path, "pos")
    elif sentiment == "neg":
        directory = os.path.join(training_data_path, "neg")
    else:
        raise Exception("The sentiment category " + sentiment + " is not recognized")

    if not os.path.isdir(directory):
        raise Exception("The path to load reviews of category " + sentiment + ": " + directory + " does not exist")

    reviews = []
    for review_file in glob.glob(os.path.join(directory, "*.txt")):
        with open(review_file, "r") as f:
            reviews.append("".join(f.readlines()))

    return reviews
