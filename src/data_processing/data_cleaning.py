# This file contains everything related to cleaning the text data before transforming them to vector representation
import re


def __clean_str(string: str):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    :param string: String to clean up
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def clean_text_samples(raw_text_samples: list):
    """
    Cleans out unwanted charachters from the text samples
    :param raw_text_samples: List of strings of all text samples
    :return: List of strings of cleaned text samples
    """
    cleaned_reviews = []

    for raw_review in raw_text_samples:
        cleaned_review = __clean_str(raw_review)
        cleaned_reviews.append(cleaned_review)

    return cleaned_reviews
