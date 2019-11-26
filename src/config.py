# General file paths
data_path = "../data"
results_path = "../results"

positive_reviews_filepath = "../data/rt-polaritydata/rt-polarity.pos"
negative_reviews_filepath = "../data/rt-polaritydata/rt-polarity.neg"

word2vec_filename = "GoogleNews-vectors-negative300.bin"

# General constants
WORD_VEC_LEN = 300


############# Model Configurations #############
FEATURE_MAPS = 100
KERNEL_SIZES = [3, 4, 5]

############# Configuration to run main ##############
# Number of epochs to run for each fold of each dataset
EPOCHS = 20
BATCH_SIZE = 50
# List of the names of the datasets to evaluate ("MR": movie reviews)
DATASETS = ["MR"]
# Declare the list of vectors types to use. (options are: "word2vec", and "random")
VECTOR_TYPES = ["word2vec", "random"]
# Declare if the embeddings should be static or dynamic (options are: "static", "dynamic")
EMBEDDING_OPTIONS = ["static", "dynamic"]
