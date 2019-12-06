# General file paths
data_path = "../data"
results_path = "../results"

# Data file paths
# Note: All paths below are relative filepaths from the data_path folder
# example: word2vec_filename should be located at "data_path/word2vec_filename"
positive_reviews_filepath = "rt-polaritydata/rt-polarity.pos"
negative_reviews_filepath = "rt-polaritydata/rt-polarity.neg"

subjective_quotes_filepath = "SUBJ/quote.tok.gt9.5000"
objective_quotes_filepath = "SUBJ/plot.tok.gt9.5000"

word2vec_filename = "GoogleNews-vectors-negative300.bin"

############ General constants ################
WORD_VEC_LEN = 300


############# Model Configurations #############
# Default 100
FEATURE_MAPS = 100
# Default [3, 4, 5]
KERNEL_SIZES = [3, 4, 5]
# Default 0.01
REGULARIZATION_STRENGTH = 0.01
# Default 0.5
DROPOUT_RATE = 0.5
# Default adadelta
# OPTIMIZER options include adadelta, adam, rmsprop, sgd...
OPTIMIZER = "adadelta"

############# Configuration to run main ##############
# If True, one print statement will be printed every epoch
PRINT_EPOCH_UPDATES = False
# Run 5 folds (If true, runs a 5 fold validation, if false, all results are evaluated on one fold)
RUN_5_FOLD = False
# Number of epochs to run for each fold of each dataset
EPOCHS = 50
# Batch size to use
BATCH_SIZE = 50
# List of the names of the datasets to evaluate ("MR": movie reviews)
DATASETS = ["SUBJ", "MR"]
# Declare the list of vectors types to use. (options are: "word2vec", and "random")
VECTOR_TYPES = ["random", "word2vec"]
# Declare if the embeddings should be static or dynamic (options are: "static", "dynamic")
EMBEDDING_OPTIONS = ["static", "dynamic"]
