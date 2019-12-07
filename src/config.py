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
# Default 0.001
REGULARIZATION_STRENGTH = 0.001
# Default 0.5
DROPOUT_RATE = 0.5
# Default adadelta
# OPTIMIZER options include adadelta, adam, rmsprop, sgd...
OPTIMIZER = "adadelta"


############# Configuration to run main ##############
# If True, one print statement will be printed every epoch
PRINT_EPOCH_UPDATES = False
# Set number of folds to run (out of 5)
FOLDS_TO_RUN = 3
# Set total number k of splits for k fold
NUMBER_OF_SPLITS = 5
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


############# Configuration to run incremental best model finder ##############
# If True, will run the incremental best model
RUN_INCREMENTAL_BEST_PARAMS = False

# Enter below the variables to explore
FEATURE_MAPS_INCREMENTAL = [10, 25, 50, 100, 200, 400, 500]
KERNEL_SIZES_INCREMENTAL = [[1], [2],[3], [4], [5], [6], [7],
                                [1,2], [2,3], [3,4], [4,5], [5,6], [6,7],
                                [1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7],
                                [1, 2, 3, 4], [2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]
                            ]
REGULARIZATION_STRENGTH_INCREMENTAL = [0, 0.0001, 0.0003, 0.0005, 0.001, 0.0015, 0.002, 0.005, 0.01]
DROPOUT_RATE_INCREMENTAL = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
OPTIMIZER_INCREMENTAL = ["adadelta", "adam", "rmsprop", "sgd", "adagrad", "adamax", "nadam"]
