# CNN_Sentence_Classification
This repository implements a Convolutional Neural Network for Sentence Classification, based on the work of Yoon Kim ([original paper](https://arxiv.org/pdf/1408.5882.pdf)). 
## Datasets Used

### Movie Review
[Movie review dataset](https://www.cs.cornell.edu/people/pabo/movie-review-data/) (Download the **sentence polarity dataset v1.0** file and extract it.)

### Subjectivity
[Subjectivity dataset](https://www.cs.cornell.edu/people/pabo/movie-review-data/) (Download the **Subjectivity dataset v1.0** file and extract it.)


## Hyper-parameter Search
To improve on the baseline accuracy, an extensive incremental hyper-parameter search is done to try and find the hyper-parameter values that result in a higher accuracy. 
The hyper-parameter values are explored and set one by one.

## Architectural Changes



## Packages Needed

- gensim
- nltk
- GoogleNews-vectors-negative300.bin


## How to Run the Program
1. Download the Movie Reviews dataset and the Subjectivity dataset as indicated above, and place them in the `data/` folder. Ensure they follow the same directory structure as below.
2. Make sure GoogleNews-vectors-negative300.bin is located under the `data/` folder.
3. Open the `src/config.py` file and do the following:
    * While they shouldn't require modification, double check that all filepaths are ok.
    * Set the configuration to run, such as the dataset (MR for Movie Review, SUBJ for Subjectivity), vector type to use (random, word2vec) and embedding option (static, dynamic).
    * Set the number of epochs and folds desired.
    * Set the desired model hyper-parameter values (feature maps, kernel size, regularization strength, dropout rate and optimizer)
    * If you wish to run incremental parameter tuning, set `RUN_INCREMENTAL_BEST_PARAMS` to `True`. Assign arrays of the values to explore to the hyper-parameter variables ending in `_INCREMENTAL`.
4. Run the `main.py` script.

## Directory Structure
```
.
+-- data
¦   +-- GoogleNews-vectors-negative300.bin
¦   ¦
¦   +-- rt-polaritydata
¦   ¦   +-- rt-polarity.neg
¦   ¦   +-- rt-polarity.pos
¦   ¦
¦   +-- SUBJ
¦       +-- plot.tok.gt9.5000
¦       +-- quote.tok.gt9.5000
¦
+-- results
¦
+-- src
    +-- config.py
    ¦
    +-- main.py
    +-- cnn.py
    ¦
    +-- data_processing
    ¦   +-- data_cleaning.py
    ¦   +-- data_loading.py
    ¦   +-- word_vectorizing.py
    ¦
    +-- util
        +-- results.py
```
The `data/` folder holds the Moview Review and Subjectivity datasets. <br />

Any results are placed automatically in the `results/` folder. These results include confusion matrices, and loss and accuracy graphs. <br />

Files in `src/`:
* `config.py` defines the datasets and word embeddings to be run, and controls the settings for specific configurations.
* `main.py` is a script that trains, evaluates and generates results through k-fold cross validation.
* `cnn.py` contains the implementation of the convolutional neural network for sentence classification.
* `util/results.py` defines the functionalities needed to save results of runs to files.
* `data_processing/` has the scripts necessary for different data processing strategies.
