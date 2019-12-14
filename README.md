# CNN_Sentence_Classification
This repository implements a Convolutional Neural Network for Sentence Classification, based on the work of Yoon Kim ([original paper](https://arxiv.org/pdf/1408.5882.pdf)). 
## Datasets Used

### Movie Review
[Movie review dataset](https://www.cs.cornell.edu/people/pabo/movie-review-data/)(**sentence polarity dataset v1.0**)

### Subjectivity
[Subjectivity dataset](https://www.cs.cornell.edu/people/pabo/movie-review-data/) (**Subjectivity dataset v1.0**)


## Hyper-parameter Search


## Architectural Changes


## Packages Needed

- gensim
- nltk
- GoogleNews-vectors-negative300.bin


## How to Run the Program
1. Download the Movie Reviews dataset and the Subjectivity dataset as indicated above, and place them in the `data/` folder. Ensure they follow the same directory structure as below.
2. Open the `src/config.py` file and do the following:
    * While they shouldn't require modification, double check that all filepaths are ok.
    * If you would like to retrain the model, let `retrain_models = True`.
		...
3. Run the `main.py` script.

## Directory Structure
```
.
+-- data
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
* `main.py` contains ...
* `cnn.py` contains ...
* `util/results.py` defines the functionalities needed to save results of runs to files.
* `data_processing/` has the scripts necessary for different data processing strategies.
