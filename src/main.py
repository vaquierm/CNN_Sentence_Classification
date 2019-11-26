# This is the main script that will train, evaluate, and generate results
from src.config import VECTOR_TYPES, DATASETS, EPOCHS, BATCH_SIZE, results_path
from src.data_processing.data_loading import get_data_loader
from src.cnn import get_cnn_model
from src.util.results import save_training_history, save_confusion_matrix, compute_average_histories

import numpy as np
import os
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical


def k_fold_cv(dataset: str, vec_type: str, k: int = 3, save_results: bool = True):
    """
    Performs a k fold cross validation os the specific vector type and dataset
    :param dataset: dataset key
    :param vec_type: Type of vector representation
    :param k: Number of folds to perform
    :param save_results: If true, results are saved in the results directory
    """
    data_loader = get_data_loader(dataset)

    # Load the dataset with the correct vector representation
    print("\tLoading dataset {" + dataset + "} with vector representation {" + vec_type + "}...")
    X, Y = data_loader.load_X_Y_embeddings(vec_type)

    class_labels = data_loader.get_class_labels()
    num_classes = len(class_labels)
    average_accuracy = 0
    confusion_mat = np.zeros((num_classes, num_classes), dtype=np.int)
    histories = []
    i = 0
    for train_index, test_index in KFold(n_splits=k, random_state=42, shuffle=True).split(X, Y):
        print("\t\tStarting fold " + str(i) + "...")

        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        model = get_cnn_model(input_shape=X[1].shape, num_categories=num_classes)

        model_checkpoint = ModelCheckpoint("temp.h5", monitor='val_accuracy', mode='max', verbose=0, save_best_only=True)
        history = model.fit(x=x_train, y=to_categorical(y_train), batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1, callbacks=[model_checkpoint], validation_data=(x_test, to_categorical(y_test)))

        # Load the best model and predict to produce a confusion matrix
        model.load_weights("temp.h5")
        os.remove("temp.h5")
        y_pred = model.predict(x_test).argmax(axis=1)

        histories.append(history.history)
        average_accuracy += accuracy_score(y_test, y_pred) / k
        confusion_mat += confusion_matrix(y_test, y_pred)

        i += 1

    print("\tAverage accuracy: " + str(average_accuracy))

    if save_results:
        result_sub_dir = os.path.join(results_path, dataset, vec_type)

        # Create the subfolder if it does not exist
        if not os.path.exists(result_sub_dir):
            os.makedirs(result_sub_dir)

        average_history = compute_average_histories(histories)

        save_training_history(average_history, os.path.join(result_sub_dir, "acc.png"), os.path.join(result_sub_dir, "loss.png"))

        save_confusion_matrix(confusion_mat, class_labels, os.path.join(result_sub_dir, "conf.png"), title="Acc: " + str(round(average_accuracy, 2)) + ", Dataset: " + dataset + ", Vec: " + vec_type)


def main():

    for dataset in DATASETS:
        for vec_type in VECTOR_TYPES:
            print("\nEvaluating model with vector type {" + vec_type + "} on dataset {" + dataset + "}...")

            # Perform a 5 fold cross validation
            k_fold_cv(dataset, vec_type)


if __name__ == '__main__':
    main()
