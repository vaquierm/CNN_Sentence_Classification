# This is the main script that will train, evaluate, and generate results
from src.config import VECTOR_TYPES, DATASETS, EMBEDDING_OPTIONS, EPOCHS, BATCH_SIZE, KERNEL_SIZES, \
    results_path, PRINT_EPOCH_UPDATES, FEATURE_MAPS, REGULARIZATION_STRENGTH, DROPOUT_RATE, OPTIMIZER, \
    RUN_INCREMENTAL_BEST_PARAMS, KERNEL_SIZES_INCREMENTAL, DROPOUT_RATE_INCREMENTAL, FEATURE_MAPS_INCREMENTAL, \
    OPTIMIZER_INCREMENTAL, REGULARIZATION_STRENGTH_INCREMENTAL, FOLDS_TO_RUN, NUMBER_OF_SPLITS
from src.data_processing.data_loading import get_data_loader
from src.cnn import get_cnn, get_model_config_string
from src.util.results import save_training_history, save_confusion_matrix, compute_average_histories

import numpy as np
import os
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix, accuracy_score
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.utils import to_categorical


def k_fold_cv(dataset: str, vec_type: str, embedding_option: str, k: int = NUMBER_OF_SPLITS, save_results: bool = True,
              kernel_sizes=KERNEL_SIZES, dropout_rate=DROPOUT_RATE, optimizer=OPTIMIZER, feature_maps=FEATURE_MAPS, regularization_strength=REGULARIZATION_STRENGTH):
    """
    Performs a k fold cross validation os the specific vector type and dataset
    :param dataset: dataset key
    :param vec_type: Type of vector representation
    :param embedding_option: Weather or not the embedding is static or not
    :param k: Number of folds to perform
    :param save_results: If true, results are saved in the results directory
    :param kernel_sizes: kernel sizes to use
    :param dropout_rate: dropout rate to use
    :param regularization_strength: regularization strength to use
    :param optimizer: optimizer to use
    :param feature_maps: feature maps to use
    :return average_accuracy: the average accuracy of the folds
    """
    if FOLDS_TO_RUN > k or FOLDS_TO_RUN <= 0 or k <= 0:
        raise Exception("Number of folds to run cannot exceed the number of splits and cannot be less or equal to 0")

    data_loader = get_data_loader(dataset)

    # Load the dataset with the correct vector representation
    print("\tLoading dataset {" + dataset + "} with vector representation {" + vec_type + "}...")
    X, Y, embeddings = data_loader.load_X_Y_embeddings(vec_type)

    class_labels = data_loader.get_class_labels()
    num_classes = len(class_labels)

    # Accumulator to get the average accuracy of all k folds
    average_accuracy = 0
    confusion_mat = np.zeros((num_classes, num_classes), dtype=np.int)
    # Used to keep track of all histories of all 5 folds
    histories = []
    i = 0
    for train_index, test_index in KFold(n_splits=k, random_state=42, shuffle=True).split(X, Y):
        print("\t\tStarting fold " + str(i) + "...")

        x_train, x_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]

        model = get_cnn(input_shape=X[0].shape, num_categories=num_classes, embedding_matrix=embeddings, embedding_option=embedding_option, kernel_sizes=kernel_sizes, dropout_rate=dropout_rate, optimizer=optimizer, feature_maps=feature_maps, regularization_strength=regularization_strength)

        reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=6, verbose=0, min_lr=0.00001)
        model_checkpoint = ModelCheckpoint("temp.h5", monitor='val_acc', mode='max', verbose=0, save_best_only=True)
        history = model.fit(x=x_train, y=to_categorical(y_train), batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=2 if PRINT_EPOCH_UPDATES else 0, callbacks=[model_checkpoint, reduce_lr], validation_data=(x_test, to_categorical(y_test)))

        # Load the best model and predict to produce a confusion matrix
        model.load_weights("temp.h5")
        os.remove("temp.h5")
        y_pred = model.predict(x_test).argmax(axis=1)

        # Accumulate the results
        histories.append(history.history)
        confusion_mat += confusion_matrix(y_test, y_pred)

        # Break once we have reached the desired number of folds
        average_accuracy += accuracy_score(y_test, y_pred) / FOLDS_TO_RUN

        i += 1
        if i >= FOLDS_TO_RUN:
            break

    print("\tAverage accuracy: " + str(average_accuracy))

    if save_results:
        result_sub_dir = os.path.join(results_path, dataset, embedding_option, vec_type, get_model_config_string(kernel_sizes, dropout_rate, optimizer, feature_maps, regularization_strength))

        # Create the subfolder if it does not exist
        if not os.path.exists(result_sub_dir):
            os.makedirs(result_sub_dir)

        average_history = compute_average_histories(histories)

        save_training_history(average_history, os.path.join(result_sub_dir, "acc.png"), os.path.join(result_sub_dir, "loss.png"))

        save_confusion_matrix(confusion_mat, class_labels, os.path.join(result_sub_dir, "conf.png"), title="Acc: " + str(round(average_accuracy, 4)) + ", Dataset: " + dataset + ", Vec: " + vec_type + "_" + embedding_option)

    return average_accuracy


def incremental_search_best_params(dataset: str, vec_type: str, embedding_option: str):
    best_accuracy = 0
    best_kernel_size = KERNEL_SIZES
    best_optimizer = OPTIMIZER
    best_regularization_strength = REGULARIZATION_STRENGTH
    best_dropout_rate = DROPOUT_RATE
    best_feature_map = FEATURE_MAPS

    # run trials and keep parameter if best accuracy so far
    for kernel_sizes in KERNEL_SIZES_INCREMENTAL:
        print("\n\tAttempting with kernel sizes:" + str(kernel_sizes))
        # Perform a 5 fold cross validation
        accuracy = k_fold_cv(dataset, vec_type, embedding_option,
                             kernel_sizes=kernel_sizes, dropout_rate=best_dropout_rate, optimizer=best_optimizer,
                             feature_maps=best_feature_map, regularization_strength=best_regularization_strength)
        if accuracy > best_accuracy:
            print("\n\tNew best kernel sizes found!")
            best_accuracy = accuracy
            best_kernel_size = kernel_sizes
    print("\n\tSet best kernel sizes as:" + str(best_kernel_size))

    for regularization_strength in REGULARIZATION_STRENGTH_INCREMENTAL:
        print("\n\tAttempting with regularization strength:" + str(regularization_strength))
        # Perform a 5 fold cross validation
        accuracy = k_fold_cv(dataset, vec_type, embedding_option,
                             kernel_sizes=best_kernel_size, dropout_rate=best_dropout_rate, optimizer=best_optimizer,
                             feature_maps=best_feature_map, regularization_strength=regularization_strength)
        if accuracy > best_accuracy:
            print("\n\tNew best regularization strength found!")
            best_accuracy = accuracy
            best_regularization_strength = regularization_strength
    print("\n\tSet best regularization strength as:" + str(best_regularization_strength))

    for optimizer in OPTIMIZER_INCREMENTAL:
        print("\n\tAttempting with optimizer:" + optimizer)
        # Perform a 5 fold cross validation
        accuracy = k_fold_cv(dataset, vec_type, embedding_option,
                             kernel_sizes=best_kernel_size, dropout_rate=best_dropout_rate, optimizer=optimizer,
                             feature_maps=best_feature_map, regularization_strength=best_regularization_strength)
        if accuracy > best_accuracy:
            print("\n\tNew best optimizer found!")
            best_accuracy = accuracy
            best_optimizer = optimizer
    print("\n\tSet best optimizer as:" + str(best_optimizer))

    for dropout_rate in DROPOUT_RATE_INCREMENTAL:
        print("\n\tAttempting with dropout rate:" + str(dropout_rate))
        # Perform a 5 fold cross validation
        accuracy = k_fold_cv(dataset, vec_type, embedding_option,
                             kernel_sizes=best_kernel_size, dropout_rate=dropout_rate, optimizer=best_optimizer,
                             feature_maps=best_feature_map, regularization_strength=best_regularization_strength)
        if accuracy > best_accuracy:
            print("\n\tNew best dropout rate found!")
            best_accuracy = accuracy
            best_dropout_rate = dropout_rate
    print("\n\tSet best dropout rate as:" + str(best_dropout_rate))

    for feature_maps in FEATURE_MAPS_INCREMENTAL:
        print("\n\tAttempting with feature maps:" + str(feature_maps))
        # Perform a 5 fold cross validation
        accuracy = k_fold_cv(dataset, vec_type, embedding_option,
                             kernel_sizes=best_kernel_size, dropout_rate=best_dropout_rate, optimizer=best_optimizer,
                             feature_maps=feature_maps, regularization_strength=best_regularization_strength)
        if accuracy > best_accuracy:
            print("\n\tNew best feature maps found!")
            best_accuracy = accuracy
            best_feature_map = feature_maps
    print("\n\tSet best feature maps as:" + str(best_feature_map))

    # Print results
    print("\nBest accuracy of " + str(best_accuracy) + " achieved for:\n\tkernel sizes: " + str(best_kernel_size) + "\n\t# feature maps: " + str(best_feature_map) + "\n\tRegularization strength: " + str(best_regularization_strength) + "\n\tDropout rate: " + str(best_dropout_rate) + "\n\tOptimizer: " + best_optimizer)


def main():
    if not RUN_INCREMENTAL_BEST_PARAMS:
        print("Evaluating model with parameters:\n\tkernel sizes: " + str(KERNEL_SIZES) + "\n\t# feature maps: " + str(FEATURE_MAPS) + "\n\tRegularization strength: " + str(REGULARIZATION_STRENGTH) + "\n\tDropout rate: " + str(DROPOUT_RATE) + "\n\tOptimizer: " + OPTIMIZER)

    for dataset in DATASETS:
        for vec_type in VECTOR_TYPES:
            for embedding_option in EMBEDDING_OPTIONS:
                if vec_type == "random" and embedding_option == "static":
                    print("\nSkipping model with vector type {" + vec_type + "} and embedding option {"+ embedding_option + "}")
                    continue
                print("\nEvaluating model with vector type {" + vec_type + "} and embedding option {"+ embedding_option + "} on dataset {" + dataset + "}...")

                if RUN_INCREMENTAL_BEST_PARAMS:
                    # find parameters that give best accuracy
                    incremental_search_best_params(dataset, vec_type, embedding_option)
                else:
                    # Perform a 5 fold cross validation
                    k_fold_cv(dataset, vec_type, embedding_option)


if __name__ == '__main__':
    main()
