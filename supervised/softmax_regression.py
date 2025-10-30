"""
Classification of a 4D iris dataset using softmax regression.

Matrix and vector dimensions are as follow:
    D: number of features
    C: number of classes
    N: number of samples
    N_train: number of training samples
    N_test: number of test samples

NOTES:

"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(filepath):
    """
    Loads data into a pandas dataframe.

    :param filepath:
    :return dataframe:
    """
    return pd.read_csv(filepath)


def split_data(dataframe, percent_train=80):
    """
    Creates a training and testing split of the dataset. Expects the
    rows of the dataset to be samples and the columns to be features.

    :param dataframe:
    :param percent_train:
    :return:
    """
    df_shuffled = dataframe.sample(frac=1, random_state=42)
    num_samples, num_features = df_shuffled.shape

    X_df = df_shuffled.drop(columns=['species', 'species_num']) # shape (N,D)
    y_class_labels = df_shuffled['species_num']                 # shape (N,)
    Y_one_hot = pd.get_dummies(y_class_labels)                  # shape (N,C)

    num_train = int((percent_train/100) * num_samples)
    num_test = num_samples - num_train # variable not used

    X_train_df = X_df.iloc[:num_train, :]                       # shape (N_train, D)
    Y_train_df = Y_one_hot.iloc[:num_train]                     # shape (N_train, C)

    X_test_df = X_df.iloc[num_train:, :]                        # shape (N_test, D)
    Y_test_df = Y_one_hot.iloc[num_train:]                      # shape (N_test, C)

    return X_train_df, Y_train_df, X_test_df, Y_test_df


def standardize_dataset(X_train_df, X_test_df):
    """
    Standardize the dataset.

    :param X_train_df:
    :param X_test_df:
    :return:
    """
    X_train_standardized = (X_train_df - X_train_df.mean()) / X_train_df.std() # shape (N_train, D)
    X_test_standardized = (X_test_df - X_train_df.mean()) / X_train_df.std()   # shape (N_test, D)

    return X_train_standardized, X_test_standardized

def initialize_parameters(num_features, num_classes):
    """
    Initializes the parameters (weights and biases) of the model.

    :param num_features:
    :param num_classes:
    :return:
    """
    np.random.seed(42)

    weight_matrix = np.random.randn(num_features, num_classes) / np.sqrt(num_features) # shape (D,C), keep numbers small
    bias_vector = np.zeros(num_classes) # shape (C,)

    return weight_matrix, bias_vector


def forward_pass(W, b, X):
    """
    Given samples, predicts a class. Computes logits and outputs
    probabilities for the provided data.

    :param W:
    :param b:
    :param X:
    :return:
    """
    Z = X @ W + b # raw logits, shape (N_train,C)

    Z_stable = Z - np.max(Z, axis=1, keepdims=True)
    Z_exp = np.exp(Z_stable)
    Z_sum = np.sum(Z_exp, axis=1, keepdims=True)

    Y_hat = Z_exp / Z_sum # probabilities, shape (N_train,C)

    return Y_hat


def cross_entropy_loss(Y_true, Y_hat):
    """
    Computes the average cross entropy loss per sample. Expects one-hot encoding.

    :param Y_hat:
    :param Y_true:
    :return:
    """
    N, _ = Y_hat.shape
    #return -np.sum(np.log(Y_hat[np.arange(N), y_train])) / N # class labels
    return -np.sum(Y_true * np.log(Y_hat)) / N               # one-hot


def backward_propagation(X, Y_true, Y_hat):
    """
    Back propagation.

    :return:
    """
    N_test, _ = Y_true.shape

    dLoss_dZ = (Y_hat - Y_true) / N_test
    dLoss_dW = X.T @ dLoss_dZ                          # loss gradient WRT W

    #dLoss_db = np.sum(dLoss_dZ, axis=1, keepdims=True)  # this is wrong... class labels
    dLoss_db = np.sum(Y_hat - Y_true, axis=0) / N_test # one-hot loss gradient WRT b, shape (C,)

    return dLoss_dW, dLoss_db


def update(W, b, dW, db):
    """
    Uses gradient descent to update weights and biases.

    :param W:
    :param b:
    :param dW:
    :param db:
    :return:
    """
    alpha = .1 # learning rate

    W = W - alpha * dW
    b = b - alpha * db

    return W, b


def train_model(W, b, X_train, Y_train, X_test, Y_test, num_epochs=10000):
    training_loss = np.zeros(num_epochs)
    testing_loss = np.zeros(num_epochs)
    correct_predictions = np.zeros(num_epochs)

    for epoch in range(num_epochs):
        Y_hat_train = forward_pass(W, b, X_train)
        training_loss[epoch] = cross_entropy_loss(Y_train, Y_hat_train)
        dW, db = backward_propagation(X_train, Y_train, Y_hat_train)
        W, b = update(W, b, dW, db)

        Y_hat_test = forward_pass(W, b, X_test)
        testing_loss[epoch] = cross_entropy_loss(Y_test, Y_hat_test)
        correct_predictions[epoch] = (int)(np.sum(np.argmax(Y_test, axis=1) == np.argmax(Y_hat_test, axis=1)))

    plt.figure()
    x = range(num_epochs)

    plt.subplot(2, 1, 1)
    y1 = training_loss
    y2 = testing_loss
    plt.plot(x, y1, label='Training loss')
    plt.plot(x, y2, label='Testing loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss Over Time')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.ylim(0, len(Y_test))
    y3 = correct_predictions
    plt.plot(x, y3, label='Correct Predictions')
    plt.xlabel('epoch')
    plt.ylabel('correct predictions')
    plt.title('Accuracy Over Time')

    plt.tight_layout(pad=2)
    plt.show()

    return W, b


def evaluate_model(W, b, X_test, Y_test):
    """

    :param W:
    :param b:
    :param X_test:
    :param y_test:
    :return:
    """
    Y_hat = forward_pass(W, b, X_test)

    y_pred = np.argmax(Y_hat, axis=1)
    y_true = np.argmax(Y_test, axis=1)
    correct = y_pred == y_true
    num_correct = np.sum(correct)

    return num_correct


def main():
    df = load_data('../datasets/iris.csv')
    df['species_num'] = df['species'].map({'setosa': 0, 'versicolor': 1, 'virginica': 2})

    # visualize data
    plt.figure()
    plt.scatter(df['petal_length'], df['petal_width'], c=df['species_num'], cmap='viridis')
    plt.xlabel('petal length')
    plt.ylabel('petal width')
    plt.show()

    plt.figure()
    plt.scatter(df['sepal_length'], df['sepal_width'], c=df['species_num'], cmap='viridis')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.show()

    X_train_df, Y_train_df, X_test_df, Y_test_df = split_data(df)
    X_train_standardized, X_test_standardized = standardize_dataset(X_train_df, X_test_df)

    X_train = X_train_standardized.to_numpy()
    Y_train = Y_train_df.to_numpy()

    W, b = initialize_parameters(4, 3)
    W, b = train_model(W, b, X_train, Y_train, X_test_standardized.to_numpy(), Y_test_df.to_numpy(), 1000) # 100% accuracy with 1_000_000 epochs

    num_correct = evaluate_model(W, b, X_test_standardized.to_numpy(), Y_test_df.to_numpy())
    print(f'Final model predictions: {num_correct}/{Y_test_df.shape[0]} correct')


if __name__ == "__main__":
    main()
