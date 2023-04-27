import numpy as np
import util
import sys
from random import random

sys.path.append('../linearclass')

### NOTE : You need to complete logreg implementation first!

from logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/save_path
WILDCARD = 'X'
# Ratio of class 0 to class 1
kappa = 0.1

def main(train_path, validation_path, save_path):
    """Problem 2: Logistic regression for imbalanced labels.

    Run under the following conditions:
        1. naive logistic regression
        2. upsampling minority class

    Args:
        train_path: Path to CSV file containing training set.
        validation_path: Path to CSV file containing validation set.
        save_path: Path to save predictions.
    """
    output_path_naive = save_path.replace(WILDCARD, 'naive')
    output_path_upsampling = save_path.replace(WILDCARD, 'upsampling')

    # *** START CODE HERE ***
    # Part (b): Vanilla logistic regression
    # Make sure to save predicted probabilities to output_path_naive using np.savetxt()
    model = LogisticRegression()
    model.fit(*util.load_dataset(train_path))
    x_val, y_val = util.load_dataset(validation_path)
    y_pred = model.predict(x_val)
    np.savetxt(output_path_naive, y_pred)
    # accuracy
    y_pred = np.round(y_pred)
    accuracy = np.mean(y_pred == y_val)
    print('Accuracy: {}'.format(accuracy))
    # accuracy for 2 classes
    acc_0 = np.sum((y_val == 0) & (y_pred == 0)) / np.sum(y_val == 0)
    acc_1 = np.sum((y_val == 1) & (y_pred == 1)) / np.sum(y_val == 1)
    # balanced accuracy
    balanced_accuracy = (acc_0 + acc_1) / 2
    print('Balanced accuracy: {}'.format(balanced_accuracy))
    print('Class Accuracy: {}'.format((acc_0, acc_1)))
    # plot the validation set with x1 on x-axis and x2 on y-axis and color the points according to their labels
    util.plot(x_val, y_val, model.theta, 'vanilla.png')
    # Part (d): Upsampling minority class
    # Make sure to save predicted probabilities to output_path_upsampling using np.savetxt()
    # Repeat minority examples 1 / kappa times
    x_train, y_train = util.load_dataset(train_path)
    rho = np.mean(y_train == 1)
    kappa = rho / (1 - rho)
    print('1/kappa: {}'.format(1 / kappa))
    x_train = np.concatenate((x_train, np.repeat(x_train[y_train == 1], int(1 / kappa), axis=0)))
    y_train = np.concatenate((y_train, np.repeat(y_train[y_train == 1], int(1 / kappa), axis=0)))
    model = LogisticRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_val)
    np.savetxt(output_path_upsampling, y_pred)
    # accuracy
    y_pred = np.round(y_pred)
    accuracy = np.mean(y_pred == y_val)
    print('Accuracy: {}'.format(accuracy))
    # accuracy for 2 classes
    acc_0 = np.sum((y_val == 0) & (y_pred == 0)) / np.sum(y_val == 0)
    acc_1 = np.sum((y_val == 1) & (y_pred == 1)) / np.sum(y_val == 1)
    # balanced accuracy
    balanced_accuracy = (acc_0 + acc_1) / 2
    print('Balanced accuracy: {}'.format(balanced_accuracy))
    print('Class Accuracy: {}'.format((acc_0, acc_1)))
    # plot the validation set with x1 on x-axis and x2 on y-axis and color the points according to their labels
    util.plot(x_val, y_val, model.theta, 'upsampling.png')
    # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='train.csv',
        validation_path='validation.csv',
        save_path='imbalanced_X_pred.txt')
