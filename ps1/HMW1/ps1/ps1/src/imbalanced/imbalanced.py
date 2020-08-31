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
    print("Vanilla Logistic Regression:")
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)
    
    x_val, y_val = util.load_dataset(validation_path, add_intercept=True)
    
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_val)
    
    np.savetxt(output_path_naive, y_predict)
    y_predict = y_predict >= 0.5
    util.plot(x_val, y_predict, clf.theta, output_path_naive[:-4])
    
    accuracy = np.mean(y_predict == y_val)
    A_0 = np.sum((y_predict == 0)*(y_val == 0))/np.sum(y_val == 0)
    A_1 = np.sum((y_predict == 1)*(y_val == 1))/np.sum(y_val == 1)
    balanced_accuracy = 0.5*(A_0 + A_1)
    print("Accuracy: {},\nAccuracy for class 0: {},\nAccuracy for class 1: {},"
                                  "\nBalanced Accuracy: {}".format(accuracy, A_0, A_1, balanced_accuracy))

    #plot the real expected outcome from the validation:
    util.plot(x_val, y_val, clf.theta, output_path_naive[:-4] + "validation")
    # Part (d): Upsampling minority class
    # Make sure to save predicted probabilities to output_path_upsampling using np.savetxt()
    # Repeat minority examples 1 / kappa times    
    num_add = int(1/kappa) - 1
    
    x_train = np.concatenate((x_train, np.repeat(x_train[y_train == 1,:], num_add, axis=0)), axis=0)
    y_train = np.concatenate((y_train, np.repeat(y_train[y_train == 1], num_add, axis=0)), axis=0)
    
    x_val, y_val = util.load_dataset(validation_path, add_intercept=True)
    
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_val)
    
    np.savetxt(output_path_upsampling, y_predict)
    y_predict = y_predict >= 0.5
    util.plot(x_val, y_predict, clf.theta, output_path_upsampling[:-4])
    
    accuracy = np.mean(y_predict == y_val)
    A_0 = np.sum((y_predict == 0)*(y_val == 0))/np.sum(y_val == 0)
    A_1 = np.sum((y_predict == 1)*(y_val == 1))/np.sum(y_val == 1)
    balanced_accuracy = 0.5*(A_0 + A_1)
    print("Accuracy: {},\nAccuracy for class 0: {},\nAccuracy for class 1: {},"
                                  "\nBalanced Accuracy: {}".format(accuracy, A_0, A_1, balanced_accuracy))
    #plot the real expected outcome from the validation:
    util.plot(x_val, y_val, clf.theta, output_path_upsampling[:-4] + "validation")
    # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='train.csv',
        validation_path='validation.csv',
        save_path='imbalanced_X_pred.txt')
