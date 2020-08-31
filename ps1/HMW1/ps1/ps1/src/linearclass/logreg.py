import numpy as np
import util
import os


def main(train_path, valid_path, save_path):
    """Problem: Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Train a logistic regression classifier
    # Plot decision boundary on top of validation set set
    # Use np.savetxt to save predictions on eval set to save_path
    x_val, y_val = util.load_dataset(valid_path, add_intercept=True)

    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_val)

    np.savetxt(save_path, y_predict)
    util.plot(x_val, y_predict >= 0.5, clf.theta, save_path[:-4])
    #plot the real expected outcome from the validation:
    util.plot(x_val, y_val, clf.theta, save_path[:-4] + "validation")
    # *** END CODE HERE ***


class LogisticRegression:
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=1000000, eps=1e-5,
                 theta_0=None, verbose=True):
        """
        Args:
            step_size: Step size for iterative solvers only.
            max_iter: Maximum number of iterations for the solver.
            eps: Threshold for determining convergence.
            theta_0: Initial guess for theta. If None, use the zero vector.
            verbose: Print loss values during training.
        """
        self.theta = theta_0
        self.step_size = step_size
        self.max_iter = max_iter
        self.eps = eps
        self.verbose = verbose

    def sigmoid(self, x):
        return 1/(1+np.exp(-x) + 1e-8)

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        num_examples = x.shape[0]
        num_features = x.shape[1]
        iteration = 1
        if self.theta == None:
            self.theta = np.zeros((num_features,))
        while iteration <= self.max_iter:
            h_theta = np.dot(x, self.theta)
            g_theta = self.sigmoid(h_theta)
            J_cost = -np.mean(y*np.log(g_theta) + (1 - y)*np.log(1 - g_theta))
            H = 1/num_examples*(np.dot(np.transpose(g_theta*(1-g_theta))*np.transpose(x), x))
            J_prime = - 1/num_examples*np.dot(np.transpose(y - g_theta), x)
            d_theta = - np.linalg.solve(H, J_prime)
            self.theta += d_theta
            if np.linalg.norm(d_theta, 1) < self.eps:
                break
            if self.verbose:
                print("Loss value: ", J_cost)
            iteration += 1
        # *** END CODE HERE ***

    def predict(self, x):
        """Return predicted probabilities given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        y_predicted = self.sigmoid(np.dot(x, self.theta))
        return y_predicted
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='logreg_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='logreg_pred_2.txt')
