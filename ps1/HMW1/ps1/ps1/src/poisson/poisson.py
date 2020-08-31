import numpy as np
import util
import matplotlib.pyplot as plt
#from scipy.misc import factorial

def main(lr, train_path, eval_path, save_path):
    """Problem: Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        save_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    # Run on the validation set, and use np.savetxt to save outputs to save_path
    x_val, y_val = util.load_dataset(eval_path, add_intercept=True)
    
    clf = PoissonRegression(step_size=lr)
    clf.fit(x_train,y_train)
    y_predict = clf.predict(x_val)
    
    np.savetxt(save_path, y_predict)
        
    plt.scatter(y_val, y_predict, c='m', alpha=0.5)
    plt.title('Scatter plot predicted/real values')
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.savefig(save_path[:-4]+"_exo2")
    plt.show()
    # *** END CODE HERE ***


class PoissonRegression:
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, step_size=1e-5, max_iter=10000000, eps=1e-5,
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

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        iteration = 1
        if self.theta == None:
            self.theta = np.zeros((x.shape[1],))
        while iteration <= self.max_iter:
            g_theta = np.exp(x @ self.theta)
            d_theta = self.step_size*np.transpose(x) @ (y - g_theta)
            self.theta += d_theta
            if np.linalg.norm(d_theta, 1) <= self.eps:
                break
            if self.verbose:
                print("Loss value is:", self.theta)
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Floating-point prediction for each input, shape (n_examples,).
        """
        # *** START CODE HERE ***
        y_predicted = np.exp(x @ self.theta)
        return y_predicted
        # *** END CODE HERE ***

if __name__ == '__main__':
    main(lr=1e-5,
        train_path='train.csv',
        eval_path='valid.csv',
        save_path='poisson_pred.txt')
