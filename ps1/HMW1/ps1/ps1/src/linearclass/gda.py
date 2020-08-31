import numpy as np
import util


def main(train_path, valid_path, save_path):
    """Problem: Gaussian discriminant analysis (GDA)
    
    Args:
        train_path: Path to CSV file containing dataset for training.
        valid_path: Path to CSV file containing dataset for validation.
        save_path: Path to save predicted probabilities using np.savetxt().
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Train a GDA classifier
    # Plot decision boundary on validation set
    # Use np.savetxt to save outputs from validation set to save_path
    x_val, y_val = util.load_dataset(valid_path, add_intercept=False)
        
    ###decomment to normalize the training and validation sets to improve the GDA performance:
#    x_train = (x_train - np.mean(x_train, axis=0))/np.std(x_train, axis=0)
#    x_val = (x_val - np.mean(x_val, axis=0))/np.std(x_val, axis=0)
    
#    x_train = (x_train - np.min(x_train, axis=0))/(np.max(x_train, axis=0) - np.min(x_train, axis=0))
#    x_val = (x_val - np.min(x_val, axis=0))/(np.max(x_val, axis=0) - np.min(x_val, axis=0))
    
    
    clf = GDA()
    clf.fit(x_train, y_train)
    y_predict = clf.predict(x_val)
    
    np.savetxt(save_path, y_predict)
    util.plot(x_val, (y_predict >= 0.5), clf.theta, save_path[:-4]+ "validation_expected")
    #plotting the real distribution
    util.plot(x_val, y_val, clf.theta, save_path[:-4] + "validation_real")
    # *** END CODE HERE ***


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=0.01, max_iter=10000, eps=1e-5,
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
        """Fit a GDA model to training set given by x and y by updating
        self.theta.

        Args:
            x: Training example inputs. Shape (n_examples, dim).
            y: Training example labels. Shape (n_examples,).
        """
        # *** START CODE HERE ***
        # Find phi, mu_0, mu_1, and sigma
        n_examples = x.shape[0]
        phi = 1/n_examples*np.sum(y)
        mu_0 = np.transpose(1-y)@x/np.sum(y==0)
        mu_1 = np.transpose(y)@x/np.sum(y==1)
        mu = (1-y).reshape((-1,1))@np.transpose(mu_0.reshape((-1,1))) + y.reshape((-1,1))@np.transpose(mu_1.reshape((-1,1)))
        sigma = 1/n_examples * np.dot(np.transpose(x-mu), x - mu)   
        # Write theta in terms of the parameters
        inv_sigma = np.linalg.inv(sigma)
        theta_0 = 0.5*(np.dot(np.transpose(mu_0), inv_sigma@mu_0) - np.dot(np.transpose(mu_1), inv_sigma@mu_1) 
                                + 2*np.log(phi/(1-phi)))
        theta = np.dot(inv_sigma, mu_1 - mu_0)
        self.theta = np.zeros((x.shape[1] + 1,))
        self.theta[0] = theta_0
        self.theta[1:] = theta
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        y_predicted = 1/(1 + np.exp(-np.dot(x, self.theta[1:]) - self.theta[0]))
        return y_predicted
        # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2.txt')
