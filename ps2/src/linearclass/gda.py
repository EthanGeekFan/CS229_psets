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
    model = GDA()
    model.fit(x_train, y_train)

    # Plot decision boundary on validation set
    x_val, y_val = util.load_dataset(valid_path, add_intercept=False)
    plot_path = save_path.replace('.txt', '.png')
    util.plot(x_val, y_val, model.theta, plot_path)

    # Use np.savetxt to save outputs from validation set to save_path
    x_val = util.add_intercept(x_val)
    pred = model.predict(x_val)
    np.savetxt(save_path, pred)
    # print accuracy
    print("Accuracy:", np.mean((pred > 0.5) == y_val))
    # *** END CODE HERE ***


class GDA:
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, step_size=1, max_iter=10000, eps=1e-5,
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
        m, n = x.shape

        # Find phi, mu_0, mu_1, and sigma
        phi = np.sum(y) / m
        mu_0 = np.sum(x[y == 0], axis=0) / np.sum(y == 0)
        mu_1 = np.sum(x[y == 1], axis=0) / np.sum(y == 1)
        sigma = np.zeros((n, n))
        for i in range(m):
            if y[i] == 0:
                sigma += np.outer(x[i] - mu_0, x[i] - mu_0)
            else:
                sigma += np.outer(x[i] - mu_1, x[i] - mu_1)
        sigma /= m

        # Write theta in terms of the parameters
        sigma_inv = np.linalg.inv(sigma)
        self.theta = np.zeros(n + 1)
        self.theta[0] = np.log(phi / (1 - phi)) + 0.5 * (mu_0 @ sigma_inv @ mu_0 - mu_1 @ sigma_inv @ mu_1)
        self.theta[1:] = mu_1 @ sigma_inv - mu_0 @ sigma_inv

        if self.verbose:
            print("Final theta:", self.theta)
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (n_examples, dim).

        Returns:
            Outputs of shape (n_examples,).
        """
        # *** START CODE HERE ***
        return 1 / (1 + np.exp(-x @ self.theta))
        # *** END CODE HERE

if __name__ == '__main__':
    main(train_path='ds1_train.csv',
         valid_path='ds1_valid.csv',
         save_path='gda_pred_1.txt')

    main(train_path='ds2_train.csv',
         valid_path='ds2_valid.csv',
         save_path='gda_pred_2.txt')
