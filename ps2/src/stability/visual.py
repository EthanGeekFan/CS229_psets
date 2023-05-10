# visualize dataset A and B
import util
import numpy as np
import matplotlib.pyplot as plt


def plot(x, y, save_path):
    """Plot dataset and fitted logistic regression parameters.

    Args:
        x: Matrix of training examples, one per row.
        y: Vector of labels in {0, 1}.
        theta: Vector of parameters for logistic regression model.
        save_path: Path to save the plot.
        correction: Correction factor to apply (Problem 2(e) only).
    """
    # Plot dataset
    plt.figure()
    plt.plot(x[y == 1, -2], x[y == 1, -1], 'bx', linewidth=2)
    plt.plot(x[y == 0, -2], x[y == 0, -1], 'go', linewidth=2)

    # Add labels and save to disk
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.savefig(save_path)


def main():
    Xa, Ya = util.load_csv('ds1_a.csv', add_intercept=True)
    Xb, Yb = util.load_csv('ds1_b.csv', add_intercept=True)
    # plot dataset A
    plot(Xa, Ya, save_path='ds1_a.png')
    # plot dataset B
    plot(Xb, Yb, save_path='ds1_b.png')


if __name__ == '__main__':
    main()