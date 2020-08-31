# Important note: you do not have to modify this file for your homework.

import util
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def calc_grad(X, Y, theta):
    """Compute the gradient of the loss with respect to theta."""
    count, _ = X.shape

    probs = 1. / (1 + np.exp(-X.dot(theta)))
    grad = (Y - probs).dot(X) #- 0.01*theta

    return grad

def sigmoid(x):
    return 1/(1+np.exp(-x))

def logistic_regression(X, Y):
    """Train a logistic regression model."""
    theta = np.zeros(X.shape[1])
    learning_rate = 0.1
    thetas =[]
    i = 0
    # cost = []
    while True:
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta)
        theta = theta + learning_rate * grad
        if i % 10000 == 0:
            print('Finished %d iterations' % i)
            # cost.append(-1/X.shape[0]*(Y@np.log(sigmoid(X.dot(theta))) + (1-Y)@np.log(1 - sigmoid(X.dot(theta)))) \
            #             + 0.005*np.linalg.norm(theta, 2))
            thetas.append(theta)
        if np.linalg.norm(prev_theta - theta) < 1e-15 or i == 30000:
            print('Converged in %d iterations' % i)
            break
        # if i % 5000 == 0:
        #     plt.plot([i for i in range(i//100)], cost, '--r', label='cost function')
        #     plt.xlabel('iterations (x100)')
        #     plt.ylabel('cost value')
        #     plt.title("cost function vs iterations")
        #     plt.legend()
        #     plt.grid()
        #     plt.savefig('cost_function_vs_time')
        #     plt.show()
    return thetas

def J(th0, th1, X, Y):
    # J = np.zeros((len(th0), len(th1), len(th2)))
    J = np.zeros_like(th0)
    for i,x in enumerate(th0[0,:]):
        for j,y in enumerate(th1[0,:]):
            theta = [x,y]
            J[i,j] = (Y@np.log(sigmoid(X.dot(theta))) + (1-Y)@np.log(1 - sigmoid(X.dot(theta))))
    # for i in range(len(th0)):
    #     for j in range(len(th1)):
    #         for k in range(len(th2)):
    #             theta = [th0[i], th1[j], th2[k]]
    #             J[i, j, k] =  (Y@np.log(sigmoid(X.dot(theta))) + (1-Y)@np.log(1 - sigmoid(X.dot(theta))))
    return J

def plot_cost(X, Y):
    
    theta0 = np.linspace(-100, 100, 100)
    theta1 = np.linspace(-100, 100, 100)
    # theta2 = np.linspace(-100, 100, 100)

    th0, th1 = np.meshgrid(theta0, theta1)
    J_theta = -1/X.shape[0]*J(th0, th1, X[:,1:], Y)
    # J_theta = -1/X.shape[0]*J(theta0, theta1, theta2, X, Y)
    
    fig = plt.figure(figsize=(8,8))
    ax = plt.axes(projection='3d')
    ax.contour3D(th0, th1, J_theta, 50, cmap='RdBu')
    # ax.scatter(theta0, theta1, theta2, c=J_theta, cmap=plt.hot())
    ax.set_xlabel('theta0')
    ax.set_ylabel('theta1')
    ax.set_zlabel('theta2');
    ax.view_init(0, 0)
    
    

def main():
    print('==== Training model on data set A ====')
    Xa, Ya = util.load_csv('ds1_a.csv', add_intercept=True)
    plot_cost(Xa, Ya)
    # util.plot_points(Xa[:,1:], Ya, theta)
    plt.show()
    thetas = logistic_regression(Xa, Ya)
    # util.plot_points(Xa[:,1:], Ya, thetas)
    
    print('\n==== Training model on data set B ====')
    Xb, Yb = util.load_csv('ds1_b.csv', add_intercept=True)
    plot_cost(Xb, Yb)
    # Xb += np.random.normal(scale=0.03, size=Xb.shape)
    # util.plot_points(Xb[:,1:], Yb)
    plt.show()
    thetas = logistic_regression(Xb, Yb)
    # util.plot_points(Xb[:,1:], Yb, thetas)


if __name__ == '__main__':
    main()
