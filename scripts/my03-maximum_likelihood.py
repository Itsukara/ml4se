import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame
from numpy.random import normal

N = 300
M = [0, 1, 3, 9]

def create_dataset(n):
    """
    Create dataset
    {x_n, y_n} (n=1...N)
    """
    dataset = DataFrame(columns=['x', 'y'])
    for i in range(n):
        x = i / (n - 1)
        y = np.sin(2 * np.pi * x) + normal(scale=0.3)
        dataset = dataset.append(Series([x, y], index=['x', 'y']),
                                ignore_index=True)
    return dataset

def log_likelihood(dataset, f):
    """
    Caliculate Maximum log likehood
    """
    dev = 0.
    n = len(dataset)
    for index, line in dataset.iterrows():
        x, y = line.x, line.y
        dev += (y - f(x)) ** 2
    err = dev * 0.5
    beta = n / dev
    lp = - beta * err + 0.5 * n * np.log(0.5 * beta / np.pi)
    return lp

def resolve(dataset, m):
    t = dataset.y
    phi = DataFrame()
    for i in range(0, m + 1):
        p = dataset.x ** i
        p.name = 'x**%d' * i
        phi = pd.concat([phi, p], axis=1)
    tmp = np.linalg.inv(np.dot(phi.T, phi))
    ws = np.dot(np.dot(tmp, phi.T), t)

    def f(x):
        y = 0.
        for i, w in enumerate(ws):
            y += w * (x ** i)
        return y

    sigma2 = 0.
    for index, line in dataset.iterrows():
        sigma2 += (f(line.x) - line.y) ** 2
    sigma2 /= len(dataset)

    return f, ws, np.sqrt(sigma2)

def main():
    train_set = create_dataset(N)
    test_set = create_dataset(N)
    df_ws = DataFrame()

    fig = plt.figure()
    for c, m in enumerate(M):
        f, ws, sigma = resolve(train_set, m)
        df_ws = df_ws.append(Series(ws, name="M=%d" % m))

        subplot = fig.add_subplot(2, 2, c + 1)
        subplot.set_xlim(-0.05, 1.05)
        subplot.set_ylim(-1.5, 1.5)
        subplot.set_title("M=%d" % m)

        subplot.scatter(train_set.x, train_set.y, marker='o', color='blue')

        linex = np.linspace(0,1,101)
        liney = np.sin(2*np.pi*linex)
        subplot.plot(linex, liney, color="green", linestyle='--')

        linex = np.linspace(0, 1, 101)
        liney = f(linex)
        label = "Sigma=%.2f" % sigma
        subplot.plot(linex, liney, color='red', label=label)
        subplot.plot(linex, liney + sigma, color='red', linestyle='--')
        subplot.plot(linex, liney - sigma, color='red', linestyle='--')
        subplot.legend(loc=1)

    fig.show()

    df = DataFrame()
    train_mlh = []
    test_mlh = []
    for m in range(0, 9):
        f, ws, sigma = resolve(train_set, m)
        train_mlh.append(log_likelihood(train_set, f))
        test_mlh.append(log_likelihood(test_set, f))
    df = pd.concat([df,
                    DataFrame(train_mlh, columns=['Training set']),
                    DataFrame(test_mlh, columns=['Test set'])],
                    axis=1)
    df.plot(title='Log likehood for N=%d' % N, grid=True, style=['-', '--'])
    plt.show()

if __name__ == "__main__":
    main()
