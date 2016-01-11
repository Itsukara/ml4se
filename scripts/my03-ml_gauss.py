import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import Series, DataFrame
from numpy.random import normal
from scipy.stats import norm

def main():
    fig = plt.figure()
    # the number of sample
    for c, datapoints in enumerate([2, 4, 10, 100]):
        ds = normal(loc=0, scale=1, size=datapoints)
        # estimate arithematic mean
        mu = np.mean(ds)
        # estimate standard deviation
        sigma = np.sqrt(np.var(ds))

        subplot = fig.add_subplot(2, 2, c + 1)
        subplot.set_title("N=%d" % datapoints)
        linex = np.arange(-10, 10.1, 0.1)
        orig = norm(loc=0, scale=1)
        # norm.pdf
        # Probability dencity function
        subplot.plot(linex, orig.pdf(linex), color='green', linestyle='--')

        est = norm(loc=mu, scale=np.sqrt(sigma))
        label = "Sigma=%.2f" % sigma
        subplot.plot(linex, est.pdf(linex), color="red", label=label)
        subplot.legend(loc=1)

        subplot.scatter(ds, orig.pdf(ds), marker='o', color='blue')
        subplot.set_xlim(-4, 4)
        subplot.set_ylim(0)
    fig.show()

if __name__ == "__main__":
    main()
