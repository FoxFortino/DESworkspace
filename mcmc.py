from time import time

import numpy as np
import astropy.units as u
import astropy.io.fits as pf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import forAustin as fa
import gpr

class MCMC(object):
    def __init__(self, nll_fn, data, theta, step_size, names=None, seed=314159):
        """Markov Chain Monte Carlo Metropolis Hastings Algorithm.
        
        Parameters
        nll_fn    : negative log likelihood function
        data      : data used to evaluate the likelihood in whichever form nll_fn requires
        theta     : list of model parameters used to evaluate the likelihood
        step_size : list step size for each model parameter (theta)
        names     : list of names for histogram and trace plots
        seed      : random seed that determines random walk
        """
        self.nll_fn = nll_fn
        self.data = data
        self.theta = np.array(theta)
        self.step_size = np.array(step_size)
        if names is None:
            names = ["Parameter {:d}".format(k+1) for k in range(self.nparams)]
        self.names = names
        self.rng = np.random.RandomState(seed)
        
        self.nParams = self.theta.shape[0]
        self.current_nll = nll_fn(self.theta, self.data)
        self.nAccept = 0
        self.samples = []

    def step(self, save=True):
        theta_new =  self.rng.normal(loc=self.theta, scale=self.step_size, size=self.nParams)
        while np.any(theta_new[:3] <= 0): # Make sure that first three parameters never go below zero
            theta_new =  self.rng.normal(loc=self.theta, scale=self.step_size, size=self.nParams)
        
        nll_new = self.nll_fn(theta_new, self.data)
        acceptance_probability = np.min((1, np.exp(nll_new - self.current_nll)))
        random_acceptance_probability = self.rng.uniform()
        
        if acceptance_probability > random_acceptance_probability:
            take_step = True
            self.current_nll = nll_new
            self.theta = theta_new
            if save:
                self.nAccept += 1
        else:
            take_step = False
        
        if save:
            self.samples.append(self.theta)
        
    def burn(self, nBurn):
        for i in range(nBurn):
            self.step(save=False)

    def run(self, nSteps):
        for i in range(nSteps):
            self.step()

    def accept_fraction(self):
        """Returns the fraction of candidate steps that were accpeted so far."""
        if len(self.samples) > 0:
            return float(self.nAccept) / len(self.samples)
        else:
            return 0.
        
    def clear(self, step_size=None, theta=None):
        """Clear the list of stored samples from any runs so far.
        
        You can also change the step_size to a new value at this time by giving a step_size as an
        optional parameter value.
        
        In addition, you can reset theta to a new starting value if theta is not None.
        """
        if step_size is not None:
            assert len(step_size) == self.nParams
            self.step_size = np.array(step_size)
        if theta is not None:
            assert len(theta) == self.nParams
            self.theta = np.array(theta)
            self.current_nll = self.nll_fn(self.data, self.theta)
        self.samples = []
        self.nAccept = 0
        
    def get_samples(self):
        """Return the sampled theta values at each step in the chain as a 2d numpy array."""
        return np.array(self.samples)
        
    def plot_hist(self):
        """Plot a histogram of the sample values for each parameter in the theta vector."""
        all_samples = self.get_samples()
        for k in range(self.nParams):
            theta_k = all_samples[:,k]
            plt.hist(theta_k, bins=100)
            plt.xlabel(self.names[k])
            plt.ylabel("N Samples")
            plt.show()
        
    def plot_samples(self):
        """Plot the sample values over the course of the chain so far."""
        all_samples = self.get_samples()
        for k in range(self.nParams):
            theta_k = all_samples[:,k]
            plt.plot(range(len(theta_k)), theta_k)
            plt.xlabel("Step in chain")
            plt.ylabel(self.names[k])
            plt.show()

    def calculate_mean(self, weight=None):
        """Calculate the mean of each parameter according to the samples taken so far.
        
        Optionally, provide a weight array to weight the samples.
        
        Returns the mean values as a numpy array.
        """

        return np.average(np.array(self.samples), axis=0, weights=weight)
    
    def calculate_cov(self, weight=None):
        """Calculate the covariance matrix of the parameters according to the samples taken so far.

        Optionally, provide a weight array to weight the samples.
        
        Returns the covariance matrix as a 2d numpy array.
        """

        return np.cov(np.array(self.samples).T, fweights=weight)