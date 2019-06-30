from time import time

import numpy as np
import astropy.units as u
import astropy.io.fits as pf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import forAustin as fa
import gpr

class MCMC(object):
    def __init__(self, log_likelihood, data, theta, step_size, names=None, seed=314159):
        self.log_likelihood = log_likelihood
        self.data = data
        self.theta = np.array(theta)
        self.nparams = len(theta)
        self.step_size = np.array(step_size)
        self.rng = np.random.RandomState(seed)
        self.naccept = 0
        self.current_loglike = log_likelihood(self.data, self.theta)
        self.samples = []
        if names is None:
            names = ["Parameter {:d}".format(k+1) for k in range(self.nparams)]
        self.names = names      
        
        self.theta_news = np.array([0, 0, 0, 0])

    def step(self, save=True):
        """Take a single step in the chain"""
        theta_new =  self.rng.normal(loc=self.theta, scale=self.step_size, size=self.theta.shape[0])
        while np.any(theta_new[:3] < 0):
            theta_new =  self.rng.normal(loc=self.theta, scale=self.step_size, size=self.theta.shape[0])
        self.theta_news = np.vstack((self.theta_news, theta_new))
        p_new = self.log_likelihood(self.data, theta_new)
        ratio = np.exp(p_new - self.current_loglike)

        if ratio >= 1.0:
            take_step = 1
        else:
            stepran = self.rng.uniform()
            if stepran < ratio:
                take_step = 1
            else:
                take_step = 0

        if take_step:
            self.current_loglike = p_new
            self.theta = theta_new

        if save:
            self.samples.append(self.theta)
            
        if save and take_step:
            self.naccept += 1
        
    def burn(self, nburn):
        """Take nburn steps, but don't save the results"""
        for i in range(nburn):
            self.step(save=False)

    def run(self, nsteps):
        """Take nsteps steps"""
        for i in range(nsteps):
            self.step()

    def accept_fraction(self):
        """Returns the fraction of candidate steps that were accpeted so far."""
        if len(self.samples) > 0:
            return float(self.naccept) / len(self.samples)
        else:
            return 0.
        
    def clear(self, step_size=None, theta=None):
        """Clear the list of stored samples from any runs so far.
        
        You can also change the step_size to a new value at this time by giving a step_size as an
        optional parameter value.
        
        In addition, you can reset theta to a new starting value if theta is not None.
        """
        if step_size is not None:
            assert len(step_size) == self.nparams
            self.step_size = np.array(step_size)
        if theta is not None:
            assert len(theta) == self.nparams
            self.theta = np.array(theta)
            self.current_loglike = self.log_likelihood(self.data, self.theta)
        self.samples = []
        self.naccept = 0
        
    def get_samples(self):
        """Return the sampled theta values at each step in the chain as a 2d numpy array."""
        return np.array(self.samples)
        
    def plot_hist(self):
        """Plot a histogram of the sample values for each parameter in the theta vector."""
        all_samples = self.get_samples()
        for k in range(self.nparams):
            theta_k = all_samples[:,k]
            plt.hist(theta_k, bins=100)
            plt.xlabel(self.names[k])
            plt.ylabel("N Samples")
            plt.show()
        
    def plot_samples(self):
        """Plot the sample values over the course of the chain so far."""
        all_samples = self.get_samples()
        for k in range(self.nparams):
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