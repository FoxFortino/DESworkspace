import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import astropy.units as u
import astropy.constants as c
import astropy.io.fits as pf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import forAustin as fa

tfd = tfp.distributions
psd_kernels = tfp.positive_semidefinite_kernels

class GPR_TFP(object):
    def extract_exposure(self):
        """Extracts all data from a specified exposure (self.nExposure). Currently only supports extracting one exposure's worth of data."""
        if self.verbose: print("Extracting exposure from fits file...")
        self.fits = pf.open(self.datafile)
        if self.verbose: self.fits.info(); print()
        self.exposure = fa.getExposure(self.fits, self.nExposure, polyOrder=3)
    
    def extract_data(self):
        """Extract star positions, residuals, and measurement error from one exposure."""
        if self.verbose: print("Extracting exposure data...")
        ind_hasGaia = np.where(self.exposure['hasGaia'])[0]
        u = np.take(self.exposure['u'], ind_hasGaia)
        v = np.take(self.exposure['v'], ind_hasGaia)
        dx = np.take(self.exposure['dx'], ind_hasGaia)
        dy = np.take(self.exposure['dy'], ind_hasGaia)
        self.E = np.take(self.exposure['measErr'], ind_hasGaia)
        
        # Extract only data in a certain region of the exposure as specified by self.sample.
        if self.sample is not None:
            ind_u = np.logical_and(u >= self.sample[0], u <= self.sample[1])
            ind_v = np.logical_and(v >= self.sample[2], v <= self.sample[3])
            ind_sample = np.where(np.logical_and(ind_u, ind_v))[0]
            u = np.take(u, ind_sample, axis=0)
            v = np.take(v, ind_sample, axis=0)
            dx = np.take(dx, ind_sample, axis=0)
            dy = np.take(dy, ind_sample, axis=0)
            self.E = np.take(self.E, ind_sample)
        self.X = np.vstack((u, v)).T
        self.Y = np.vstack((dx, dy)).T
        
    def split_data(self):
        """Separate data into training and testing sets with sklearn."""
        if self.verbose: print("Splitting data into training and testing sets...")
        self.Xtrain, self.Xtest, self.Ytrain, self.Ytest, self.Etrain, self.Etest = \
            train_test_split(self.X, self.Y, self.E, test_size=self.test_size, random_state=self.random_state)
        
        self.nTrain = self.Xtrain.shape[0]
        self.nTest = self.Xtest.shape[0]
    
    def gen_White_Covariance(self):
        """Generate white noise covariance matrix."""
        if self.verbose: print("Generating white noise covariance function...")
        self.W = np.diag(self.Etrain) + self.eps * np.eye(self.nTrain)
        self.Wss = np.diag(self.Etest) + self.eps * np.eye(self.nTest)

    def __init__(self, datafile, nExposure, sample=None, verbose=False, eps=1.49e-8, test_size=0.20, random_state=None):
        self.datafile = datafile        
        self.nExposure = nExposure
        self.sample = sample
        self.verbose = verbose
        self.eps = eps
        self.test_size = test_size
        self.random_state = random_state
        
        self.extract_exposure()
        self.extract_data()
        self.split_data()
        self.gen_White_Covariance()
        
    def gen_kernel(self):
        self.kernel = psd_kernels.ExponentiatedQuadratic(
            amplitude=np.float64(450),
            length_scale=np.float64((5 * u.arcmin).to(u.deg).value),
            feature_ndims=1)

    def fit(self):
        self.model_u = tfd.GaussianProcessRegressionModel(
            kernel=self.kernel,
            index_points=self.Xtest,
            observation_index_points=self.Xtrain,
            observations=self.Ytrain[:, 0],
            observation_noise_variance=self.Etrain.mean())
        
        self.model_v = tfd.GaussianProcessRegressionModel(
            kernel=self.kernel,
            index_points=self.Xtest,
            observation_index_points=self.Xtrain,
            observations=self.Ytrain[:, 1],
            observation_noise_variance=self.Etrain.mean())
        
        self.dx = tf.Session().run(self.model_u.mean())
        self.dy = tf.Session().run(self.model_v.mean())
        self.sigma_dx = tf.Session().run(self.model_u.stddev())
        self.sigma_dy = tf.Session().run(self.model_v.stddev())
        
    def check_error(self, sigma=1):
        within_x = np.sum((np.abs(self.Ytest[:, 0] - self.dx) < sigma*self.sigma_dx).astype(int))
        within_y = np.sum((np.abs(self.Ytest[:, 1] - self.dy) < sigma*self.sigma_dy).astype(int))
        print(f"Fraction of test points within {sigma} standard deviation(s) of posterior predictive mean:")
        print(f"dx: {within_x / self.nTest}; dy: {within_y / self.nTest}")
        