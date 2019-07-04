from time import time
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import astropy.units as u
import astropy.constants as c
import astropy.io.fits as pf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

import forAustin as fa

class GPR(object):
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
    
    def gen_White_Covariance(self):
        """Generate white noise covariance matrix."""
        if self.verbose: print("Generating white noise covariance function...")
        self.W = np.diag(self.Etrain) + self.eps * np.eye(self.Etrain.shape[0])
        self.Wss = np.diag(self.Etest) + self.eps * np.eye(self.Etest.shape[0])
    
    def __init__(self, datafile, nExposure, sample=None, verbose=False, eps=1.49e-8, test_size=0.20, random_state=None, tensor=False):
        self.datafile = datafile        
        self.nExposure = nExposure
        self.sample = sample
        self.verbose = verbose
        self.eps = eps
        self.test_size = test_size
        self.random_state = random_state
        self.tensor = tensor
        
        self.extract_exposure()
        self.extract_data()
        self.split_data()
        self.gen_White_Covariance()
    
    def EBF(self, data, theta):
        uu1 = data[0]
        uu2 = data[1]
        vv1 = data[2]
        vv2 = data[3]
        
        var_s = theta[0]
        sigma_x = theta[1]
        sigma_y = theta[2]
        phi = theta[3]
        
        if self.tensor:
            a = tf.cos(phi)**2 / (2 * sigma_x**2) + tf.sin(phi)**2 / (2 * sigma_y**2)
            b = - tf.sin(2 * phi) / (4 * sigma_x**2) + tf.sin(2 * phi) / (4 * sigma_y**2)
            c = tf.sin(phi)**2 / (2 * sigma_x**2) + tf.cos(phi)**2 / (2 * sigma_y**2)
           
            uu = a * (uu1 - uu2)**2
            vv = c * (vv1 - vv2)**2
            uv = 2 * b * (uu1 - uu2)*(vv1 - vv2)
            
            K = var_s * tf.exp(-(uu + vv + uv))
            
            return K

        a = np.cos(phi)**2 / (2 * sigma_x**2) + np.sin(phi)**2 / (2 * sigma_y**2)
        b = - np.sin(2 * phi) / (4 * sigma_x**2) + np.sin(2 * phi) / (4 * sigma_y**2)
        c = np.sin(phi)**2 / (2 * sigma_x**2) + np.cos(phi)**2 / (2 * sigma_y**2)

        uu = a * (uu1 - uu2)**2
        vv = c * (vv1 - vv2)**2
        uv = 2 * b * (uu1 - uu2)*(vv1 - vv2)
        
        K = var_s * np.exp(-(uu + vv + uv))
    
        return K
    
    def gen_coordinate_arrays(self, X, Y):
        u1, u2 = X[:, 0], Y[:, 0]
        v1, v2 = X[:, 1], Y[:, 1]
        
        uu1, uu2 = np.meshgrid(u1, u2)
        vv1, vv2 = np.meshgrid(v1, v2)
        
        return uu1, uu2, vv1, vv2
        
    def gen_EBF_Covariance(self, theta):
        """Generate relevant covariance matrices."""
        if self.verbose: print("Generating elliptical covariance function...")
        data = self.gen_coordinate_arrays(self.Xtrain, self.Xtrain)
        self.K = self.EBF(data, theta)
        data = self.gen_coordinate_arrays(self.Xtest, self.Xtest)
        self.Kss = self.EBF(data, theta)
        data = self.gen_coordinate_arrays(self.Xtest, self.Xtrain)
        self.Ks = self.EBF(data, theta)
        
    def train(self):
        if self.verbose: print("Solving for posterior...")
        t0 = time()
        # The following commented-out code solves for the posterior distribution but
        # does not involve cholesky decomposition (it simply uses np.linalg.inv) and is therefore
        # computationally slower than the following block of code that does use np.linalg.cholesky
        # (and by extension, it also uses np.linalg.solve.).
        # KW_inv = np.linalg.inv(self.K + self.W)
        # self.fbar_s = (self.Ks.T).dot(KW_inv).dot(self.Ytrain)
        # self.V_s = self.Kss - (self.Ks.T).dot(KW_inv).dot(self.Ks)
        # self.sigma = np.sqrt(np.abs(np.diag(self.V_s)))
        
        if self.tensor:
            self.L = tf.linalg.cholesky(self.K + self.W)
            self.alpha = tf.linalg.solve(tf.transpose(self.L), tf.linalg.solve(self.L, self.Ytrain))
            self.fbar_s = tf.tensordot(tf.transpose(self.Ks), self.alpha, axes=1)

            self.v = tf.linalg.solve(self.L, self.Ks)
            self.V_s = self.Kss - tf.tensordot(tf.transpose(self.v), self.v, axes=1)
            self.sigma = tf.math.sqrt(tf.math.abs(tf.linalg.tensor_diag(self.V_s)))

            return
        
        self.L = np.linalg.cholesky(self.K + self.W)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.Ytrain))
        self.fbar_s = np.dot(self.Ks.T, self.alpha)

        self.v = np.linalg.solve(self.L, self.Ks)
        self.V_s = self.Kss - np.dot(self.v.T, self.v)
        self.sigma = np.sqrt(np.abs(np.diag(self.V_s)))
        t1 = time()
        if self.verbose: print(f"Posterior found in {np.round(t1-t0, 3)} seconds.\n")

    def fit(self, theta):
        self.gen_EBF_Covariance(theta)
        self.train()
    
    def summary(self):
        print(f"Current Log Marginal Likelihood: {self.get_LML()}")
        self.get_std()
        self.plot_uv()
        self.plot_residuals()
        self.plot_closeup()
        self.plot_quiver_GP()
        self.plot_quiver_fits()
        self.plot_resres()
        
    def convert2numpy(self):
        """Converts posterior solution from tensorflow tensors to numpy arrays."""
        self.L = tf.Session().run(self.L)
        self.alpha = tf.Session().run(self.alpha)
        self.fbar_s = tf.Session().run(self.fbar_s)
        
        self.v = tf.Session().run(self.v)
        self.V_s = tf.Session().run(self.V_s)
        self.sigma = tf.Session().run(self.sigma)
        
        self.K = tf.Session().run(self.K)
        self.Ks = tf.Session().run(self.Ks)
        self.Kss = tf.Session().run(self.Kss)
    
    def draw_posterior(self, size=1):
        """Draw from the posterior distribution."""
        dx = np.random.multivariate_normal(self.fbar_s[:, 0], self.V_s, size=size).T
        dy = np.random.multivariate_normal(self.fbar_s[:, 1], self.V_s, size=size).T
        return dx, dy
    
    def draw_prior(self, size=1):
        """Draw from the prior distribution."""
        dx = np.random.multivariate_normal(np.zeros(self.Xtest.shape[0]), self.Kss + self.Wss, size=size).T
        dy = np.random.multivariate_normal(np.zeros(self.Xtest.shape[0]), self.Kss + self.Wss, size=size).T
        return dx, dy
    
    def get_LML(self):
        """Calculates the log marginal likelihood based on the current posterior predictive
        mean and the current posterior predictive variance."""
        if self.tensor:
            LML_a = (-1/2) * tf.tensordot(self.Ytrain.T, self.alpha, axes=1)
            LML_b =  - tf.math.reduce_sum(tf.math.log(tf.linalg.tensor_diag(self.L)))
            LML_c =  - (self.Ytest.shape[0] / 2) * np.log(2 * np.pi)
            LML = tf.math.reduce_sum(tf.linalg.diag(LML_a + LML_b + LML_c))
            return LML
        
        LML_a = (-1/2) * np.dot(self.Ytrain.T, self.alpha)
        LML_b = - np.sum(np.log(np.diag(self.L)))
        LML_C = -(self.Ytest.shape[0] / 2) * np.log(2 * np.pi)
        LML = np.sum(np.diag(LML_a + LML_b + LML_c))

        return LML
    
    def get_std(self):
        std0_dx = np.std(self.Ytest[:, 0])
        std0_dy = np.std(self.Ytest[:, 1])
        stdf_dx = np.std(self.fbar_s[:, 0])
        stdf_dy = np.std(self.fbar_s[:, 1])
        improvement_dx = std0_dx / stdf_dx
        improvement_dy = std0_dy / stdf_dy
        print(f"Standard deviation of validation residuals: dx {np.round(std0_dx, 3)}, dy {np.round(std0_dy, 3)}")
        print(f"Standard deviation of Gaussian Process residuals: dx {np.round(stdf_dx, 3)}, dy {np.round(stdf_dy, 3)}")
        print(f"The ratio of std(valid) / std(GP): dx {np.round(improvement_dx, 3)}, dy {np.round(improvement_dy, 3)}")
              
    def plot_uv(self):
        plt.figure(figsize=(8, 8))
        plt.title(f"u, v Positions")
        plt.xlabel("u (deg)")
        plt.ylabel("v (deg)")
        plt.scatter(self.Xtrain[:, 0], self.Xtrain[:, 1], alpha=0.5, label="Training Positions")
        plt.scatter(self.Xtest[:, 0], self.Xtest[:, 1], alpha=0.5, label="Testing Positions")
        plt.legend()
        plt.show()
              
    def plot_residuals(self):
        plt.figure(figsize=(8, 8))
        plt.title(f"dx, dy Residuals")
        plt.xlabel("dx (mas)")
        plt.ylabel("dy (mas)")
        plt.xlim((-100, 100))
        plt.ylim((-100, 100))
        plt.scatter(self.Ytrain[:, 0], self.Ytrain[:, 1], alpha=0.5, label="Training Residuals")
        plt.scatter(self.Ytest[:, 0], self.Ytest[:, 1], alpha=0.5, label="Testing Residuals")
        plt.scatter(self.fbar_s[:, 0], self.fbar_s[:, 1], alpha=0.5, label="Posterior Predictive Mean of Residuals")
        plt.legend()
        plt.show()
              
    def plot_closeup(self):
        plt.figure(figsize=(8, 8))
        plt.title("Closeup of Posterior Predictive Mean of Residuals with 1$\sigma$ Error Bars")
        plt.xlabel("dx (mas)")
        plt.ylabel("dy (mas)")
        # plt.xlim((-100, 100))
        # plt.ylim((-100, 100))
        plt.errorbar(
            self.fbar_s[:, 0],
            self.fbar_s[:, 1],
            xerr=1*self.sigma,
            yerr=1*self.sigma,
            fmt='o',
            color='green',
            alpha=0.5)
        plt.show()
    
    def plot_quiver_GP(self):
        plt.figure(figsize=(8, 8))
        plt.title("Quiver Plot of Error for the Test Set with Errors given by the GP")
        plt.quiver(self.Xtest[:, 0], self.Xtest[:, 1], self.fbar_s[:, 0], self.fbar_s[:, 1])
        plt.show()
              
    def plot_quiver_fits(self):
        plt.figure(figsize=(8, 8))
        plt.title("Quiver Plot of Error for the Test Set with Errors given by the fits file")
        plt.quiver(self.Xtest[:, 0], self.Xtest[:, 1], self.Ytest[:, 0], self.Ytest[:, 1])
        plt.show()
    
    def plot_resres(self):
        plt.figure(figsize=(8, 8))
        plt.title(f"Test Set Residuals (given by fits file) Subtracted by Estimated Residuals (given by GP).")
        plt.xlabel("dx (mas)")
        plt.ylabel("dy (mas)")
        plt.xlim((-100, 100))
        plt.ylim((-100, 100))
        plt.scatter(self.Ytest[:, 0] - self.fbar_s[:, 0], self.Ytest[:, 1] - self.fbar_s[:, 1], alpha=0.5)
        plt.show()