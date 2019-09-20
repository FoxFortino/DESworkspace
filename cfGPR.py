import gbutil
import treecorr as tc

import os

import numpy as np
import astropy.units as u
import astropy.constants as c
import astropy.stats as stats
import astropy.io.fits as fits
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

class CurlFreeGPR(object):
    """Curl Free Gaussian Process Regressor class."""
    
    def __init__(self, random_state=0):
        """
        Constructor for the Curl Free GPR class.
        
        Parameters:
        random_state (int): Random state variable for the various numpy and scipy random operations.
        """
        
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def load_fits(self, datafile):
        if datafile == 'hoid':
            self.datafile = '/media/data/austinfortino/austinFull.fits'
        elif datafile == 'folio2':
            self.datafile = '/data4/paper/fox/DES/austinFull.fits'
        else:
            self.datafile = datafile

        self.fits = fits.open(self.datafile)
        
    def extract_data(self, nExposure, polyOrder=3, hasGaia=True, sample=None):
        """
        Extract exposure information from current self.fits object.
        
        Parameters:
            nExposure (int): Exposure number from fits file that you want to extract
            polyOrder (int): The order of the unweighted polynomial fit in (u, v) that will be removed from (dx, dy). polyOrder=None means no fit is performed.
            hasGaia (bool): Whether or not to only take data points which has Gaia solutions.
            sample (dict): Dictionary denoting the coordinates (u1, u2, v1, v2) of a rectangle in (u, v). Only points in this rectangle will be kept. sample=None means the entire exposure is used.
            """
        
        self.nExposure = nExposure
        
        self.exposure = self.fits['Residuals'].data[self.fits['Residuals'].data['exposure'] == self.nExposure]
        if polyOrder is not None:
            poly = Poly2d(polyOrder)
            poly.fit(self.exposure['u'], self.exposure['v'], self.exposure['dx'])
            self.exposure['dx'] -= poly.evaluate(self.exposure['u'], self.exposure['v'])
            poly.fit(self.exposure['u'], self.exposure['v'], self.exposure['dy'])
            self.exposure['dy'] -= poly.evaluate(self.exposure['u'], self.exposure['v'])
            
        u = self.exposure['u']
        v = self.exposure['v']
        dx = self.exposure['dx']
        dy = self.exposure['dy']
        E = self.exposure['measErr']
        
        if hasGaia:
            ind_hasGaia = np.where(self.exposure['hasGaia'])[0]
            u = np.take(self.exposure['u'], ind_hasGaia)
            v = np.take(self.exposure['v'], ind_hasGaia)
            dx = np.take(self.exposure['dx'], ind_hasGaia)
            dy = np.take(self.exposure['dy'], ind_hasGaia)
            E = np.take(self.exposure['measErr'], ind_hasGaia)
        
        if sample is not None:
            ind_u = np.logical_and(u >= sample['u1'], u <= sample['u2'])
            ind_v = np.logical_and(v >= sample['v1'], v <= sample['v2'])
            ind = np.where(np.logical_and(ind_u, ind_v))[0]
            
            u = np.take(u, ind, axis=0)
            v = np.take(v, ind, axis=0)
            dx = np.take(dx, ind, axis=0)
            dy = np.take(dy, ind, axis=0)
            E = np.take(E, ind)
            
        self.X = np.vstack((u, v)).T
        self.Y = np.vstack((dx, dy)).T
        self.E = np.vstack((E, E)).T
        
    def gen_synthetic_data(self, nSynth, theta):
        """
        Generates syntheta astrometric positions (u, v) and residuals (dx, dy) based on the curl-free kernel.
        
        Parameters:
            nSynth (int): Number of data points.
            theta (dict): Dictionary of the kernel parameters.
        """

        X = self.rng.uniform(low=-1, high=1, size=(nSynth, 2))
        self.X = X
        
        E = np.abs(self.rng.normal(loc=0, scale=1, size=nSynth))
        self.E = np.vstack((E, E)).T
        
        K = self.curl_free_kernel(theta, self.X, self.X)
        W = self.white_noise_kernel(theta, self.E)
        
        Y = self.rng.multivariate_normal(np.zeros(2 * nSynth), K + W)
        self.Y = self.unflat(Y)
        
    def sigma_clip(self, nSigma=4):
        """
        Performs sigma clipping in (dx, dy) to nSigma standard deviations.
        
        Parameters:
            nSigma (int): Number of standard deviations to sigma clip to.
        """
        
        mask = stats.sigma_clip(self.Y, sigma=nSigma, axis=0).mask
        mask = ~np.logical_or(*mask.T)
            
        self.X = self.X[mask, :]
        self.Y = self.Y[mask, :]
        self.E = self.E[mask, :]
        
    def split_data(self, train_size=0.50, test_size=None):
        """Splits the data into training, validation, and testing sets.
        
        Example:
            If train_size=0.60 and test_size=0.50, then the data is partitioned thus:
            
            60% training
            20% validation
            20% testing
        
        Parameters:
            train_size (int or float): Numerical (if int) or fractional (if float) size of data set to be allocated for training (as opposed to validation/testing).
            test_size (int or float): Numerical (if int) or fractional (if float) size of data set to be allocated for testing (as opposed to validation). If test_size=None, then no validation set will be generated.
        """
        self.nData = self.X.shape[0]
        
        Xtrain, Xtv, Ytrain, Ytv, Etrain, Etv = train_test_split(self.X, self.Y, self.E, train_size=train_size, random_state=self.random_state)
        
        if test_size is not None:
            Xvalid, Xtest, Yvalid, Ytest, Evalid, Etest = train_test_split(Xtv, Ytv, Etv, test_size=test_size, random_state=self.random_state)
            self.Xtrain, self.Xvalid, self.Xtest = Xtrain, Xvalid, Xtest
            self.Ytrain, self.Yvalid, self.Ytest = Ytrain, Yvalid, Ytest
            self.Etrain, self.Evalid, self.Etest = Etrain, Evalid, Etest
        
        else:
            self.Xtrain, self.Xtest = Xtrain, Xtv
            self.Ytrain, self.Ytest = Ytrain, Ytv
            self.Etrain, self.Etest = Etrain, Etv

        self.nTrain = self.Xtrain.shape[0]
        self.nTest = self.Xtest.shape[0]
        
    def curl_free_kernel(self, theta, X1, X2):
        """
        This function generates the full (2N, 2N) covariance matrix.
        
        Parameters:
            theta (dict): Dictionary of the kernel parameters.
            X1 (ndarray): (N, 2) 2d array of astrometric positions (u, v).
            X2 (ndarray): (N, 2) 2d array of astrometric positions (u, v).
            
        Returns:
            K (ndarray): (N, N) 2d array, the kernel.
        """
        
        var_s = theta['var_s']
        sigma_x = theta['sigma_x']
        sigma_y = theta['sigma_y']
        phi = theta['phi']
        
        u1, u2 = X1[:, 0], X2[:, 0]
        v1, v2 = X1[:, 1], X2[:, 1]
        
        uu1, uu2 = np.meshgrid(u1, u2)
        vv1, vv2 = np.meshgrid(v1, v2)
        
        du = uu1 - uu2
        dv = vv1 - vv2
        
        coeff = np.pi * var_s / (4 * sigma_x**5 * sigma_y**5)
        
        Ku_11_1 = -8 * np.cos(phi)**2 * (du * np.cos(phi) - dv * np.sin(phi))**2 * sigma_y**4
        Ku_11_2 = 8 * np.sin(phi)**2 * sigma_x**4 * (-(dv * np.cos(phi) + du * np.sin(phi))**2 + sigma_y**2)
        Ku_11_3 = 8 * np.cos(phi) * sigma_x**2 * sigma_y**2 * (np.sin(phi) * (2 * du * dv * np.cos(2 * phi) + (du - dv) * (du + dv) * np.sin(2 * phi)) + np.cos(phi) * sigma_y**2)
        Ku_11 = Ku_11_1 + Ku_11_2 + Ku_11_3
        
        Ku_12_1 = -4 * (du * np.cos(phi) - dv * np.sin(phi))**2 * np.sin(2 * phi) * sigma_y**4
        Ku_12_2 = 4 * np.sin(2 * phi) * sigma_x**4 * ((dv * np.cos(phi) + du * np.sin(phi))**2 - sigma_y**2)
        Ku_12_3 = 2 * sigma_x**2 * sigma_y**2 * (-4 * du * dv * np.cos(2 * phi)**2 + (-du**2 + dv**2) * np.sin(4 * phi) + 2 * np.sin(2 * phi) * sigma_y**2)
        Ku_12 = Ku_12_1 + Ku_12_2 + Ku_12_3
        
        Ku_22_1 = -8 * np.sin(phi)**2 * (du * np.cos(phi) - dv * np.sin(phi))**2 * sigma_y**4
        Ku_22_2 = 8 * np.cos(phi)**2 * sigma_x**4 * (-(dv * np.cos(phi) + du * np.sin(phi))**2 + sigma_y**2)
        Ku_22_3 = 4 * sigma_x**2 * sigma_y**2 * ((-du**2 + dv**2) * np.sin(2 * phi)**2 - du * dv * np.sin(4 * phi) + 2 * np.sin(phi)**2 * sigma_y**2)
        Ku_22 = Ku_22_1 + Ku_22_2 + Ku_22_3
        
        exp = np.exp(-(1/2) * (((du * np.cos(phi) - dv * np.sin(phi))**2 / sigma_x**2) + ((dv * np.cos(phi) + du * np.sin(phi))**2 / sigma_y**2)))
        
        n1 = X1.shape[0]
        n2 = X2.shape[0]
        K = np.zeros((2*n1, 2*n2), dtype=float)
        
        K[::2, ::2] = (Ku_11 * exp).T
        K[1::2, ::2] = (Ku_12 * exp).T
        K[::2, 1::2] = (Ku_12 * exp).T
        K[1::2, 1::2] = (Ku_22 * exp).T
        
        K = K * coeff
        
        return K

    def white_noise_kernel(self, theta, E):
        """
        This function generates the full (2N, 2N) white kernel covariance matrix.
        
        Parameters:
            theta (dict): Dictionary of the kernel parameters.
            E (ndarray): 2d array of measurement errors (standard deviations)
        """
        return np.diag(self.flat(E)**2)
    
    def fit(self, theta):
        """
        Fits a curl-free GPR to the data contained in self.X, self.Y, and self.E given kernel parameters theta.
        
        Parameters:
            theta (dict): Dictionary of the kernel parameters.
        """
        
        if isinstance(theta, (np.ndarray, list)):
            self.theta = {
                'var_s': theta[0],
                'sigma_x': theta[1],
                'sigma_y': theta[2],
                'phi': theta[3]
            }
        elif isinstance(theta, dict):
            self.theta = theta
        else:
            raise Exception(f"type(theta) is {type(theta)} when it should be dict or list/ndarray.")
        
        self.K = self.curl_free_kernel(self.theta, self.Xtrain, self.Xtrain)
        self.W = self.white_noise_kernel(self.theta, self.Etrain)
        self.L = np.linalg.cholesky(self.K + self.W)
        self.alpha = np.linalg.solve(self.L.T, np.linalg.solve(self.L, self.flat(self.Ytrain)))
        self.nLML = (-1/2) * np.dot(self.flat(self.Ytrain), self.alpha) - np.sum(np.log(np.diag(self.L))) - (self.nTest / 2) * np.log(2 * np.pi)
        
    def predict(self, Xnew=None):     
        """
        Predicts new astrometric residuals (dx, dy) based on the curl-free GPR model and some astrometric positions.
        
        Parameters:
            Xnew (ndarray): Shape (N, 2) array of astrometric positions (u, v).
        """
        
        if Xnew is not None:
            X = Xnew
        else:
            X = self.Xtest
    
        self.Ks = self.curl_free_kernel(self.theta, self.Xtrain, X)
        self.fbar_s = self.unflat(np.dot(self.Ks.T, self.alpha))

        self.Kss = self.curl_free_kernel(self.theta, X, X)
        self.v = np.linalg.solve(self.L, self.Ks)
        self.V_s = self.Kss - np.dot(self.v.T, self.v)
        self.sigma = np.sqrt(np.abs(np.diag(self.V_s)))
        
    def flat(self, arr):
        """
        Flattens 2d arrays in row-major order.
        
        Useful because the more natural way to process data is to use arrays of shape (N, 2), but for a curl-free kernel a shape (2N,) array is necessary.
        
        Parameters:
            arr (ndarray): 2d array to be flattened in row-major order (C order).
            
        Returns:
            arr (ndarray): 1d array.
        """
        
        assert isinstance(arr, np.ndarray)
        assert arr.ndim == 2
        return arr.flatten(order='C')
    
    def unflat(self, arr):
        """Unflattens a 1d array in row-major order.
        
        Turns a shape (2N,) array into a shape (N, 2) array.
        
        Parameters:
            arr (ndarray): 1d array to be unflattened in row-major order (C order).
            
        Returns:
            arr (ndarray): 2d array.
        """
        
        assert isinstance(arr, np.ndarray)
        assert arr.ndim == 1
        return arr.reshape((arr.shape[0] // 2, 2), order='C')
    
    def plot_residuals(self, X, Y, E, binpix=512):
        """
        Plots astrometric residuals as a function of astrometric position on the sky.
        
        Parameters:
            X (ndarray): (N, 2) array of (u, v) astrometric positions.
            Y (ndarray): (N, 2) array of (dx, dy) astrometric residuals.
            E (ndarray): (N,) array of measurement errors.
        """
        
        u, v, dx, dy, arrowScale = residInPixels(X[:, 0], X[:, 1], Y[:, 0], Y[:, 1], E[:, 0], binpix=binpix)

        plt.figure(figsize=(8, 8))
        quiver = plt.quiver(
            u, v, dx, dy,
            pivot='middle',
            color='green',
            angles='xy',
            scale_units='xy',
            scale=arrowScale,
            units='x')

        quiverkey = plt.quiverkey(quiver, 0.5, 0.5, 20, '{:.1f} mas'.format(20), coordinates='axes', color='red', labelpos='N', labelcolor='red')
        
        plt.xlim(np.min(u) - xyBuffer, np.max(u) + xyBuffer)
        plt.ylim(np.min(v) - xyBuffer, np.max(v) + xyBuffer)

        plt.gca().set_aspect('equal')
        plt.grid()
        plt.show()
        
    def plot_div_curl(self, X, Y, E, binpix=1024, scale=50):
        """
        Plot div and curl of the specified vector field, assumed to be samples from a grid.
        
        Parameters:
            X (ndarray): (N, 2) array of (u, v) astrometric positions.
            Y (ndarray): (N, 2) array of (dx, dy) astrometric residuals.
            E (ndarray): (N,) array of measurement errors.
        """
        
        u, v, dx, dy, arrowScale = residInPixels(X[:, 0], X[:, 1], Y[:, 0], Y[:, 1], E[:, 0], binpix=binpix)

        step = np.min(np.abs(np.diff(u)))
        ix = np.array( np.floor(u/step+0.1), dtype=int)
        iy = np.array( np.floor(v/step+0.1), dtype=int)
        ix = ix - np.min(ix)
        iy = iy - np.min(iy)
        dx2d = np.zeros( (np.max(iy)+1,np.max(ix)+1), dtype=float)
        dy2d = np.array(dx2d)
        valid = np.zeros_like(dx2d, dtype=bool)
        valid[iy,ix] = True
        dx2d[iy,ix] = dx
        dy2d[iy,ix] = dy
        div, curl = calcEB(dy2d, dx2d, valid)
        
        
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
        divplot = axes[0].imshow(div, origin='lower', cmap='Spectral', vmin=-scale, vmax=scale)
        axes[0].axis('off')
        axes[0].set_title('Divergence')

        curlplot = axes[1].imshow(curl, origin='lower', cmap='Spectral', vmin=-scale, vmax=scale)
        axes[1].axis('off')
        axes[1].set_title('Curl')
        
        fig.colorbar(divplot, ax=fig.get_axes())
        plt.show()
        
        
        # Some stats:
        vardiv = gbutil.clippedMean(div[div==div],5.)[1]
        varcurl = gbutil.clippedMean(curl[div==div],5.)[1]
        print("RMS of div: {:.2f}; curl: {:.2f}".format(np.sqrt(vardiv), np.sqrt(varcurl)))
        
    def plot_Emode_2ptcorr(self, X, Y, Bmode=False, rrange=(5./3600., 1.5), nbins=100):
        """
        Use treecorr to produce angle-averaged 2-point correlation functions of astrometric error for the supplied sample of data, using brute-force pair counting.
        
        Returns:
            logr (ndarray): mean log of radius in each bin
            xi_+ (ndarray): <vr1 vr2 + vt1 vt2> = <vx1 vx2 + vy1 vy2>
            xi_- (ndarray): <vr1 vr2 - vt1 vt2>
        """

        u, v, dx, dy = X[:, 0], X[:, 1], Y[:, 0], Y[:, 1]
        
#         catx = tc.Catalog(x=u, y=v, k=dx)
#         kk = tc.KKCorrelation(min_sep=rrange[0], max_sep=rrange[1], nbins=nbins)
#         kk.process(catx)
#         xx = np.array(kk.xi)

#         caty = tc.Catalog(x=u, y=v, k=dy)
#         kk = tc.KKCorrelation(min_sep=rrange[0], max_sep=rrange[1], nbins=nbins)
#         kk.process(caty)
#         yy = np.array(kk.xi)
        
#         xiplus = xx + yy
#         ximinus = xx - yy
#         logr = kk.meanlogr

        logr, xiplus, ximinus, xicross, xiz2 = vcorr(u, v, dx, dy)
    
        dlogr = np.zeros_like(logr)
        dlogr[1:-1] = 0.5 * (logr[2:] - logr[:-2])
        tmp = np.array(ximinus) * dlogr
        integral = np.cumsum(tmp[::-1])[::-1]
        self.xiB = 0.5 * (xiplus - ximinus) + integral
        self.xiE = xiplus - self.xiB

        
        plt.figure(figsize=(6, 6))
        plt.title("E Mode Correlation")
        plt.semilogx(np.exp(logr), self.xiE, 'ro', label="E Mode")
        plt.xlabel('Separation (degrees)')
        plt.ylabel('xi (mas^2)')

        if Bmode:
            plt.title("E and B Mode Correlation")
            plt.semilogx(np.exp(logr), self.xiB, 'bo', label="B Mode")
            plt.legend(framealpha=0.3)
        
        plt.grid()
        plt.show()



class Poly2d:
    # Polynomial function of 2 dimensions
    def __init__(self, order, sumOrder=True):
        # Set up order; sumOrder=True if order is max sum of powers of x and y.
        self.order = order
        mask = np.ones((self.order+1, self.order+1), dtype=bool)
        if sumOrder:
            # Clear mask where sum of x & y orders is >order
            for i in range(1, self.order+1):
                mask[i, self.order-i+1:] = False
        self.use = mask
        self.sumOrder = sumOrder
        self.coeffs = None
        return
    
    def evaluate(self, x, y):
        xypow = np.ones((x.shape[0], self.order+1), dtype=float)
        for i in range(1, self.order+1):
            xypow[:, i] = xypow[:, i-1] * x
        tmp = np.dot(xypow, self.coeffs)
        for i in range(1, self.order+1):
            xypow[:, i] = xypow[:, i-1] * y
        return np.sum(tmp * xypow, axis=1)
    
    def fit(self, x, y, z):
        # Least-square fit polynomial coefficients to P(x,y)=z
        # Make cofactor array
        npts = x.shape[0]
        xpow = np.ones((npts, self.order+1), dtype=float)
        for i in range(1, self.order+1):
            xpow[:, i] = xpow[:, i-1] * x
        ypow = np.ones_like(xpow)
        for i in range(1, self.order+1):
            ypow[:, i] = ypow[:, i-1] * y
        A = xpow[:, :, np.newaxis] * ypow[:, np.newaxis, :]
        # Retain powers wanted
        A = A.reshape(npts, (self.order+1) * (self.order+1))[:, self.use.flatten()]
        b = np.linalg.lstsq(A, z, rcond=None)[0]
        self.coeffs = np.zeros((self.order+1, self.order+1), dtype=float)
        self.coeffs[self.use] = b
        return
    
    def getCoeffs(self):
        # Return the current coefficients in a vector.
        return self.coeffs[self.use]
    
    def setCoeffs(self,c):
        # Set the coefficients to the specified vector.
        if len(c.shape) != 1 or c.shape[0] != np.count_nonzero(self.use):
            print("Poly2d.setCoeffs did not get proper-size array", c.shape)
            sys.exit(1)
        self.coeffs[self.use] = c
        return
    
pixScale = 264.   # nominal mas per pixel
xyBuffer = 0.05   # buffer in degrees
    
def residInPixels(u, v, dx, dy, E, binpix=512, scaleFudge=1., maxErr=50):
    '''
    Return a 2d vector diagram of weighted mean astrometric residual in pixelized areas.
    Residuals shown in milliarcsec, and the binned resid arrays are returned.
    binpix is the width of cells (in nominal pixel size)
    scaleFudge is multiplier of arrow length scaling (and key size) 
    to apply to default choice. By default, arrows are scaled so that 
    typical arrow is of length equal to the distance between arrows (cell size).

    maxErr is the largest error that a binned vector can have (mas) and still be plotted,
    to avoid cluttering up the plot with noisy arrows.
    '''
    
    noData = np.nan
    
    if len(u) < 100:
        # Not enough residuals to even try.
        return None
    # use exposure coordinates:
    x = u  ### Problem is these are not relative to array center!!
    y = v
    residx = dx
    residy = dy
    sig = E
    
    cellSize = binpix * pixScale / (1000.*3600.)

    weight = np.where(sig>0, 1./(sig*sig), 0.)
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)
    
    xsize = int(np.ceil( (xmax-xmin) / cellSize))
    ysize = int(np.ceil( (ymax-ymin) / cellSize))

    # Figure out the image pixel for each residual point
    ix = np.array( np.floor( (x-xmin) / cellSize), dtype=int)
    ix = np.clip(ix, 0, xsize-1)
    iy = np.array( np.floor( (y-ymin) / cellSize), dtype=int)
    iy = np.clip(iy, 0, ysize-1)
    index = iy*xsize + ix
    
    sumxw = np.histogram(index, bins=xsize*ysize, range=(-0.5,xsize*ysize+0.5),
                        weights=weight*residx)[0]
    sumyw = np.histogram(index, bins=xsize*ysize, range=(-0.5,xsize*ysize+0.5),
                        weights=weight*residy)[0]
    sumw = np.histogram(index, bins=xsize*ysize, range=(-0.5,xsize*ysize+0.5),
                        weights=weight)[0]
    # Smallest weight that we'd want to plot a point for
    minWeight = maxErr**(-2)
    # Value to put in where weight is below threshold:
    sumxw = np.where( sumw > minWeight, sumxw / sumw, noData)
    sumyw = np.where( sumw > minWeight, sumyw / sumw, noData)
    sumw = np.where( sumw > minWeight, 1./sumw, noData)
    rmsx = np.std(sumxw[sumw>0.])
    rmsy = np.std(sumyw[sumw>0.])
    print('RMSx, RMSy, noise:', rmsx, rmsy, np.sqrt(np.mean(sumw[sumw>0.])))

    # Make an x and y position for each cell to be the center of its cell
    xpos = np.arange(xsize*ysize,dtype=int)
    xpos = ((xpos%xsize)+0.5)*cellSize + xmin
    ypos = np.arange(xsize*ysize,dtype=int)
    ypos = ((ypos//xsize)+0.5)*cellSize + ymin

    useful = np.logical_and(sumw!=0, sumw<(maxErr*maxErr))
    dx = sumxw[useful]
    dy = sumyw[useful]
    xpos = xpos[useful]
    ypos = ypos[useful]

    arrowScale = 20 / cellSize
    arrowScale /= scaleFudge   # Adjust lengths of arrows if desired
    
    return xpos, ypos, dx, dy, arrowScale

def calcEB(dx, dy, valid):
    """
    Given vector displacement (dx,dy) defined on identical 2d grids,
    and boolean 2d array valid which is True where data are valid,
    return arrays giving divergence and curl of the vector field.
    These will have NaN in pixels without useful info.
    """
    dxdx = dx[2:,1:-1] - dx[:-2,1:-1]
    dydx = dy[2:,1:-1] - dy[:-2,1:-1]
    dxdy = dx[1:-1,2:] - dx[1:-1,:-2]
    dydy = dy[1:-1,2:] - dy[1:-1,:-2]
    use = np.logical_and(valid[1:-1,:-2], valid[1:-1,2:])
    use = np.logical_and(use, valid[:-2,1:-1])
    use = np.logical_and(use, valid[2:,1:-1])
    div = np.where(use, dxdx+dydy, np.nan)
    curl= np.where(use, dydx-dxdy, np.nan)
    return div, curl

def vcorr(u, v, dx, dy, rmin=5./3600., rmax=1.5, dlogr=0.05):
    """
    Produce angle-averaged 2-point correlation functions of astrometric error
    for the supplied sample of data, using brute-force pair counting.
    Output are the following functions:
    logr - mean log of radius in each bin
    xi_+ - <vr1 vr2 + vt1 vt2> = <vx1 vx2 + vy1 vy2>
    xi_- - <vr1 vr2 - vt1 vt2>
    xi_x - <vr1 vt2 + vt1 vr2>
    xi_z2 - <vx1 vx2 - vy1 vy2 + 2 i vx1 vy2>
    """

    print("Length ",len(u))
    # Get index arrays that make all unique pairs
    i1, i2 = np.triu_indices(len(u))
    # Omit self-pairs
    use = i1!=i2
    i1 = i1[use]
    i2 = i2[use]
    del use
    
    # Make complex separation vector
    dr = 1j * (v[i2]-v[i1])
    dr += u[i2]-u[i1]

    # log radius vector used to bin data
    logdr = np.log(np.absolute(dr))
    logrmin = np.log(rmin)
    bins = int(np.ceil(np.log(rmax/rmin)/dlogr))
    hrange = (logrmin, logrmin+bins*dlogr)
    counts = np.histogram(logdr, bins=bins, range=hrange)[0]
    logr = np.histogram(logdr, bins=bins, range=hrange, weights=logdr)[0] / counts

    # First accumulate un-rotated stats
    vec =  dx + 1j*dy
    vvec = dx[i1] * dx[i2] + dy[i1] * dy[i2]
    xiplus = np.histogram(logdr, bins=bins, range=hrange, weights=vvec)[0]/counts
    vvec = vec[i1] * vec[i2]
    xiz2 = np.histogram(logdr, bins=bins, range=hrange, weights=vvec)[0]/counts

    # Now rotate into radial / perp components
    print(type(vvec),type(dr)) ###
    tmp = vvec * np.conj(dr)
    vvec = tmp * np.conj(dr)
    dr = dr.real*dr.real + dr.imag*dr.imag
    vvec /= dr
    del dr
    ximinus = np.histogram(logdr, bins=bins, range=hrange, weights=vvec)[0]/counts
    xicross = np.imag(ximinus)
    ximinus = np.real(ximinus)

    return logr, xiplus, ximinus, xicross, xiz2

def vcorr2d(u, v, dx, dy, rmax=1., bins=513):
    """
    Produce 2d 2-point correlation function of total displacement power
    for the supplied sample of data, using brute-force pair counting.
    Output are 2d arrays giving the 2PCF and then the number of pairs that
    went into each bin.  The 2PCF calculated is
    xi_+ - <vr1 vr2 + vt1 vt2> = <vx1 vx2 + vy1 vy2>
    Output function will be symmetric about origin.
    """

    hrange = [ [-rmax,rmax], [-rmax,rmax] ]

    print("Length ",len(u))
    # Get index arrays that make all unique pairs
    i1, i2 = np.triu_indices(len(u))
    # Omit self-pairs
    use = i1 != i2
    i1 = i1[use]
    i2 = i2[use]
    del use
    
    # Make separation vectors and count pairs
    yshift = v[i2] - v[i1]
    xshift = u[i2] - u[i1]
    counts = np.histogram2d(xshift, yshift, bins=bins, range=hrange)[0]

    # Accumulate displacement sums
    vec =  dx + 1j*dy
    vvec = dx[i1] * dx[i2] + dy[i1] * dy[i2]
    xiplus = np.histogram2d(xshift, yshift, bins=bins, range=hrange, weights=vvec)[0]/counts
    
    # XXX Austin's addition here: is this the right way to get xi_-?
    # vvec = dx[i1] * dx[i2] - dy[i1] * dy[i2]
    # xiplus = np.histogram2d(xshift, yshift, bins=bins, range=hrange, weights=vvec)[0]/counts

    xiplus = 0.5*(xiplus + xiplus[::-1,::-1])  # Combine pairs
    return xiplus, counts