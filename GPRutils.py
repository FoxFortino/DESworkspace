import plotGPR

import gbutil

import os

import numpy as np
import astropy.units as u
import astropy.io.fits as fits
import astropy.stats as stats
from sklearn.model_selection import train_test_split


class dataContainer(object):

    def __init__(self, FITSfile, expNum, randomState=0):
        """
        Stores data for GPR analysis.

        Loads astrometry data from fits files and contains a few methods to easily format it in a way that GPR method can deal with.

        Arguments:
            FITSfile -- (str) FITS file to the grab data from
            expNum -- (int) Exposure number from in fits file

        Keyword Arguments
            randomState -- (int) Seed for the various numpy and scipy random 
                operations.
        """

        self.FITSfile = FITSfile
        self.expNum = expNum
        self.randomState = randomState

    def loadFITS(self):
        if self.FITSfile == 'hoid':
            self.datafile = '/media/pedro/Data/austinfortino/austinFull.fits'
        elif self.FITSfile == 'folio2':
            self.datafile = '/data4/paper/fox/DES/austinFull.fits'
        else:
            self.datafile = self.FITSfile

        self.FITS = fits.open(self.datafile)

    def extractData(self, polyOrder=3, hasGaia=True, sample=None):
        """
        Extract exposure information from current self.fits object.
        
        Keyword Arguments:
            polyOrder -- (int) Order of the unweighted polynomial fit in (u, 
                v) that will be femoved from (dx, dy). If `None` no fit is 
                performed.
            hasGaia (bool) Whether or not to only take data points which have 
                Gaia truth positions
            sample (dict): Dictionary denoting the coordinates (u1, u2, v1, 
                v2) of a rectangle in (u, v). Only points in this rectangle 
                will be kept. sample=None means the entire exposure is used. 
                Example dictionary would be `{"u1": -0.1, "u2": 0, "v1": -0.1, 
                "v2": 0}`.
        """
        
        expKey = self.FITS['Residuals'].data['exposure'] == self.expNum
        exposure = self.FITS['Residuals'].data[expKey]

        if polyOrder is not None:
            poly = Poly2d(polyOrder)
            poly.fit(exposure['u'], exposure['v'], exposure['dx'])
            exposure['dx'] -= poly.evaluate(exposure['u'], exposure['v'])
            poly.fit(exposure['u'], exposure['v'], exposure['dy'])
            exposure['dy'] -= poly.evaluate(exposure['u'],exposure['v'])
            
        u = exposure['u']
        v = exposure['v']
        dx = exposure['dx']
        dy = exposure['dy']
        err = exposure['measErr']
        
        if hasGaia:
            ind_hasGaia = np.where(exposure['hasGaia'])[0]
            u = np.take(u, ind_hasGaia)
            v = np.take(v, ind_hasGaia)
            dx = np.take(dx, ind_hasGaia)
            dy = np.take(dy, ind_hasGaia)
            err = np.take(err, ind_hasGaia)
        
        if sample is not None:
            ind_u = np.logical_and(u >= sample['u1'], u <= sample['u2'])
            ind_v = np.logical_and(v >= sample['v1'], v <= sample['v2'])
            ind = np.where(np.logical_and(ind_u, ind_v))[0]
            
            u = np.take(u, ind, axis=0)
            v = np.take(v, ind, axis=0)
            dx = np.take(dx, ind, axis=0)
            dy = np.take(dy, ind, axis=0)
            err = np.take(err, ind)
            
        self.X = np.vstack((u, v)).T
        self.Y = np.vstack((dx, dy)).T
        self.E = np.vstack((err, err)).T

    def sigmaClip(self, nSigma=4):
        """
        Performs sigma clipping in (dx, dy) to nSigma standard deviations.
        
        Parameters:
            nSigma -- (int) Number of standard deviations to sigma clip to.
        """
        
        mask = stats.sigma_clip(self.Y, sigma=nSigma, axis=0).mask
        mask = ~np.logical_or(*mask.T)
            
        self.X = self.X[mask, :]
        self.Y = self.Y[mask, :]
        self.E = self.E[mask, :]
        
    def splitData(self, train_size=0.50, test_size=None):
        """
        Splits the data into training, validation, and testing sets.
        
        Example:
            If train_size=0.60 and test_size=0.50, then the data is partitioned thus:
            
            60% training
            20% validation
            20% testing
        
        Parameters:
            train_size -- (int or float) Numerical (if int) or fractional (if 
                float) size of data set to be allocated for training (as 
                opposed to validation/testing).
            test_size -- (int or float) Numerical (if int) or fractional (if 
                float) size of data set to be allocated for testing (as 
                opposed to validation). If test_size=None, then no validation 
                set will be generated.

        """
        self.train_size = train_size
        self.test_size = test_size

        self.nData = self.X.shape[0]
        
        Xtrain, Xtv, Ytrain, Ytv, Etrain, Etv = train_test_split(
            self.X, self.Y, self.E,
            train_size=train_size,
            random_state=self.randomState
            )
        
        if test_size is not None:
            Xvalid, Xtest, Yvalid, Ytest, Evalid, Etest = train_test_split(
                Xtv, Ytv, Etv,
                test_size=test_size,
                random_state=self.randomState
                )
            self.Xtrain, self.Xvalid, self.Xtest = Xtrain, Xvalid, Xtest
            self.Ytrain, self.Yvalid, self.Ytest = Ytrain, Yvalid, Ytest
            self.Etrain, self.Evalid, self.Etest = Etrain, Evalid, Etest
        
        else:
            self.Xtrain, self.Xtest = Xtrain, Xtv
            self.Ytrain, self.Ytest = Ytrain, Ytv
            self.Etrain, self.Etest = Etrain, Etv

        self.nTrain = self.Xtrain.shape[0]
        self.nTest = self.Xtest.shape[0]

    def sigmaClipSolution(self, nSigma=4):
        pass

    def saveNPZ(self, outDir):

        np.savez(
            os.path.join(outDir, f"{self.expNum}.npz"),
            FITSfile=self.FITSfile,
            expNum=self.expNum,
            randomState=self.randomState,
            train_size=self.train_size, test_size=self.test_size,
            X=self.X, Y=self.Y, E=self.E,
            params=self.params,
            fbar_s=self.fbar_s
            )

        self.quickPlot(outDir)

    def quickPlot(self, outDir=None):
        x = self.Xtest[:, 0]*u.deg
        y = self.Xtest[:, 1]*u.deg
        dx = self.Ytest[:, 0]*u.mas
        dy = self.Ytest[:, 1]*u.mas
        err = self.Etest[:, 0]*u.mas

        x2 = self.Xtest[:, 0]*u.deg
        y2 = self.Xtest[:, 1]*u.deg
        dx2 = self.Ytest[:, 0]*u.mas - self.fbar_s[:, 0]*u.mas
        dy2 = self.Ytest[:, 1]*u.mas - self.fbar_s[:, 1]*u.mas
        err2 = self.Etest[:, 0]*u.mas

        plotGPR.AstrometricResiduals(
            x, y, dx, dy, err,
            x2=x2, y2=y2, dx2=dx2, dy2=dy2, err2=err2,
            savePath=outDir,
            plotShow=False,
            exposure=self.expNum)

        plotGPR.DivCurl(
            x, y, dx, dy, err,
            x2=x2, y2=y2, dx2=dx2, dy2=dy2, err2=err2,
            savePath=outDir,
            plotShow=False,
            exposure=self.expNum)

        plotGPR.Correlation(
            x, y, dx, dy,
            x2=x2, y2=y2, dx2=dx2, dy2=dy2,
            savePath=outDir,
            plotShow=False,
            exposure=self.expNum)

        plotGPR.Correlation2D(
            x, y, dx, dy,
            x2=x2, y2=y2, dx2=dx2, dy2=dy2,
            savePath=outDir,
            plotShow=False,
            exposure=self.expNum)

        
def loadNPZ(file):
    
    data = np.load(file, allow_pickle=True)
    
    FITSfile = data["FITSfile"]
    expNum = data["expNum"]
    randomState = data["randomState"].item()
    
    dataC = dataContainer(FITSfile, expNum, randomState)
    dataC.X, dataC.Y, dataC.E = data["X"], data["Y"], data["E"]

    dataC.splitData(
        train_size=data["train_size"].item(),
        test_size=data["test_size"].item()
        )

    dataC.params = data["params"]
    dataC.fbar_s = data["fbar_s"]
    
    return dataC

def getGrid(X1, X2):
    u1, u2 = X1[:, 0], X2[:, 0]
    v1, v2 = X1[:, 1], X2[:, 1]
    uu1, uu2 = np.meshgrid(u1, u2)
    vv1, vv2 = np.meshgrid(v1, v2)
    
    return uu1 - uu2, vv1 - vv2

def flat(arr):
    """
    Flattens 2d arrays in row-major order.
    
    Useful because the more natural way to process data is to use arrays of shape (N, 2), but for a curl-free kernel a shape (2N,) array is necessary.
    
    Parameters:
        arr -- (ndarray) 2d array to be flattened in row-major order (C order).
        
    Returns:
        arr -- (ndarray) 1d array.
    """
    
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 2
    return arr.flatten(order='C')

def unflat(arr):
    """Unflattens a 1d array in row-major order.
    
    Turns a shape (2N,) array into a shape (N, 2) array.
    
    Parameters:
        arr -- (ndarray) 1d array to be unflattened in row-major order (C 
            order).
        
    Returns:
        arr -- (ndarray) 2d array.
    """
    
    assert isinstance(arr, np.ndarray)
    assert arr.ndim == 1
    return arr.reshape((arr.shape[0] // 2, 2), order='C')

def calcPixelGrid(
    x, y, dx, dy, err,
    minPoints=100,
    pixelsPerBin=500,
    maxErr=50*u.mas
    ):
    """
    Calculates a pixel grid to make a weighted and binned 2d vector diagram.
    
    Return a 2d vector diagram (quiver plot) of weighted mean astrometric residuals in pixelized areas. Takes in positions (x, y) and vectors (dx, dy) to make a quiver plot, but the plot is binned (according to pixelsPerBin) and weighted (according to err).
    
    Arguments:
        x -- (astropy.units.quantity.Quantity) (N,) specifies x position
        y -- (astropy.units.quantity.Quantity) (N,) specifies y position
        dx -- (astropy.units.quantity.Quantity) (N,) specifies dx vector
        dy -- (astropy.units.quantity.Quantity) (N,) specifies dy vector
        err -- (astropy.units.quantity.Quantity) (N,) specifies error for each vector (dx, dy)
    
    Keyword Arguments:
        minPoints -- (int) Minimum number of max plots that will be plotted
        pixelsPerBin -- (int) number of pixels that are represented by one bin
        maxErr -- (astropy.units.quantity.Quantity) (scalar) largest error that a binned vector can have and still be plotted. Avoids cluttering the plot with noisy arrows.
        
    Returns:
        x -- (astropy.units.quantity.Quantity) (N,) binned x position
        y -- (astropy.units.quantity.Quantity) (N,) binned y position
        dx -- (astropy.units.quantity.Quantity) (N,) binned dx position
        dy -- (astropy.units.quantity.Quantity) (N,) binned dy position
        errors -- (tuple) (3,) RMS error of binned x and y, and noise
        cellSize -- (astropy.units.quantity.Quantity) (scalar) helpful for making the arrow scale on a quiver plot
    """

    # Check that all arrays (x, y, dx, dy, err) are astropy quantity objects
    if not np.all([isinstance(arr, u.quantity.Quantity) 
                   for arr in [x, y, dx, dy, err]]):
        raise TypeError("All input arrays must be of type astropy.units.quantity.Quantity.")

    # Check that x, y have the same units
    if x.unit != y.unit:
        raise u.UnitsError(f"x and y arrays must have the same units but are {x.unit} and {y.unit} respectively.")

    # Check that dx, dy, err have the same units
    if not (dx.unit == dy.unit and dx.unit == err.unit):
        raise u.UnitsError(f"dx, dy, and err arrays must have the same units but are {dx.unit}, {dy.unit}, and {err.unit} respectively.")
        
    # Check that all arrays (x, y, dx, dy, err) are of shape (N,)
    if not (np.all([arr.shape == x.shape for arr in [x, y, dx, dy, err]])
            and np.all([arr.ndim == 1 for arr in [x, y, dx, dy, err]])):
        raise TypeError(f"x, y, dx, dy, and err arrays must be 1 dimensional and the same shape but are {x.shape}, {y.shape}, {dx.shape}, {dy.shape}, and {err.shape} respectively.")
    
    # Check that there are enough data points to do this computation
    if x.shape[0] < minPoints:
        raise ValueError(f"There are not enough points to do this calculation. The minimum number of points is {minPoints} and the length of the dataset is {x.shape[0]}.")
        
    # Check that maxErr is the correct type
    if not isinstance(maxErr, u.quantity.Quantity):
        raise TypeError("maxErr must be of type astropy.units.quantity.Quantity.")
    if maxErr.unit != err.unit:
        raise u.UnitsError(f"maxErr has units of {err.unit} but should have the same units as dx, dy and err ({err.unit}).")
        
    x = x.to(u.deg)
    y = y.to(u.deg)
    dx = dx.to(u.mas)
    dy = dy.to(u.mas)
    err = err.to(u.mas)
    
    # Find the min and max values of x and y in order calculate the bin grid.
    min_x, max_x = np.min(x), np.max(x)
    min_y, max_y = np.min(y), np.max(y)
    
    # Calculate the size of each bin in degrees.
    pixelScale = 264.*u.mas # Nominal mas per pixel
    cellSize = (pixelsPerBin * pixelScale).to(u.deg)
    
    # Calculate the size of the bin grid for x and y.
    binGridSize_x = int(np.ceil((max_x - min_x) / cellSize))
    binGridSize_y = int(np.ceil((max_y - min_y) / cellSize))
    
    # Find the bin that each data point belongs to
    ix = np.array(np.floor((x - min_x) / cellSize), dtype=int)
    iy = np.array(np.floor((y - min_y) / cellSize), dtype=int)
    
    # Make sure that every point is in a bin between the first bin (0)
    # and the last bin (binGridSize - 1)
    ix = np.clip(ix, 0, binGridSize_x - 1)
    iy = np.clip(iy, 0, binGridSize_y - 1)
    
    # ???
    index = iy * binGridSize_x + ix
    
    # Converts from standard deviation to 1/variance
    weight = np.where(err > 0., err**-2., 0.)
    
    # ???
    totalBinGridSize = binGridSize_x * binGridSize_y
    sumWeights_x = np.histogram(
        index,
        bins=totalBinGridSize,
        range=(-0.5, totalBinGridSize + 0.5),
        weights=(weight * dx))[0]
    sumWeights_y = np.histogram(
        index,
        bins=totalBinGridSize,
        range=(-0.5, totalBinGridSize + 0.5),
        weights=(weight * dy))[0]
    sumWeights = np.histogram(
        index,
        bins=totalBinGridSize,
        range=(-0.5, totalBinGridSize + 0.5),
        weights=weight)[0]
    
    # Smallest weight that we'd want to plot a point for
    minWeight = (maxErr**-2.).to(u.mas**-2).value
    
    # If a value is below minWeight, then replace it with noData
    sumWeights_x = np.where(
        sumWeights > minWeight,
        sumWeights_x / sumWeights,
        np.nan)
    sumWeights_y = np.where(
        sumWeights > minWeight,
        sumWeights_y / sumWeights,
        np.nan)
    sumWeights = np.where(
        sumWeights > minWeight,
        sumWeights**-1.,
        np.nan)

    # Make an x and y position for each cell to be the center of its cell
    x = np.arange(totalBinGridSize ,dtype=int)
    x = (((x % binGridSize_x) + 0.5) * cellSize) + min_x
    y = np.arange(totalBinGridSize ,dtype=int)
    y = (((y // binGridSize_x) + 0.5) * cellSize) + min_y

    # Finally take the relevant points to plot
    usefulIndices = np.logical_and(sumWeights != 0,
                                   sumWeights < (maxErr**2.).value)
    x = x[usefulIndices]
    y = y[usefulIndices]
    dx = sumWeights_x[usefulIndices]*u.mas
    dy = sumWeights_y[usefulIndices]*u.mas
    
    # Calculate rms and noise and print it.
    RMS_x = np.std(sumWeights_x[sumWeights > 0.])*u.mas
    RMS_y = np.std(sumWeights_y[sumWeights > 0.])*u.mas
    noise = np.sqrt(np.mean(sumWeights[sumWeights > 0.]))*u.mas
    
    return x, y, dx, dy, (RMS_x, RMS_y, noise), cellSize

def calcDivCurl(
    x, y, dx, dy
    ):
    """
    Calculate divergence and curl of the given vector field.
    
    Given vector displacement (dx, dy) defined on identical 2d grids, return arrays giving divergence and curl of the vector field. These will have NaN in pixels without useful info.
    
    Arguments:
        x -- (astropy.units.quantity.Quantity) (N,) specifies weighted and binned x position
        y -- (astropy.units.quantity.Quantity) (N,) specifies weighted and binned y position
        dx -- (astropy.units.quantity.Quantity) (N,) specifies weighted and binned dx vector
        dy -- (astropy.units.quantity.Quantity) (N,) specifies weighted and binned dy vector
    
    Return:
        div -- (np.ndarray) (2dim) curl of the vector field
        curl -- (np.ndarray) (2dim) divergence of the vector field
    """

    # This line has been replaced because sometimes this line happens to be zero and messes everything up. Removing all the points from np.diff(x) that are zero seems to have little to no effect on the resulting plot and helps get rid of this fairly frequent error.
    # 31 January 2020
    # step = np.min(np.abs(np.diff(u)))
    step = np.min(np.abs(np.diff(x)[np.diff(x) != 0]))

    ix = np.array(np.floor((x / step) + 0.1), dtype=int)
    iy = np.array(np.floor((y / step) + 0.1), dtype=int)

    ix -= np.min(ix)
    iy -= np.min(iy)
    
    dx2d = np.zeros((np.max(iy) + 1, np.max(ix) + 1), dtype=float)
    dy2d = np.array(dx2d)
    
    valid = np.zeros_like(dx2d, dtype=bool)
    valid[iy,ix] = True
    use = np.logical_and(valid[1:-1, :-2], valid[1:-1, 2:])
    use = np.logical_and(use, valid[:-2, 1:-1])
    use = np.logical_and(use, valid[2:, 1:-1])
    
    dx2d[iy,ix] = dx
    dy2d[iy,ix] = dy
    
    # XXX These lines may be wrong?
    dxdx = dy2d[2:, 1:-1] - dy2d[:-2, 1:-1]
    dydx = dx2d[2:, 1:-1] - dx2d[:-2, 1:-1]
    dxdy = dy2d[1:-1, 2:] - dy2d[1:-1, :-2]
    dydy = dx2d[1:-1, 2:] - dx2d[1:-1, :-2]

    div = np.where(use, dxdx + dydy, np.nan)
    curl = np.where(use, dydx - dxdy, np.nan)

    vardiv = np.sqrt(gbutil.clippedMean(div[div == div], 5.)[1])
    varcurl = np.sqrt(gbutil.clippedMean(curl[div == div], 5.)[1])
    
    return div, curl, vardiv, varcurl

def calcCorrelation(
    x, y, dx, dy,
    rmin=5*u.arcsec, rmax=1.5*u.deg,
    dlogr=0.05
    ):
    """
    Produce angle-averaged 2-point correlation functions of astrometric error.

    Using bute-force pair counting, calculate the angle-averaged 2-point correlation functions (xi_+, xi_-, xi_z^2, xi_x) for the supplied sample of data. See appendix of Bernstein 2017 for more detailed explanation of calculations.

    Arguments:
        x -- (astropy.units.quantity.Quantity) (N,) specifies x position
        y -- (astropy.units.quantity.Quantity) (N,) specifies y position
        dx -- (astropy.units.quantity.Quantity) (N,) specifies dx vector
        dy -- (astropy.units.quantity.Quantity) (N,) specifies dy vector

    Keyword Arguments:
        rmin -- (astropy.units.quantity.Quantity) (scalar) minimum separation radius bin
        rmax -- (astropy.units.quantity.Quantity) (scalar) maximum separation radius bun
        dlogr -- (float) step size between rmin and rmax

    Return:
        logr -- (np.ndarray) mean log(radius) in each bin (deg)
        xiplus -- (np.ndarray) <vr1 vr2 + vt1 vt2> = <vx1 vx2 + vy1 vy2> (mas2)
        ximinus -- (np.ndarray) <vr1 vr2 - vt1 vt2> (mas2)
        xicross -- (np.ndarray) <vr1 vt2 + vt1 vr2> (mas2)
        xiz2 -- (np.ndarray) <vx1 vx2 - vy1 vy2 + 2 i vx1 vy2> (mas2)
        xiE -- (np.ndarray) E-mode correlation
        xiB -- (np.ndarray) B-mode correlation
    """

    # Check that all (x, y, dx, dy, rmin, rmax) are astropy quantity objects
    if not np.all([isinstance(arr, u.quantity.Quantity) 
                   for arr in [x, y, dx, dy, rmin, rmax]]):
        raise TypeError("All input arrays and rmax and rmin must be of type astropy.units.quantity.Quantity.")

    # Check that x, y have the same units
    if x.unit != y.unit:
        raise u.UnitsError(f"x and y arrays must have the same units but are {x.unit} and {y.unit} respectively.")

    # Check that dx, dy have the same units
    if not (dx.unit == dy.unit):
        raise u.UnitsError(f"dx, and dy arrays must have the same units but are {dx.unit} and {dy.unit} respectively.")
        
    # Check that all arrays (x, y, dx, dy, err) are of shape (N,)
    if not (np.all([arr.shape == x.shape for arr in [x, y, dx, dy]])
            and np.all([arr.ndim == 1 for arr in [x, y, dx, dy]])):
        raise TypeError(f"x, y, dx, and dy arrays must be 1 dimensional and the same shape but are {x.shape}, {y.shape}, {dx.shape}, and {dy.shape} respectively.")

    if not isinstance(dlogr, float):
        raise TypeError(f"dlogr must be a float but is {type(dlogr)}.")

    # Make sure everything is in the correct units and then just take the value. Don't feel like integrating units into the calculation.
    rmin = rmin.to(u.deg).value
    rmax = rmax.to(u.deg).value
    x = x.to(u.deg).value
    y = y.to(u.deg).value
    dx = dx.to(u.mas).value
    dy = dy.to(u.mas).value

    # Get index arrays that make all unique pairs
    i1, i2 = np.triu_indices(len(x))

    # Omit self-pairs
    use = i1!=i2
    i1 = i1[use]
    i2 = i2[use]
    del use
    
    # Make complex separation vector
    dr = 1j * (y[i2] - y[i1])
    dr += x[i2] - x[i1]

    # Make log(radius) vector to bin data
    logdr = np.log(np.absolute(dr))
    logrmin = np.log(rmin)
    nBins = int(np.ceil(np.log(rmax / rmin) / dlogr))
    hrange = (logrmin, logrmin + (nBins * dlogr))
    counts = np.histogram(logdr, bins=nBins, range=hrange)[0]
    logr = np.histogram(logdr, bins=nBins, range=hrange, weights=logdr)[0]
    logr /= counts

    # Calculate xi_+
    vec =  dx + 1j*dy
    vvec = dx[i1] * dx[i2] + dy[i1] * dy[i2]
    xiplus = np.histogram(logdr, bins=nBins, range=hrange, weights=vvec)[0]
    xiplus /= counts

    # Calculate xi_z^2
    vvec = vec[i1] * vec[i2]
    xiz2 = np.histogram(logdr, bins=nBins, range=hrange, weights=vvec)[0]
    xiz2 /= counts

    # Calculate xi_- and xi_x
    tmp = vvec * np.conj(dr)
    vvec = tmp * np.conj(dr)
    dr = dr.real*dr.real + dr.imag*dr.imag
    vvec /= dr
    del dr
    ximinus = np.histogram(logdr, bins=nBins, range=hrange, weights=vvec)[0]
    ximinus /= counts
    xicross = np.imag(ximinus)
    ximinus = np.real(ximinus)

    # Calculate xi_E and xi_B
    dlogr = np.zeros_like(logr)
    dlogr[1:-1] = 0.5 * (logr[2:] - logr[:-2])
    tmp = np.array(ximinus) * dlogr
    integral = np.cumsum(tmp[::-1])[::-1]
    xiB = 0.5 * (xiplus - ximinus) + integral
    xiE = xiplus - xiB

    return logr, xiplus, ximinus, xicross, xiz2, xiE, xiB

def calcCorrelation2D(
    x, y, dx, dy,
    rmax=1*u.deg,
    nBins=250
    ):
    """
    Produce 2d 2-point correlation functions of astrometric error.

    Produce 2d 2-point correlation function of total displacement power for the supplied sample of data. Uses brute-force pair counting.

    Arguments:
        x -- (astropy.units.quantity.Quantity) (N,) specifies x position
        y -- (astropy.units.quantity.Quantity) (N,) specifies y position
        dx -- (astropy.units.quantity.Quantity) (N,) specifies dx vector
        dy -- (astropy.units.quantity.Quantity) (N,) specifies dy vector

    Keyword Arguments:
        rmax -- (astropy.units.quantity.Quantity) (scalar) maximum separation radius bin. Will calculate from -rmax to rmax separation.
        nBins -- (int) final 2d array will be of shape (nBins, nBins)

    Return:
        xiplus -- (np.ndarray) (2dim) 2pt correlation function (mas2). Is symmetric about the origin. Calculated like so: xi_+ - <vr1 vr2 + vt1 vt2> = <vx1 vx2 + vy1 vy2> 
        counts -- (np.ndarray) (1dim) number of pairs in each separation bin
    """
    
    # Check that all (x, y, dx, dy, rmin, rmax) are astropy quantity objects
    if not np.all([isinstance(arr, u.quantity.Quantity) 
                   for arr in [x, y, dx, dy, rmax]]):
        raise TypeError("All input arrays and rmax must be of type astropy.units.quantity.Quantity.")

    # Check that x, y have the same units
    if x.unit != y.unit:
        raise u.UnitsError(f"x and y arrays must have the same units but are {x.unit} and {y.unit} respectively.")

    # Check that dx, dy have the same units
    if not (dx.unit == dy.unit):
        raise u.UnitsError(f"dx, and dy arrays must have the same units but are {dx.unit} and {dy.unit} respectively.")
        
    # Check that all arrays (x, y, dx, dy) are of shape (N,)
    if not (np.all([arr.shape == x.shape for arr in [x, y, dx, dy]])
            and np.all([arr.ndim == 1 for arr in [x, y, dx, dy]])):
        raise TypeError(f"x, y, dx, and dy arrays must be 1 dimensional and the same shape but are {x.shape}, {y.shape}, {dx.shape}, and {dy.shape} respectively.")

    if not isinstance(nBins, int):
        raise TypeError(f"nBins must be int but is {type(nBins)}.")

    # Make sure everything is in the correct units and then just take the value. Don't feel like integrating units into the calculation.
    rmax = rmax.to(u.deg).value
    x = x.to(u.deg).value
    y = y.to(u.deg).value
    dx = dx.to(u.mas).value
    dy = dy.to(u.mas).value

    hrange = [[-rmax, rmax], [-rmax, rmax]]

    # Get index arrays that make all unique pairs
    i1, i2 = np.triu_indices(len(x))
    
    # Omit self-pairs
    use = i1 != i2
    i1 = i1[use]
    i2 = i2[use]
    del use
    
    # Make separation vectors and count pairs
    yshift = y[i2] - y[i1]
    xshift = x[i2] - x[i1]
    counts = np.histogram2d(xshift, yshift, bins=nBins, range=hrange)[0]

    # Accumulate displacement sums
    vec =  dx + 1j*dy
    vvec = dx[i1] * dx[i2] + dy[i1] * dy[i2]
    xiplus = np.histogram2d(
        xshift,
        yshift,
        bins=nBins,
        range=hrange,
        weights=vvec)[0]
    xiplus /= counts

    # Combine pars
    xiplus = 0.5 * (xiplus + xiplus[::-1, ::-1])
    
    return xiplus, counts

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

