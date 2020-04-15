# Standard modules
import os
import shutil

# Willow Fox Fortino's modules
import vK2KGPR
import plotGPR

# Professor Gary Bernstein's modules
import getGaiaDR2 as gaia
import gbutil

# Science modules
import numpy as np
import astropy.units as u
import astropy.constants as c
import astropy.table as tb
import astropy.coordinates as co
import astropy.io.fits as fits
import astropy.stats as stats
from astropy.time import Time
from scipy.spatial.ckdtree import cKDTree
from sklearn.model_selection import train_test_split


class dataContainer(object):

    def __init__(self, randomState=0):
        """
        Stores data for GPR analysis.

        Loads astrometry data from fits files and contains a few methods to
        easily format it in a way that GPR methods can deal with.

        Keyword Arguments
            randomState -- (int) Seed for the various numpy and scipy random 
                operations.
        """

        self.randomState = randomState

    def load(
        self,
        expNum=None,
        zoneDir="/data3/garyb/tno/y6/zone134",
        tile0="DES2203-4623_final.fits",
        earthRef="/home/fortino/y6a1.exposures.positions.fits.gz",
        tileRef="/home/fortino/expnum_tile.fits.gz",
        tol=0.5*u.arcsec
        ):

        # Load in data from a reference tile (tile0). This tile is arbitrary.
        # At the time of implementation, I do not have access to a reference
        # file relating tiles to zones, and exposures to tiles. I only have a
        # reference file (tileRef) that relates exposures to tiles. Therefore,
        # this block of code opens a tile (tile0) that is actually in our
        # (arbitrary) zone of interest for this thesis (zone 134). From this
        # tile I can pick one of its constituent exposures. This way I have
        # chosen an exposure that I know is in zone 134 (the zone I have
        # access to).
        file0 = os.path.join(zoneDir, tile0)
        tab0 = tb.Table.read(file0)
        if expNum is None:
            expNum = np.unique(tab0["EXPNUM"])[10]

        #--------------------#

        # Use earthRef to find the center (ra, dec) of the exposure as well as
        # the MJD of the exposure.
        pos_tab = tb.Table.read(earthRef, hdu=1)
        pos_tab = pos_tab[pos_tab["expnum"] == expNum]
        ra0 = pos_tab["ra"][0]
        dec0 = pos_tab["dec"][0]

        #--------------------#

        # Use tileRef to find all of the tiles that our exposure is a part of.
        tiles_tab = tb.Table.read(tileRef)
        tiles = tiles_tab[tiles_tab["EXPNUM"] == expNum]["TILENAME"]

        #--------------------#

        # Create an empty astropy table with all of the necessary columns.
        DES_tab = tab0.copy()
        DES_tab.remove_rows(np.arange(len(DES_tab)))

        # Loop through each tile, open the table, and append (with tv.vstack)
        # the data to our empty table. Also check if the selected exposure is
        # in the Y band because we do not want to use those.
        for tile in tiles:
            try:
                tile = str(tile) + "_final.fits"
                file = os.path.join(zoneDir, tile)
                tab = tb.Table.read(file)
                tab = tab[tab["EXPNUM"] == expNum]
                band = np.unique(tab["BAND"])[0]
                assert band != "y", "This exposure is in the y band. No."
                DES_tab = tb.vstack([DES_tab, tab])
            except FileNotFoundError:
                print(f"File not found: {file}, continuing without it")
                continue

        print(f"Exposure: {expNum}")
        print(f"Band: {np.unique(DES_tab['BAND'])[0]}")
        print(f"Number of objects: {len(DES_tab)}")

        # Initialize variables for the relevant columns.
        DES_obs = Time(pos_tab["mjd_mid"][0], format="mjd")
        DES_ra = np.array(DES_tab["NEW_RA"])*u.deg
        DES_dec = np.array(DES_tab["NEW_DEC"])*u.deg
        DES_err = np.array(DES_tab["ERRAWIN_WORLD"])*u.deg

        #--------------------#

        # Retrieve Gaia data and initialize variables for the relevant columns.
        GAIA_tab = gaia.getGaiaCat(ra0, dec0, 2.5, 2.5)

        GAIA_obs = Time("J2015.5", format="jyear_str", scale="tcb")
        GAIA_ra = np.array(GAIA_tab["ra"])*u.deg - 360*u.deg
        GAIA_dec = np.array(GAIA_tab["dec"])*u.deg
        GAIA_pmra_cosdec = np.array(GAIA_tab["pmra"])*u.mas/u.yr
        GAIA_pmdec = np.array(GAIA_tab["pmdec"])*u.mas/u.yr
        GAIA_parallax = np.array(GAIA_tab["parallax"])*u.mas
        GAIA_err = np.array(GAIA_tab["error"])*u.deg
        GAIA_cov = np.array(GAIA_tab["cov"])
        GAIA_cov = np.reshape(GAIA_cov, (GAIA_cov.shape[0], 5, 5)) # XXX units?

        #--------------------#

        # Perform an epoch transformation on the Gaia catalog to the DES
        # catalog.
        dt = DES_obs - GAIA_obs
        GAIA_ra += dt * GAIA_pmra_cosdec / np.cos(GAIA_dec)
        GAIA_dec += dt * GAIA_pmdec
        # XXX Parallax transformation not implemented yet

        #--------------------#

        # Initialize astropy SkyCoord objects to take advantage of astropy's
        # `match_coordinates_sky`.
        X_DES = co.SkyCoord(DES_ra, DES_dec)
        X_GAIA = co.SkyCoord(GAIA_ra, GAIA_dec)

        # Match DES objects with Gaia counterparts based on how close together
        # they are on the sky. 
        idx, sep2d, dist3d = co.match_coordinates_sky(X_GAIA, X_DES)

        # slice that can index the Gaia catalog for only the stars that have a
        # match
        self.ind_GAIA = np.where(sep2d < tol)[0]

        # slice that can index the DES catalog for only the stars that have a
        # match. Will be in the same order as ind_GAIA
        self.ind_DES = idx[self.ind_GAIA]

        print(f"There were {self.ind_GAIA.size} matches within {tol}.")

        #--------------------#

        # Perform a gnomonic projection on both the DES and Gaia catalogues.
        X_gn_DES = gnomonicProjection(X_DES, RA0=ra0, dec0=dec0) * u.deg
        X_gn_GAIA = gnomonicProjection(X_GAIA, RA0=ra0, dec0=dec0) * u.deg

        #--------------------#

        self.X = X_gn_DES
        self.Y = X_gn_GAIA[self.ind_GAIA] - X_gn_DES[self.ind_DES]
        self.E_GAIA = GAIA_err[self.ind_GAIA]
        self.E_DES = DES_err

        assert self.X.unit == u.deg
        assert self.Y.unit == u.deg
        assert self.E_GAIA.unit == u.deg
        assert self.E_DES.unit == u.deg

    def splitData(self, nSigma=4, train_size=0.80, subSample=None):
        
        self.mask = stats.sigma_clip(self.Y, sigma=nSigma, axis=0).mask
        self.mask = ~np.logical_or(*self.mask.T)

        X_tv = self.X[self.ind_DES][self.mask]
        Y_tv = self.Y[self.mask]
        E_tv_GAIA = self.E_GAIA[self.mask]
        E_tv_DES = self.E_DES[self.ind_DES][self.mask]
        E_tv = np.sqrt(E_tv_GAIA**2 + E_tv_DES**2)

        # XXX What is the best train size to use?
        split = train_test_split(
            X_tv, Y_tv, E_tv,
            train_size=train_size,
            random_state=self.randomState)
        self.Xtrain, self.Xvalid = split[0], split[1]
        self.Ytrain, self.Yvalid = split[2], split[3]
        self.Etrain, self.Evalid = split[4], split[5]

        self.Xpred = np.delete(self.X, self.ind_DES, axis=0)
        self.Epred = np.delete(self.E_DES, self.ind_DES, axis=0)
        # Should I be using the errors that the GP provides instead of
        # Epred_DES? These errors go into the plotting algorithms.

        if subSample is not None:
            assert subSample < 1 and subSample > 0

            split = train_test_split(
                self.Xtrain, self.Ytrain, self.Etrain,
                train_size=subSample,
                random_state=self.randomState)
            self.Xtrain = split[0]
            self.Ytrain = split[2]
            self.Etrain = split[4]

            split = train_test_split(
                self.Xvalid, self.Yvalid, self.Evalid,
                train_size=subSample,
                random_state=self.randomState)
            self.Xvalid = split[0]
            self.Yvalid = split[2]
            self.Evalid = split[4]

            split = train_test_split(
                self.Xpred, self.Epred,
                train_size=subSample,
                random_state=self.randomState)
            self.Xpred = split[0]
            self.Epred = split[2]

        self.train_size = train_size
        self.subSample = subSample
        
        self.nTrain = self.Xtrain.shape[0]
        self.nValid = self.Xvalid.shape[0]
        self.nPred = self.Xpred.shape[0]
        self.nData = self.nTrain + self.nValid + self.nPred
        # note that self.X.shape[0] will be different from self.nData because
        # of the mask that isn't applied to self.X but is applied to the other
        # arrays.

    def saveNPZ(self, savePath):

        np.savez(
            os.path.join(savePath, f"{self.expNum}.npz"),
            expNum=self.expNum,
            randomState=self.randomState,
            train_size=self.train_size,
            subSample=self.subSample,
            
            X=self.X, Y=self.Y,
            Xtrain=self.Xtrain, Ytrain=self.Ytrain, 
            Xvalid=self.Xvalid, Yvalid=self.Yvalid,
            Xpred=self.Xpred,

            E_GAIA=self.E_GAIA, E_DES=self.E_DES,
            Etrain=self.Etrain,
            Evalid=self.Evalid,
            Epred=self.Epred,

            params=self.params,
            fbar_s=self.fbar_s
            )

        self.quickPlot(plotShow=False, savePath=savePath)

    def quickPlot(self, plotShow=True, savePath=None, sigmaClip=None):


        # XXX Need to come back here and address the fact that I have multiple
        # errors now
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
        
        if sigmaClip is not None:
            mask = stats.sigma_clip(
                np.vstack([dx2.value, dy2.value]).T,
                sigma=sigmaClip, axis=0).mask
            mask = ~np.logical_or(*mask.T)

            x = x[mask]
            y = y[mask]
            dx = dx[mask]
            dy = dy[mask]
            err = err[mask]

            x2 = x2[mask]
            y2 = y2[mask]
            dx2 = dx2[mask]
            dy2 = dy2[mask]
            err2 = err2[mask]

        plotGPR.AstrometricResiduals(
            x, y, dx, dy, err,
            x2=x2, y2=y2, dx2=dx2, dy2=dy2, err2=err2,
            savePath=savePath,
            plotShow=plotShow,
            exposure=self.expNum)

        plotGPR.DivCurl(
            x, y, dx, dy, err,
            x2=x2, y2=y2, dx2=dx2, dy2=dy2, err2=err2,
            savePath=savePath,
            plotShow=plotShow,
            exposure=self.expNum)

        plotGPR.Correlation(
            x, y, dx, dy,
            x2=x2, y2=y2, dx2=dx2, dy2=dy2,
            savePath=savePath,
            plotShow=plotShow,
            exposure=self.expNum)

        plotGPR.Correlation2D(
            x, y, dx, dy,
            x2=x2, y2=y2, dx2=dx2, dy2=dy2,
            savePath=savePath,
            plotShow=plotShow,
            exposure=self.expNum)


def runExposures(expNums, outDir):
    
    for expNum in expNums:
        
        expFile = os.path.join(outDir, str(exp))
        try:
            os.mkdir(expFile)
        except FileExistsError:
            shutil.rmtree(expFile)
            os.mkdir(expFile)
        
        dataC = dataContainer(expNum)
        dataC.sigmaClip()
        dataC.splitData()
        GP = vK2KGPR.vonKarman2KernelGPR(dataC, printing=True, outDir=expFile)
        GP.fitCorr()
        GP.optimize()
        GP.fit(GP.opt_result_GP[0])
        GP.predict(dataC.Xpred)
        dataC.saveNPZ(expFile)
        
def loadNPZ(file):
    
    data = np.load(file, allow_pickle=True)
    
    expNum = data["expNum"].item()
    randomState = data["randomState"].item()
    
    dataC = dataContainer(expNum, randomState)
    dataC.X, dataC.Y, dataC.E = data["X"], data["Y"], data["E"]

    # Need to store Xtrain/Ytrain/Etrain and Xpred/Yvalid/Evalid and Xpred
    # separately.
    # dataC.splitData(
    #     train_size=data["train_size"].item(),
    #     test_size=data["test_size"].item()
    #     )

    dataC.params = data["params"]
    dataC.fbar_s = data["fbar_s"]
    
    return dataC

def getGrid(X1, X2):
    u1, u2 = X1[:, 0], X2[:, 0]
    v1, v2 = X1[:, 1], X2[:, 1]
    uu1, uu2 = np.meshgrid(u1, u2)
    vv1, vv2 = np.meshgrid(v1, v2)
    
    return uu1 - uu2, vv1 - vv2

def gnomonicProjection(X, RA0, dec0, rot=0):
    """
    Perform a gnomonic projection on X.

    X must be an astropy SkyCoord object.
    """
    pole = co.SkyCoord(RA0, dec0, unit='deg', frame='icrs')
    frame = pole.skyoffset_frame(rotation=co.Angle(rot, unit='deg'))

    s = X.transform_to(frame)

    # Get 3 components on unit sphere
    x = np.cos(s.lat.radian)*np.cos(s.lon.radian)
    y = np.cos(s.lat.radian)*np.sin(s.lon.radian)
    z = np.sin(s.lat.radian)
    out_x = y/x * (180. / np.pi)
    out_y = z/x * (180. / np.pi)
    return np.array([out_x, out_y]).T

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

