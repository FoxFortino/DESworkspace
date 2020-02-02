import os

import numpy as np
import astropy.units as u
import astropy.constants as c
import matplotlib.pyplot as plt

def astrometricError(
        x, y, dx, dy, err,
        x2=None, y2=None, dx2=None, dy2=None, err2=None,
        title1="Observed", title2="GPR Applied",
        minPoints=100,
        pixelsPerBin=500,
        maxErr=50*u.mas,
        scale=350*u.mas,
        savedir=None
    ):
    """
    Plots the error on each star as a function of sky position.
    
    Makes a 2d vector diagram (quiver plot) where the astrometric errors/residuals for each measurement is plotted as a function of astrometric position. The input data is binned (according to pixelsPerBin) and weighted (according to err).
    
    Arguments:
        x -- (astropy.units.quantity.Quantity) (N,) specifies x position
        y -- (astropy.units.quantity.Quantity) (N,) specifies y position
        dx -- (astropy.units.quantity.Quantity) (N,) specifies dx vector
        dy -- (astropy.units.quantity.Quantity) (N,) specifies dy vector
        err -- (astropy.units.quantity.Quantity) (N,) specifies error for
each vector (dx, dy)
        
    Keyword Arguments:
        x2 -- (astropy.units.quantity.Quantity) (N,) specifies x position
        y2 -- (astropy.units.quantity.Quantity) (N,) specifies y position
        dx2 -- (astropy.units.quantity.Quantity) (N,) specifies dx vector
        dy2 -- (astropy.units.quantity.Quantity) (N,) specifies dy vector
        err2 -- (astropy.units.quantity.Quantity) (N,) specifies error for
each vector (dx, dy)
        title1 -- (str) Title of the first plot (x, y, etc.)
        title2 -- (str) TItle of the secnod plot (x2, y2, etc.)
        minPoints -- (int) Minimum number of max plots that will be plotted
        pixelsPerBin -- (int) number of pixels that are represented by one bin
        maxErr -- (astropy.units.quantity.Quantity) (scalar) largest error that a binned vector can have and still be plotted. Avoids cluttering the plot with noisy arrows.
        scale -- (astrpoy.units.quantity.Quantity) (scalar) specifies the scale parameter for the quiver plot
        savedir -- (str) specifies the path to save the plot pdf to
    """
    
    # Check that scale is an astropy quantity object
    if not isinstance(scale, u.quantity.Quantity):
        raise u.TypeError(f"scale must be of type astropy.quantity.Quantity but is {type(scale)}.")
    
    # Grab binned and weighted 2d vector diagram
    x, y, dx, dy, errors, cellSize = \
        calcPixelGrid(
            x, y, dx, dy, err,
            minPoints=minPoints,
            pixelsPerBin=pixelsPerBin,
            maxErr=maxErr)
    RMS_x = f"RMS x: {np.round(errors[0].value, 1)} {errors[0].unit}"
    RMS_y = f"RMS y: {np.round(errors[1].value, 1)} {errors[1].unit}"
    noise = f"Noise: {np.round(errors[2].value, 1)} {errors[2].unit}"
    
    if np.all([arr is not None for arr in [x2, y2, dx2, dy2, err2]]):
        x2, y2, dx2, dy2, errors2, cellSize2 = \
            calcPixelGrid(
                x2, y2, dx2, dy2, err2,
                minPoints=minPoints,
                pixelsPerBin=pixelsPerBin,
                maxErr=maxErr)
        RMS_x2 = f"RMS x: {np.round(errors2[0].value, 1)} {errors2[0].unit}"
        RMS_y2 = f"RMS y: {np.round(errors2[1].value, 1)} {errors2[1].unit}"
        noise2 = f"Noise: {np.round(errors2[2].value, 1)} {errors2[2].unit}"
        
        fig, axes = plt.subplots(
            nrows=1, ncols=2,
            sharex=True, sharey=True,
            figsize=(16, 8))
        fig.subplots_adjust(wspace=0)
        
        quiver = axes[0].quiver(
            x.to(u.deg).value,
            y.to(u.deg).value,
            dx.to(u.deg).value,
            dy.to(u.deg).value,
            pivot='middle',
            color='green',
            angles='xy',
            scale_units='xy',
            scale=scale.to(u.deg).value,
            units='x')
        axes[0].text(-1.2, 1.4, RMS_x)
        axes[0].text(-1.2, 1.3, RMS_y)
        axes[0].text(-1.2, 1.2, noise)
        axes[0].set_xlabel("Sky Position (deg)", fontsize=14)
        axes[0].set_ylabel("Sky Position (deg)", fontsize=14)
        axes[0].set_aspect("equal")
        axes[0].grid()
        axes[0].set_title(title1)
        axes[0].quiverkey(
            quiver,
            0.875,
            0.85,
            (25*u.mas).to(u.deg).value,
            "25 mas",
            coordinates='data',
            color='red',
            labelpos='N',
            labelcolor='red')

        quiver = axes[1].quiver(
            x2.to(u.deg).value,
            y2.to(u.deg).value,
            dx2.to(u.deg).value,
            dy2.to(u.deg).value,
            pivot='middle',
            color='green',
            angles='xy',
            scale_units='xy',
            scale=scale.to(u.deg).value,
            units='x')
        axes[1].text(0.6, 1.4, RMS_x2)
        axes[1].text(0.6, 1.3, RMS_y2)
        axes[1].text(0.6, 1.2, noise2)
        axes[1].set_xlabel("Sky Position (deg)", fontsize=14)
        axes[1].set_aspect("equal")
        axes[1].grid()
        axes[1].set_title(title2)
        axes[1].quiverkey(
            quiver,
            0.875,
            0.85,
            (25*u.mas).to(u.deg).value,
            "25 mas",
            coordinates='data',
            color='red',
            labelpos='N',
            labelcolor='red')
        
        fig.suptitle("Weighted Astrometric Residuals", fontsize=20)

    else:
        # Make the quiver plot
        plt.figure(figsize=(8, 8))
        quiver = plt.quiver(
            x.to(u.deg).value,
            y.to(u.deg).value,
            dx.to(u.deg).value,
            dy.to(u.deg).value,
            pivot='middle',
            color='green',
            angles='xy',
            scale_units='xy',
            scale=scale.to(u.deg).value,
            units='x')

        # Make the quiver key and label
        quiverkey = plt.quiverkey(
            quiver,
            0.875,
            0.85,
            (25*u.mas).to(u.deg).value,
            "25 mas",
            coordinates='data',
            color='red',
            labelpos='N',
            labelcolor='red')
        
        plt.title("Weighted Astrometric Residuals", fontsize=14)
        plt.xlabel("Sky Position (deg)", fontsize=14)
        plt.ylabel("Sky Position (deg)", fontsize=14)
        
        plt.gca().set_aspect('equal')
        plt.grid()

    xyBuffer = 0.05*u.deg
    plt.xlim((np.min(x) - xyBuffer).value, (np.max(x) + xyBuffer).value)
    plt.ylim((np.min(y) - xyBuffer).value, (np.max(y) + xyBuffer).value)
    
    if savedir is not None:
        plt.savefig(os.path.join(savedir, "astroRes.pdf"))

    plt.show()
    
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
        err -- (astropy.units.quantity.Quantity) (N,) specifies error for
each vector (dx, dy)
    
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
    
    def plot_div_curl(self, X, Y, E, binpix=1024, save=False, ext=''):
        """
        Plot div and curl of the specified vector field, assumed to be samples from a grid.
        
        Parameters:
            X (ndarray): (N, 2) array of (u, v) astrometric positions.
            Y (ndarray): (N, 2) array of (dx, dy) astrometric residuals.
            E (ndarray): (N,) array of measurement errors.
        """
        
        u, v, dx, dy, arrowScale = residInPixels(X[:, 0], X[:, 1], Y[:, 0], Y[:, 1], E[:, 0], binpix=binpix)

        # This line has been replaced because sometimes this step happens to be zero and messes it all up
        # Removing all the points from np.diff(u) that are zero seems to have little to no effect on the
        # resulting plot and helps get rid of this fairly common error.
        # 31 January 2020
        # step = np.min(np.abs(np.diff(u)))
        step = np.min(np.abs(np.diff(u)[np.diff(u) != 0]))
        
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
        
        scale = 50
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
        divplot = axes[0].imshow(div, origin='lower', cmap='Spectral', vmin=-scale, vmax=scale)
        axes[0].axis('off')
        axes[0].set_title('Divergence')

        curlplot = axes[1].imshow(curl, origin='lower', cmap='Spectral', vmin=-scale, vmax=scale)
        axes[1].axis('off')
        axes[1].set_title('Curl')
        
        fig.colorbar(divplot, ax=fig.get_axes())
        
        if save:
            plt.savefig(os.path.join(self.outdir, ext + "div_curl.pdf"))
        
        plt.show()
        
        # Some stats:
        vardiv = gbutil.clippedMean(div[div==div],5.)[1]
        varcurl = gbutil.clippedMean(curl[div==div],5.)[1]
        print("RMS of div: {:.2f}; curl: {:.2f}".format(np.sqrt(vardiv), np.sqrt(varcurl)))
        
    def plot_Emode_2ptcorr(self, X, Y, Y2=None, Bmode=True, rrange=(5./3600., 1.5), nbins=100, nAvg=30, plot_avg_line=True, save=False):
        """
        Use treecorr to produce angle-averaged 2-point correlation functions of astrometric error for the supplied sample of data, using brute-force pair counting.
        
        Returns:
            logr (ndarray): mean log of radius in each bin
            xi_+ (ndarray): <vr1 vr2 + vt1 vt2> = <vx1 vx2 + vy1 vy2>
            xi_- (ndarray): <vr1 vr2 - vt1 vt2>
        """
        
        # Initialize the pyplot figure
        plt.figure(figsize=(8, 8))
        plt.title("E Mode Correlation")
        plt.xlabel('Separation (degrees)')
        plt.ylabel('xi (mas^2)')
        
        # Solve for weighted and pixelized residuals, as well as angle averaged correlation function.
        u, v, dx, dy = X[:, 0], X[:, 1], Y[:, 0], Y[:, 1]
        logr, xiplus, ximinus, xicross, xiz2 = vcorr(u, v, dx, dy)
        
        # Calculate xiE and xiB
        dlogr = np.zeros_like(logr)
        dlogr[1:-1] = 0.5 * (logr[2:] - logr[:-2])
        tmp = np.array(ximinus) * dlogr
        integral = np.cumsum(tmp[::-1])[::-1]
        self.xiB = 0.5 * (xiplus - ximinus) + integral
        self.xiE = xiplus - self.xiB
        
        # Plot xiE
        plt.semilogx(np.exp(logr), self.xiE, 'r.', label="E Mode (Observed)")
        print(f"Mean of first {nAvg} points (Emode (Observed)): ", np.nanmean(self.xiE[:nAvg]))

        if Bmode:
            # Plot xiB
            plt.title("E and B Mode Correlation")
            plt.semilogx(np.exp(logr), self.xiB, 'b.', label="B Mode (Observed)")
            plt.legend(framealpha=0.3)
            print(f"Mean of first {nAvg} points (Bmode (Observed)): ", np.nanmean(self.xiB[:nAvg]))
        
        
        if Y2 is not None:
            
            # Solve for weighted and pixelized residuals, as well as angle averaged correlation function.
            u, v, dx, dy = X[:, 0], X[:, 1], Y2[:, 0], Y2[:, 1]
            logr2, xiplus2, ximinus2, xicross2, xiz22 = vcorr(u, v, dx, dy)
            
            # Calculate xiE and xiB
            dlogr2 = np.zeros_like(logr2)
            dlogr2[1:-1] = 0.5 * (logr2[2:] - logr2[:-2])
            tmp2 = np.array(ximinus2) * dlogr2
            integral2 = np.cumsum(tmp2[::-1])[::-1]
            self.xiB2 = 0.5 * (xiplus2 - ximinus2) + integral2
            self.xiE2 = xiplus2 - self.xiB2
            
            # Plot xiE
            plt.semilogx(np.exp(logr2), self.xiE2, 'rx', label="E Mode (GPR Applied)")    
            print(f"Mean of first {nAvg} points (Emode (GPR Applied)): ", np.nanmean(self.xiE2[:nAvg]))
            
            if Bmode:
                # Plot xiB
                plt.semilogx(np.exp(logr2), self.xiB2, 'bx', label='B Mode (GPR Applied)')
                print(f"Mean of first {nAvg} points (Bmode (GPR Applied)): ", np.nanmean(self.xiB2[:nAvg]))
            
            
            plt.legend(framealpha=0.3)
            
            print(f"Ratio of E modes: {np.nanmean(self.xiE[:nAvg]) / np.nanmean(self.xiE2[:nAvg])}")
            print(f"Ratio of B modes: {np.nanmean(self.xiB[:nAvg]) / np.nanmean(self.xiB2[:nAvg])}")

            
        if plot_avg_line:
            plt.axvline(x=np.exp(logr)[nAvg], color='k', linestyle='--')
        
        # Show plots
        plt.grid()
        
        if save:
            plt.savefig(os.path.join(self.outdir, "Emode.pdf"))
        
        plt.show()
        
    def plot_xiplus_2d(self, X, Y, Y2=None, rmax=0.3, bins=75, vmin=-100, vmax=450, save=False):

        ncols = 1
        xiplus, counts = vcorr2d(X[:, 0], X[:, 1], Y[:, 0], Y[:, 1], rmax=rmax, bins=bins)
        if Y2 is not None:
            ncols += 2
            xiplus2, counts = vcorr2d(X[:, 0], X[:, 1], Y2[:, 0], Y2[:, 1], rmax=rmax, bins=bins)
            xiplus3, counts = vcorr2d(X[:, 0], X[:, 1], Y[:, 0] - Y2[:, 0], Y[:, 1] - Y2[:, 1], rmax=rmax, bins=bins)

        if vmin is None:
            vmin = xiplus.min()
        if vmax is None:
            vmax = xiplus.max()

        fig, axes = plt.subplots(nrows=1, ncols=ncols, sharex=True, sharey=True, figsize=(8*ncols, 8*ncols))

        if Y2 is None:
            axes.set_title("2D Correlation")
            axes.set_xlabel("Bins")
            axes.set_ylabel("Bins")
            im = axes.imshow(xiplus, origin='lower', cmap='Spectral', interpolation='nearest', vmin=vmin, vmax=vmax)
            cbar = fig.colorbar(im)

        else:
            fig.subplots_adjust(wspace=0)

            axes[0].set_title("Observed")
            axes[0].set_xlabel("Bins")
            axes[0].set_ylabel("Bins")
            im = axes[0].imshow(xiplus, origin='lower', cmap='Spectral', interpolation='nearest', vmin=vmin, vmax=vmax)

            axes[1].set_title("GPR")
            axes[1].set_xlabel("Bins")
            im = axes[1].imshow(xiplus2, origin='lower', cmap='Spectral', interpolation='nearest', vmin=vmin, vmax=vmax)

            axes[2].set_title("GPR Applied")
            axes[2].set_xlabel("Bins")
            im = axes[2].imshow(xiplus3, origin='lower', cmap='Spectral', interpolation='nearest', vmin=vmin, vmax=vmax)

            cb_ax = fig.add_axes([0.92, 0.375, 0.025, 0.25])
            cbar = fig.colorbar(im, cax=cb_ax)

        cbar.set_label('Correlation (mas^2)', rotation=270)
        
        if save:
            plt.savefig(os.path.join(self.outdir, f"{save}xiplus_2d.pdf"))
            
        plt.show()

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

#     print("Length ",len(u))
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
#     print(type(vvec),type(dr)) ###
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

#     print("Length ",len(u))
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

    xiplus = 0.5*(xiplus + xiplus[::-1,::-1])  # Combine pairs
    return xiplus, counts

def calcDivCurl(x, y, dx, dy):
    """
    Calculate divergence and curl of the given vector field.
    
    Given vector displacement (dx, dy) defined on identical 2d grids, return arrays giving divergence and curl of the vector field. These will have NaN in pixels without useful info.
    
    Arguments:
        x --
        y --
        dx --
        dy --
    
    Return:
        div --
        curl --
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
    
    dxdx = dx2d[2:, 1:-1] - dx2d[:-2, 1:-1]
    dydy = dy2d[1:-1, 2:] - dy2d[1:-1, :-2]
    
    dydx = dy2d[2:, 1:-1] - dx2d[:-2, 1:-1]
    dxdy = dy2d[1:-1, 2:] - dx2d[1:-1, :-2]

    div = np.where(use, dxdx + dydy, np.nan)
    curl = np.where(use, dydx - dxdy, np.nan)
    
    return div, curl
