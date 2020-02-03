import os

import gbutil

import numpy as np
import astropy.units as u
import matplotlib.pyplot as plt

def AstrometricError(
    x, y, dx, dy, err,
    x2=None, y2=None, dx2=None, dy2=None, err2=None,
    title1="Observed", title2="GPR Applied",
    minPoints=100,
    pixelsPerBin=500,
    maxErr=50*u.mas,
    scale=350*u.mas,
    arrowScale=50*u.mas,
    saveDir=None
    ):
    """
    Plots the error on each star as a function of sky position.
    
    Makes a 2d vector diagram (quiver plot) where the astrometric errors/residuals for each measurement is plotted as a function of astrometric position. The input data is binned (according to pixelsPerBin) and weighted (according to err).
    
    Arguments:
        x -- (astropy.units.quantity.Quantity) (N,) specifies x position
        y -- (astropy.units.quantity.Quantity) (N,) specifies y position
        dx -- (astropy.units.quantity.Quantity) (N,) specifies dx vector
        dy -- (astropy.units.quantity.Quantity) (N,) specifies dy vector
        err -- (astropy.units.quantity.Quantity) (N,) specifies error for each vector (dx, dy)
        
    Keyword Arguments:
        x2 -- (astropy.units.quantity.Quantity) (N,) specifies x position
        y2 -- (astropy.units.quantity.Quantity) (N,) specifies y position
        dx2 -- (astropy.units.quantity.Quantity) (N,) specifies dx vector
        dy2 -- (astropy.units.quantity.Quantity) (N,) specifies dy vector
        err2 -- (astropy.units.quantity.Quantity) (N,) specifies error for each vector (dx, dy)
        title1 -- (str) Title of the first plot (x, y, etc.)
        title2 -- (str) TItle of the secnod plot (x2, y2, etc.)
        minPoints -- (int) Minimum number of max plots that will be plotted
        pixelsPerBin -- (int) number of pixels that are represented by one bin
        maxErr -- (astropy.units.quantity.Quantity) (scalar) largest error that a binned vector can have and still be plotted. Avoids cluttering the plot with noisy arrows.
        scale -- (astrpoy.units.quantity.Quantity) (scalar) specifies the scale parameter for the quiver plot
        savedir -- (str) specifies the path to save the plot pdf to

    Returns:
        None
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
            arrowScale.to(u.deg).value,
            f"{arrowScale.to(u.mas).value} mas",
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
            arrowScale.to(u.deg).value,
            f"{arrowScale.to(u.mas).value} mas",
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
            arrowScale.to(u.deg).value,
            f"{arrowScale.to(u.mas).value} mas",
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

    if saveDir is not None:
        if np.all([arr is not None for arr in [x2, y2, dx2, dy2, err2]]):
            extension = "compare_"
        else:
            extension = ""
        plt.savefig(os.path.join(saveDir, f"{extension}AstroRes.pdf"))

    plt.show()
    
def DivCurl(
    x, y, dx, dy, err,
    x2=None, y2=None, dx2=None, dy2=None, err2=None,
    title1="Observed", title2="GPR Applied",
    minPoints=100,
    pixelsPerBin=1000,
    maxErr=50*u.mas,
    scale=50,
    saveDir=None
    ):
    """ 
    Make 2d divergence and curl plots for the supplied vector fields.

    Vector field is assumed to be samples from a grid. Plots the divergence and curl of two different sets of data. This is often useful when plotting the observed data and also the data after the GPR has been applied.

    Arguments:
        x -- (astropy.units.quantity.Quantity) (N,) specifies x position
        y -- (astropy.units.quantity.Quantity) (N,) specifies y position
        dx -- (astropy.units.quantity.Quantity) (N,) specifies dx vector
        dy -- (astropy.units.quantity.Quantity) (N,) specifies dy vector
        err -- (astropy.units.quantity.Quantity) (N,) specifies error for each vector (dx, dy)
        
    Keyword Arguments:
        x2 -- (astropy.units.quantity.Quantity) (N,) specifies x position
        y2 -- (astropy.units.quantity.Quantity) (N,) specifies y position
        dx2 -- (astropy.units.quantity.Quantity) (N,) specifies dx vector
        dy2 -- (astropy.units.quantity.Quantity) (N,) specifies dy vector
        err2 -- (astropy.units.quantity.Quantity) (N,) specifies error for each vector (dx, dy)
        title1 -- (str) Title of the first plot (x, y, etc.)
        title2 -- (str) TItle of the secnod plot (x2, y2, etc.)
        minPoints -- (int) Minimum number of max plots that will be plotted
        pixelsPerBin -- (int) number of pixels that are represented by one bin
        maxErr -- (astropy.units.quantity.Quantity) (scalar) largest error that a binned vector can have and still be plotted. Avoids cluttering the plot with noisy points.
        scale -- (int) (scalar) dynamic range (-scale, scale) for the imshow plots
        savedir -- (str) specifies the path to save the plot pdf to

    Returns:
        None
    """

    # Calculate pixel grid
    x, y, dx, dy, errors, cellSize = \
        calcPixelGrid(
            x, y, dx, dy, err,
            minPoints=minPoints,
            pixelsPerBin=pixelsPerBin,
            maxErr=maxErr)
    RMS_x = f"RMS x: {np.round(errors[0].value, 1)} {errors[0].unit}"
    RMS_y = f"RMS y: {np.round(errors[1].value, 1)} {errors[1].unit}"
    noise = f"Noise: {np.round(errors[2].value, 1)} {errors[2].unit}"

    # Calculate div and curl
    div, curl = calcDivCurl(x, y, dx, dy)
    vardiv = np.sqrt(gbutil.clippedMean(div[div == div], 5.)[1])
    varcurl = np.sqrt(gbutil.clippedMean(curl[div == div], 5.)[1])
    
    if np.all([arr is not None for arr in [x2, y2, dx2, dy2, err2]]):
        
        # Calculate second pixel grid
        x2, y2, dx2, dy2, errors2, cellSize2 = \
            calcPixelGrid(
                x2, y2, dx2, dy2, err2,
                minPoints=minPoints,
                pixelsPerBin=pixelsPerBin,
                maxErr=maxErr)
        RMS_x2 = f"RMS x: {np.round(errors2[0].value, 1)} {errors2[0].unit}"
        RMS_y2 = f"RMS y: {np.round(errors2[1].value, 1)} {errors2[1].unit}"
        noise2 = f"Noise: {np.round(errors2[2].value, 1)} {errors2[2].unit}"
        
        # Calculate second div and curl
        div2, curl2 = calcDivCurl(x2, y2, dx2, dy2)
        vardiv2 = np.sqrt(gbutil.clippedMean(div2[div2 == div2], 5.)[1])
        varcurl2 = np.sqrt(gbutil.clippedMean(curl2[div2 == div2], 5.)[1])
        
        # Create plot
        fig, axes = plt.subplots(
            nrows=2, ncols=2,
            sharex=True, sharey=True,
            figsize=(12, 12))
        fig.subplots_adjust(wspace=0.1)
        fig.subplots_adjust(hspace=0)
        
        
        # First divergence plot
        divplot = axes[0, 0].imshow(
            div,
            origin="lower",
            cmap="Spectral",
            vmin=-scale,
            vmax=scale)
        axes[0, 0].set_title("Divergence", fontsize=14)
        axes[0, 0].text(0.2, 22, f"RMS\n{np.round(vardiv, 2)}", fontsize=12)
        axes[0, 0].axis("off")
        
        # First curl plot
        curlplot = axes[0, 1].imshow(
            curl,
            origin="lower",
            cmap="Spectral",
            vmin=-scale,
            vmax=scale)
        axes[0, 1].set_title("Curl", fontsize=14)
        axes[0, 1].text(22, 22, f"RMS\n{np.round(varcurl, 2)}", fontsize=12)
        axes[0, 1].axis("off")
        
        # Second divergence plot
        divplot = axes[1, 0].imshow(
            div2,
            origin="lower",
            cmap="Spectral",
            vmin=-scale,
            vmax=scale)
        axes[1, 0].set_title("Divergence", fontsize=14)
        axes[1, 0].text(0.2, 22, f"RMS\n{np.round(vardiv2, 2)}", fontsize=12)
        axes[1, 0].axis("off")
        
        # Second curl plot
        curlplot = axes[1, 1].imshow(
            curl2,
            origin="lower",
            cmap="Spectral",
            vmin=-scale,
            vmax=scale)
        axes[1, 1].set_title("Curl", fontsize=14)
        axes[1, 1].text(22, 22, f"RMS\n{np.round(varcurl2, 2)}", fontsize=12)
        axes[1, 1].axis("off")
        
        # Titles and colorbar
        axes[0, 0].text(22.5, 27, title1, fontsize=20)
        axes[1, 0].text(22.5, 27, title2, fontsize=20)
        plt.suptitle("Divergence and Curl Fields", fontsize=20)
        fig.colorbar(divplot, ax=fig.get_axes())
        
    else:
        # Create plot
        fig, axes = plt.subplots(
            nrows=1, ncols=2,
            sharex=True, sharey=True,
            figsize=(16, 8))

        # Divergence plot
        divplot = axes[0].imshow(
            div,
            origin="lower",
            cmap="Spectral",
            vmin=-scale,
            vmax=scale)
        axes[0].set_title("Divergence", fontsize=14)
        axes[0].text(0.2, 22, f"RMS\n{np.round(vardiv, 2)}", fontsize=12)
        axes[0].axis("off")
        
        # Curl plot
        curlplot = axes[1].imshow(
            curl,
            origin="lower",
            cmap="Spectral",
            vmin=-scale,
            vmax=scale)
        axes[1].set_title("Curl", fontsize=14)
        axes[1].text(22, 22, f"RMS\n{np.round(varcurl, 2)}", fontsize=12)
        axes[1].axis("off")
        
        fig.colorbar(divplot, ax=fig.get_axes())
        plt.suptitle("Divergence and Curl Fields", fontsize=20)

    if saveDir is not None:
        if np.all([arr is not None for arr in [x2, y2, dx2, dy2, err2]]):
            extension = "compare_"
        else:
            extension = ""
        plt.savefig(os.path.join(saveDir, f"{extension}DivCurl.pdf"))

    plt.show()

def Correlation(
    x, y, dx, dy,
    x2=None, y2=None, dx2=None, dy2=None,
    title1="Observed", title2="GPR Applied",
    xiE_ON=True, xiB_ON=True,
    xiplus_ON=False, ximinus_ON=False,
    xicross_ON=False, xiz2_ON=False,
    rmin=5*u.arcsec, rmax=1.5*u.deg, dlogr=0.05,
    ylim=(-50, 500),
    sep=1e-2*u.deg, avgLine=True,
    printFile=None, saveDir=None
    ):

    # Calculate correlation functions
    correlations = calcCorrelation(x, y, dx, dy,
                                   rmin=rmin, rmax=rmax,
                                   dlogr=dlogr)
    logr, xiplus, ximinus, xicross, xiz2, xiE, xiB = correlations
    r = np.exp(logr)

    # Calculate the indices to average together for each correlation function
    ind = np.where(r <= sep.to(u.deg).value)[0]

    plt.figure(figsize=(10, 10))
    plt.title("Angle Averaged 2-Point Correlation Function of Astrometric Residuals")
    plt.xlabel("Separation (deg)")
    plt.ylabel("Correlation (mas2)")

    avgs = {}
    stds = {}
    
    # Check for second set of data:
    if np.all([arr is not None for arr in [x2, y2, dx2, dy2]]):
        data2 = True
        correlations2 = calcCorrelation(x2, y2, dx2, dy2,
                                        rmin=rmin, rmax=rmax,
                                        dlogr=dlogr)
        logr2, xiplus2, ximinus2, xicross2, xiz22, xiE2, xiB2 = correlations2
        r2 = np.exp(logr2)
        assert np.all(r2 == r), "r and r2 are not the same"
        
        avgs2 = {}
        stds2 = {}
        ratios = {}
    else:
        data2 = False
        
    corrData = {}
    corrTypes = []
    if xiE_ON:
        corrType = "E-Mode"
        corrTypes.append(corrType)
        corrData[corrType] = [xiE, "r"]
        if data2:
            corrData[corrType].append(xiE2)
    if xiB_ON:
        corrType = "B-Mode"
        corrTypes.append(corrType)
        corrData[corrType] = [xiB, "b"]
        if data2:
            corrData[corrType].append(xiB2)
    if xiplus_ON:
        corrType = "xi_+"
        corrTypes.append(corrType)
        corrData[corrType] = [xiplus, "g"]
        if data2:
            corrData[corrType].append(xiplus2)
    if ximinus_ON:
        corrTypes.append(corrType)
        corrData[corrType] = [ximinus, "c"]
        if data2:
            corrData[corrType].append(ximinus2)
    if xicross_ON:
        corrType = "xi_x"
        corrTypes.append(corrType)
        corrData[corrType] = [xicross, "m"]
        if data2:
            corrData[corrType].append(xicross2)
    if xiz2_ON:
        corrType = "xi_z^2"
        corrTypes.append(corrType)
        corrData[corrType] = [xiz2, "y"]
        if data2:
            corrData[corrType].append(xiz22)
            
            
    def plotCorr(r, data, color, marker, title, corr):
        plt.semilogx(
            r,
            data,
            f"{color}{marker}",
            label=f"{corr} {title}",
            alpha=0.5)
            
    for i, corr in enumerate(corrTypes):
        plotCorr(
            r,
            corrData[corr][0],
            corrData[corr][1],
            ".",
            title1,
            corr)
        avg = np.nanmean(corrData[corr][0][ind])
        std = np.nanstd(corrData[corr][0][ind])
        corrData[corr].append(avg)
        corrData[corr].append(std)
        
        if data2:
            plotCorr(
                r,
                corrData[corr][2],
                corrData[corr][1],
                "o",
                title2,
                corr)
            avg2 = np.nanmean(corrData[corr][2][ind])
            std2 = np.nanstd(corrData[corr][2][ind])
            ratio = np.abs(avg / avg2)
            corrData[corr].append(avg2)
            corrData[corr].append(std2)
            corrData[corr].append(ratio)
            plt.text(0.11, 285-15*i,
                f"{corr:<7}: {np.round(ratio, 3)}",
                **{"fontname": "monospace"})


    corrTypes.insert(0, " "*20)
    data2_int = int(np.logical_not(data2))
    means = [f"{np.round(corrData[corr][3-data2_int], 3):<10}"
             for corr in corrTypes[1:]]
    stds = [f"{np.round(corrData[corr][4-data2_int], 3):<10}"
            for corr in corrTypes[1:]]
    
    meantitle1 = "Mean " + title1
    means.insert(0, f"{meantitle1:<20}")
    
    stdtitle1 = "Std  " + title1
    stds.insert(0, f"{stdtitle1:<20}")
    
    if data2:
        means2 = [f"{np.round(corrData[corr][5], 3):<10}" 
                  for corr in corrTypes[1:]]
        stds2 = [f"{np.round(corrData[corr][6], 3):<10}"
                 for corr in corrTypes[1:]]
        
        meantitle2 = "Mean " + title2
        means2.insert(0, f"{meantitle2:<20}")
        
        stdtitle2 = "Std  " + title2
        stds2.insert(0, f"{stdtitle2:<20}")
        
        ratios = [f"{np.round(corrData[corr][7], 3):<10}"
                for corr in corrTypes[1:]]
        ratiotitle = "Mean Ratio"
        ratios.insert(0, f"{ratiotitle:<20}")
        
    corrInfo = f"For the first {len(ind)} points..." + "\n"
    corrInfo += "".join([f"{corr:<10}" for corr in corrTypes]) + "\n"
    if data2:
        corrInfo += "".join(means) + "\n"
        corrInfo += "".join(means2) + "\n\n"
        corrInfo += "".join(stds) + "\n"
        corrInfo += "".join(stds2) + "\n\n"
        corrInfo += "".join(ratios)
    else:
        corrInfo += "".join(means) + "\n"
        corrInfo += "".join(stds) + "\n"
    print(corrInfo)

    plt.xlim((r[0], r[-1]))
    plt.ylim((ylim[0], ylim[1]))
            
    plt.axhline(y=0, c="k")
    if avgLine:
        plt.axvline(x=r[ind[-1]+1], c="k", ls=":")

    plt.grid()
    plt.legend()
    plt.show()
    
    if saveDir is not None:
        if np.all([arr is not None for arr in [x2, y2, dx2, dy2, err2]]):
            extension = "compare_"
        else:
            extension = ""
        plt.savefig(os.path.join(savedir, f"{extension}2ptCorr.pdf"))
        
    if printFile is not None:
        with open(printFile, mode='a+') as file:
                file.write(corrInfo)

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

def calcDivCurl(x, y, dx, dy):
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
    
    return div, curl

def calcCorrelation(x, y, dx, dy, rmin=5*u.arcsec, rmax=1.5*u.deg, dlogr=0.05):
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
        raise TypeError("All input arrays must be of type astropy.units.quantity.Quantity.")

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
