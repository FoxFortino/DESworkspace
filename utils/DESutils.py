import os
import glob
import pickle

import numpy as np
import astropy.units as u
import astropy.table as tb


# All complete griz exposures from /data3/garyb/tno/y6/zone134.
exps = np.array([
    348819, 355303, 361577, 361580, 361582, 362365, 362366, 364209,
    364210, 364213, 364215, 367482, 367483, 367484, 367488, 369801,
    369802, 369803, 369804, 370199, 370200, 370204, 370600, 370601,
    370602, 370609, 371367, 371368, 371369, 372006, 372064, 372437,
    372522, 373245, 374797, 474260, 474261, 474262, 474263, 474264,
    474265, 476846, 484481, 484482, 484483, 484490, 484491, 484499,
    573396, 573398, 576861, 576862, 576863, 576864, 576865, 576866,
    579815, 579816, 586534, 592152, 674340, 675645, 676791, 676792,
    676799, 676800, 676801, 680497, 681166, 686427, 686457, 686459,
    689611, 689612, 689613, 691478, 696547, 696552, 784503, 788112,
    788113, 788116, 788117, 791184, 791186, 791215, 791229, 791593,
    791640])

bandDict = {
    'g': [
        361580, 367484, 369801, 369803, 370204,
        370601, 371369, 372064, 372522, 474262,
        474265, 484483, 576863, 576866, 579816,
        676792, 676800, 696547, 791229, 791593,
        791640],
    'r': [
        361577, 362366, 367482, 367488, 369802,
        370200, 370600, 370602, 371368, 474261,
        474264, 484481, 576861, 576865, 579815,
        676791, 676799, 676801, 681166, 689612,
        791184, 791186, 791215],
    'i': [
        361582, 362365, 364210, 364213, 367483,
        369804, 370199, 370609, 371367, 474260,
        474263, 484499, 576862, 576864, 586534,
        680497, 686427, 686457, 689611, 696552,
        788113, 788116, 788117],
    'z': [
        348819, 355303, 364209, 364215, 372006,
        372437, 373245, 374797, 476846, 484482,
        484490, 484491, 573396, 573398, 592152,
        674340, 675645, 686459, 689613, 691478,
        784503, 788112]}

# These are the exposures, from /home/fortino/DESworkspace/data/eris.fits,
# that the TNO Eris appears in. There are 9 more exposures in
# /home/fortino DESworkspace/data/eris.fits that don't seem to be present in
# any zone in /data3/garyb/tno/y6 as of July 28, 2020.
erisExps = np.array([
    233221, 234928, 240777, 241125, 246881, 264536, 364725, 364726,
    364727, 370653, 370685, 374544, 374550, 382258, 384049, 388143,
    398226, 398231, 478983, 486775, 488823, 488824, 490665, 503010,
    503041, 507393, 507394, 567918, 568777, 591792, 591793, 596103,
    596104, 596105, 596512, 597221, 597555, 600469, 600470, 683902,
    692049, 692050, 696363, 697638, 697639, 706447, 776431, 782097,
    782098, 782099])

erisBandDict = {
    'Y': [
        234928, 264536, 374544, 384049,
        478983, 591793, 600470, 706447],
    'g': [
        233221, 240777, 241125, 246881,
        364726, 568777, 596105, 597555,
        696363, 697639, 782099],
    'r': [
        364725, 370653, 370685, 382258,
        490665, 567918, 596103, 596512,
        692049, 697638, 782098],
    'i': [
        364727, 388143, 488824, 503041,
        507393, 596104, 597221, 692050,
        782097],
    'z': [
        374550, 398226, 398231, 486775,
        488823, 503010, 507394, 591792,
        600469, 683902, 776431]}

"""
My tileRef file, /home/fortino/DESworkspace/data/expnum_tile.fits.gz, seems
to think that tile /data3/garyb/tno/y6/zone019/DES0139-0124_final.fits
should include exposure 364726, but doesn't.

My tileRef file, /home/fortino/DESworkspace/data/expnum_tile.fits.gz, seems
to think that tile /data3/garyb/tno/y6/zone020/DES0145-0041_final.fits
should include exposure 782099, but doesn't.

The tileRef file is probably correct, but the data in the y6 folder seems to
be somewhat incomplete.
"""


def findTiles(
    expNum: int,
    confirmTiles: bool = False,
    tileRef: str = "/home/fortino/DESworkspace/data/expnum_tile.fits.gz"
        ) -> list:
    """
    Given an exposure, search for all of its constituent tiles.

    Arguments
    ---------
    expNum : int
        DES exposure number.

    Keyword Arguments
    -----------------
    confirmTiles : bool
        Whether or not to take extra time to confirm that each tile
        contains expNum.
    tileRef : str
        Reference file that relates exposure number to DES tile name.

    Returns
    -------
    tilefiles : list
        If confirmTiles is False. List of full path information to DES tiles.
    confirmed_tilefiles : list
        If confirmTiles is True. List of full path information to DES tiles
        that have been confirmed to contain expNum
    """
    # Get a list of all tiles in /data3/garyb/tno/y6/zone???.
    zoneDir = "/data3/garyb/tno/y6"
    zoneWC = "zone[0,1,2][0,1,2,3,4,5,6,7,8,9][0,1,2,3,4,5,6,7,8,9]"
    tileWC = "DES????[+,-]????_final.fits"
    allTiles = sorted(glob.glob(os.path.join(zoneDir, zoneWC, tileWC)))

    # Open the tileRef file that relates exposure number to tile name. Find
    # all of the tile names that include expNum.
    tileRef = tb.Table.read(tileRef)
    tilenames = tileRef[tileRef["EXPNUM"] == expNum]["TILENAME"]
    tilenames = tilenames.data.astype(str)

    tilefiles = []
    for tile in allTiles:

        # Remove the "_final.fits" from the file name to get the DES
        # designation of the tile that appears in tileRef.
        tilename = os.path.basename(tile)[:-11]
        if tilename in tilenames:
            tilefiles.append(tile)

    # Check that there is at least one tile in the zoneDir.
    if len(tilefiles) == 0:
        raise FileNotFoundError("There are no known tiles for exposure "
                                f"{expNum} in {zoneDir}.")

    # Check if tiles should be confimed.
    if not confirmTiles:
        return tilefiles

    # Loop through each tile, open it, and confirm that the exposure is in it.
    confirmed_tilefiles = []
    for tilefile in tilefiles:
        tile = tb.Table.read(tilefile)
        confirm = len(tile[tile["EXPNUM"] == expNum])
        if confirm != 0:
            confirmed_tilefiles.append(tilefile)
        else:
            print(f"{tilefile} doesn't include exposure {expNum}.")

    # Check if at least one tile could be confirmed.
    if len(confirmed_tilefiles) != 0:
        return confirmed_tilefiles
    else:
        print(f"All found tiles did not include exposure {expNum}.")


def getBand(
    expNum: int,
    confirmTiles: bool = False
        ) -> str:
    """
    Given an exposure, figure out what passband it is.

    Arguments
    ---------
    expNum : int
        DES exposure number.

    Keyword Arguments
    -----------------
    confirmTiles : bool
        Whether or not to take extra time to confirm that each tile
        contains expNum.
    Returns
    -------
    band : str
        One of Y, g, r, i, z, the 5 DES passband names.
    """
    # First look through bandDict, containing all of the griz exposures from
    # zone 134. This provides very fast passband lookup times for frequently
    # used exposures.
    for key, value in bandDict.items():
        if expNum in value:
            return key

    # Also try a fast lookup in erisBandDict.
    for key, value in erisBandDict.items():
        if expNum in value:
            return key

    # Find all of the tiles that make up this exposures.
    tilefiles = findTiles(expNum, confirmTiles=confirmTiles)

    # Open any of the tiles and open it.
    tile = tb.Table.read(tilefiles[0])

    # Find which band the exposure is in by indexing the table.
    band = np.unique(tile[tile["EXPNUM"] == expNum]["BAND"])[0]

    return band


def get_allExposures(zoneDir: str) -> list:
    """
    Given an directory of DES zones, find all exposures represented in it.
    
    This function searches through all tile files in a directory. It opens each one and gets the list of
    all exposures for each tile and compiles them in a list. There will obviously be many repeat exposures
    in this list so we take np.unique on the list.
    
    Arugments
    --------
    zoneDir : str
        The directory where the DES tiles are.
        
    Returns
    -------
    allExposures : np.ndarray
        All Exposures that are present in a particular DES zone.
    """
    # Ensure that zoneDir is a real directory
    if not os.path.isdir(zoneDir):
        raise FileNotFoundError(f"Directory does not exist: {zoneDir}")
    
    tileWC = "DES????[+,-]????_final.fits"
    tiles = sorted(glob.glob(os.path.join(zoneDir, tileWC)))
    allExposures = []
    for tile in tiles:
        table = tb.Table.read(tile)
        exps = np.unique(np.array(table["EXPNUM"]))
        allExposures.extend(exps)
        
    # Exposures number 999999 is the exposure number given to coadd objects.
    allExposures = np.delete(np.unique(allExposures), np.argmax([np.unique(allExposures) == 999999]))
    
    return allExposures

def get_bandDict(
    zoneDir: str,
    allExposures: list,
    tileRef: str = "/home/fortino/DESworkspace/data/expnum_tile.fits.gz"
        ) -> tuple:
    """
    Return a list of complete exposures and its bandDict.
    
    Given a list of DES exposures and a DES zone directory, return a list of only 
    the complete exposures in that zone, as well as the bandDict that organizes the
    complete exposures based on the DES passband of that exposure.
    
    Arugments
    ---------
    exposures : list
        List of DES exposure numbers.
        
    Keyword Arguments
    -----------------
    tileRef : str
        Reference file that relates exposure number to DES tile name.
    
    Returns
    -------
    completeExposures : list
        List of all complete exposures in the zone.
    bandDict : dict
        Dictionary of complete exposures, organized by their DES passband.
    """
    tileRef = tb.Table.read(tileRef)
    
    completeExposures = []
    bandDict = {"g": [], "r": [], "i": [], "z": [], "Y": []}
    for expNum in allExposures:

        # Find all tiles for the particular exposure.
        tilefiles = findTiles(expNum)
        for tilefile in tilefiles:
            
            # Get the directory of the tile. Each zone has its own directory,
            # so making sure that this directory matches zoneDir will ensure that
            # each tile is inside of zoneDir.
            directory = os.path.dirname(tilefile)
            if directory != zoneDir:
                break
        else:
            completeExposures.append(expNum)
            band = getBand(expNum)
            bandDict[band].append(expNum)
            
    return completeExposures, bandDict

def createBandDict(
    zoneDir: str,
    savePath: str = "/home/fortino/DESworkspace/data",
    searchPath: str = "/home/fortino/DESworkspace/data"
        ) -> tuple:
    """
    Given a DES zone directory, find all complete exposures in it.
    
    A complete exposure in a zone is one where all of its associated tiles lie inside
    the zone. get_allExposures will return all exposures, complete and incomplete, from 
    a particular zone. This function will return only complete exposures. Additionally, 
    this function return bandDict, a dictionary where the keys are DES passband letters 
    (g, r, i, z, Y) and the values are lists of exposures.
    
    Additionally, this function will save
    
    Arguments
    ---------
    zoneDir : str
        The directory where the DES tiles are.
    
    Keyword Arguments
    -----------------
    savePath : str
        The path to save the pickle files to.
    searchPath : str
        The path to look for pickle files.
        
    Returns
    -------
    completeExposures : list
        List of all complete exposures in the zone.
    bandDict : dict
        Dictionary of complete exposures, organized by their DES passband.
    """
    
    # Get the DES zone from the path.
    zone = os.path.basename(zoneDir)
    
    # Load the completeExposures list and bandDict dictionary if they exist.
    if searchPath is not None:
        bdfile = os.path.join(searchPath, zone+"_bd.pkl")
        cefile = os.path.join(searchPath, zone+"_ce.pkl")
        if os.path.isfile(bdfile) and os.path.isfile(cefile):
            with open(bdfile, "rb") as file:
                bandDict = pickle.load(file)
            with open(cefile, "rb") as file:
                completeExposures = pickle.load(file)
            return completeExposures, bandDict
    
    # Get all of the exposures in the zone.
    allExposures = get_allExposures(zoneDir)
    
    # Narrow down all of the exposures to only the complete ones, and get the
    # completeExposures list and the bandDict dictionary.
    completeExposures, bandDict = get_bandDict(zoneDir, allExposures)

    # Save the completeExposures list and the bandDict dictionary as pickle files.
    if savePath is not None:
        bdfile = os.path.join(savePath, zone+"_bd.pkl")
        with open(bdfile, "wb") as file:
            pickle.dump(bandDict, file)

        cefile = os.path.join(savePath, zone+"_ce.pkl")
        with open(cefile, "wb") as file:
            pickle.dump(completeExposures, file)

    return completeExposures, bandDict


class parseOutfile(object):

    def __init__(self, OUTfile):
        root, ext = os.path.splitext(OUTfile)
        self.FITSfile = root + ".fits"

        # Open the OUTfile and read all of the lines.
        with open(OUTfile, "r") as f:
            out = f.readlines()

        # Check if the algorithm ended because of an np.linalg.LinAlgErr.
        if "LinAlgError" in out[-1]:
            self.LinAlgErr = True
            self.LinAlgErr_params = out[-1]
            self.finished = False
            return
        else:
            self.LinAlgErr = False

        # Check for completion.
        if "Total Time" not in out[-6]:
            self.finished = False
            return
        else:
            self.finished = True

        # While looping through the OUTfile, these flags are necessary to
        # determine which part of the file the current line belongs to.
        fitCorr = False
        opt1 = False
        opt2 = False

        # Initialize variables for counting the number of iterations each type
        # of optimization took.
        self.nfC = 0
        self.nOpt1 = 0
        self.nOpt2 = 0

        # Start looping through the lines of the OUTfile.
        for line in out:

            # The very first line of the file will have "RSS" in it. This is
            # how we know when to start counting the number of fitCorr steps.
            if "RSS" in line:
                fitCorr = True

            # If "xi +" is in the line then we are starting the first or
            # second round of GP optimization.
            elif "xi +" in line:

                # If we are starting the first or second round of GP
                # optimization, the fitCorr flag must be False.
                fitCorr = False

                # If opt1 is already true, then that means we are starting the
                # second round of GP optimization, so we should set opt1 to
                # False and opt2 to True.
                if opt1:
                    opt1 = False
                    opt2 = True

                # If opt1 is not already True then that means we are starting
                # the first rough of GP optimization
                else:
                    opt1 = True

            # If "Time" is in the line the we are at the very end of the
            # OUTfile and need to start parsing the lines to retrieve the time
            # values for each step.
            elif "Time" in line:
                def parseTime(line, unit):
                    return float(line.split(":")[1][:7])*unit

                if "Total" in line:
                    self.totalTime = parseTime(line, u.hr)
                elif "Load" in line:
                    self.loadTime = parseTime(line, u.s)
                elif "Correlation Fitting" in line:
                    self.fitCorrTime = parseTime(line, u.min)
                elif "Correlating Fitting Jackknife" in line:
                    self.fitCorrJKTime = parseTime(line, u.min)
                elif "Optimization Time" in line:
                    self.optTime = parseTime(line, u.hr)
                elif "Optimization Jackknife" in line:
                    self.optJKTime = parseTime(line, u.min)

            # If none of the above are true then we must be at a line of
            # iteration, so we count it for the correct counter.
            else:
                if fitCorr:
                    self.nfC += 1
                if opt1:
                    self.nOpt1 += 1
                if opt2:
                    self.nOpt2 += 1

        self.nGP = self.nOpt1 + self.nOpt2
        self.avgGPTime = (self.optTime / self.nGP).to(u.min)


def get_LinAlgErrors(OUTfiles: list):
    if len(OUTfiles) == 0:
        raise Exception("OUTfiles is empty.")

    for OUTfile in OUTfiles:
        parseOut = parseOutfile(OUTfile)
        if parseOut.LinAlgErr:
            print(parseOut.FITSfile)
            print(parseOut.LinAlgErr_params)


def get_expAvg_Times(
    OUTfiles: list,
    GPtime: bool = False,
    optTime: bool = False,
    returnPercs: bool = False,
    returnList: bool = False
        ) -> float:
    """
    Find the optimzation time, averaged across multiple exposures.

    Arguments
    ---------
        OUTfiles : list
            List of .out files from GPR.

    Keyword Arguments
    -----------------
        GPtime : bool
            If True, the value of interest will be the average GP calculation
            time.
        optTime : bool
            If True, the value of interest will be the total optimzation time.
        returnPercs : bool
            Instead of returning the value of interest averaged across all
            exposures, return the 25th, 50th, and 75th percentiles of the
            value for all exposures.
        returnList : bool
            Instead of returning the value of tinerest averaged across all
            exposures, return the list of values for all exposures.

    Returns
    -------
        valueAvg : float
            If returnPercs and returnList are True, return the value of
            interest averaged across all exposures.
        valueList : list
            If returnList is True, return the list of values for all exposures.
        percs : list
            If returnPerds is True, return the 25th, 50th, and 75th
            percentiles of the value for all exposures.
    """
    if len(OUTfiles) == 0:
        raise Exception("OUTfiles is empty.")

    times = np.array([])
    for OUTfile in OUTfiles:
        parseOut = parseOutfile(OUTfile)
        if parseOut.finished:
            if GPtime:
                value = parseOut.avgGPTime
            elif optTime:
                value = parseOut.optTime
            times = np.append(times, value.value)
        else:
            continue

    if len(times) == 0:
        raise Exception("There are no finished files.")

    if returnPercs:
        valuePercs = np.percentile(times, (25, 50, 75))*(value.unit)
        return valuePercs
    elif returnList:
        valueList = times*(parseOut.value.unit)
        return valueList
    else:
        valueAvg = np.mean(times)*(parseOut.value.unit)
        return valueAvg
