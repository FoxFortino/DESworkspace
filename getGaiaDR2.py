# Routine to make FITS file in our format with the Gaia stars in a chosen box on sky
# Version to acquire DR2 data, including PM and parallax
from __future__ import print_function
from astroquery.gaia import Gaia
from astropy.coordinates import SkyCoord
import numpy as np
import sys

def getGaiaBox(ra, dec, width, height=None):
    """
    Acquire table of all Gaia stars in a box centered at ra, dec
    and having half-side-length equal to radius.
    All three arguments should be in degrees.
    Returns an astropy Table object.
    """
    
    if height is None:
        height = width

    query ="SELECT s.source_id, s.ra, s.ra_error, s.dec, s.dec_error, s.parallax, s.parallax_error, " + \
               "s.pmra, s.pmra_error, s.pmdec, s.pmdec_error, " + \
               "s.ra_dec_corr, s.ra_parallax_corr, s.ra_pmra_corr, s.ra_pmdec_corr, " + \
               "s.dec_parallax_corr, s.dec_pmra_corr, s.dec_pmdec_corr, " + \
               "s.parallax_pmra_corr, s.parallax_pmdec_corr, " + \
               "s.pmra_pmdec_corr, " + \
               "s.phot_g_mean_mag as Gmag, s.phot_g_mean_flux_over_error as g_sn, " + \
               "s.phot_bp_mean_mag as bpmag, s.phot_rp_mean_mag as rpmag " + \
               "FROM gaiadr2.gaia_source s " + \
               "INNER JOIN gaiadr2.ruwe r on s.source_id=r.source_id " + \
               "WHERE s.astrometric_params_solved=31 " + \
               " AND r.ruwe<1.4 AND s.visibility_periods_used>8 AND " + \
               "CONTAINS(POINT('ICRS',s.ra,s.dec),BOX('ICRS',{:f},{:f},{:f},{:f}))=1;".format(\
                                                                ra, dec, \
                                                                width/np.cos(dec*np.pi/180.), height)
    job = Gaia.launch_job_async(query, dump_to_file=False)
    #print("Job launched")
    #print(job)
    return job.get_data()

def getGaiaCat(ra, dec, width, height=None):
    if height is None:
        height = width
    mas = 0.001/3600.  # One mas, in degrees
    tab = getGaiaBox(ra,dec,width,height=height)
    # Make a column that has a circular error estimate
    tab['error'] = 0.5*(tab['ra_error']+tab['dec_error'])*mas

    # And one that has G mag error
    tab['gmag_error'] = 0.4*np.log(10.) / np.abs(tab['g_sn'])

    # Get rid of potentially unreliable Gaia sources
    use = np.logical_and( tab['error']<3*mas,   # < 3 mas positional error
                          tab['g_sn'] > 10)     # or S/N<10 in photometry

    # We won't use things that have statistically significant excess Gaia
    # position noise above 2 mas.
    # noisy = np.logical_and(tab['astrometric_excess_noise']>2,
    #                        tab['astrometric_excess_noise_sig']>3)
    # use = np.logical_and(use, np.logical_not(noisy))
    
    tab = tab[use] # Filter catalog with above.
    
    #add some new things to go into the FITS header
    # Reorganize all the error information into covariance matrix
    cov = np.ones( (len(tab),25), dtype=np.float32)
    dra = tab['ra_error']    # Should be in mas
    ddec = tab['dec_error']
    dparallax = tab['parallax_error'] # Keep parallax, PM in mas
    dpmra = tab['pmra_error']
    dpmdec = tab['pmdec_error']
    # Here is the standard ordering of components in the cov matrix,
    # to match the PM enumeration in C++ code of gbdes package's Match.
    # Each tuple gives: the array holding the 1d error,
    #                   the string in Gaia column names for this
    #                   the ordering in the Gaia catalog
    # and the ordering of the tuples is the order we want in our cov matrix
    stdOrder = ( (dra, 'ra',0),
                 (ddec, 'dec',1),
                 (dpmra, 'pmra',3),
                 (dpmdec, 'pmdec',4),
                 (dparallax, 'parallax',2) )
    #
    k = 0
    for i, pr1 in enumerate(stdOrder):
        for j,pr2 in enumerate(stdOrder):
            if pr1[2]<pr2[2]:
                # add correlation coefficient
                cov[:,k] =  pr1[0] * pr2[0] * tab[pr1[1] + '_' + pr2[1] + '_corr']
            elif pr1[2]>pr2[2]:
                # add correlation coefficient
                cov[:,k] =  pr1[0] * pr2[0]* tab[pr2[1] + '_' + pr1[1] + '_corr']
            else:
                # diagnonal element
                cov[:,k] = pr1[0] * pr2[0]
            k = k+1
    tab['cov'] = cov
    tab.remove_columns(['ra_error', 'dec_error', 'parallax_error',
            'pmra_error', 'pmdec_error', 'ra_dec_corr', 'ra_parallax_corr',
            'ra_pmra_corr', 'ra_pmdec_corr','dec_parallax_corr',
            'dec_pmra_corr', 'dec_pmdec_corr','parallax_pmra_corr',
            'parallax_pmdec_corr','pmra_pmdec_corr'])

    tab.meta['RA'] = ra
    tab.meta['DEC'] = dec
    tab.meta['WIDTH'] = width
    tab.meta['HEIGHT'] = height
    tab.meta['EPOCH']=2015.5  # Gaia DR2 epoch
    tab.meta['MJD']=57206.  # Gaia DR2 epoch in MJD
    return tab;

if __name__=='__main__':
    if len(sys.argv)==2:
        import easyaccess as ea
        conn = ea.connect()
        zone = sys.argv[1]
        tab = conn.query_to_pandas('''
         SELECT 0.5*(RAMIN+RAMAX) as RA, 0.5*(DECMIN+DECMAX) as DEC
         FROM rbutler.deszones@dessci
         WHERE zone={:s}'''.format(zone))
        conn.close()
        ra = tab['RA'][0]
        dec = tab['DEC'][0]
        print("Searching at ",ra,dec)
        width = 5.5 # Use standard box size
        out = 'zone{:03d}.gaia.pmcat'.format(int(zone))
    elif len(sys.argv)==5:
        ra = float(sys.argv[1])
        dec = float(sys.argv[2])
        width = float(sys.argv[3])
        out = sys.argv[4]
    else:
        print("Usage:\n getGaiaDR2.py <zone>\n -OR- \n"
                      " getGaiaDR2.py <ra> <dec> <width> <catname>")
        sys.exit(1)
    tab = getGaiaCat(ra,dec,width)
    tab.write(out, format='fits', overwrite=True)


    # ?? Could trim the catalog to RA/Dec limits of zone
    # Also could write this to obtain DR2 for a single exposure
    sys.exit(0)

