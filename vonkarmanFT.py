# Astrometric covariances from von Karman turbulence
# This version incorporates wind smear and slant angle
# into a 2d FFT

import numpy as np
from matplotlib import pyplot as pl
from scipy.special import j0,j1,jv
from scipy.interpolate import interp1d,RectBivariateSpline
import concurrent.futures

class TurbulentLayer:
    # Calculates astrometric turbulence correlation functions from
    # ray-optic model of a Von Karman turbulent layer of atmosphere.
    def __init__(self,
                 variance=100.,
                 outerScale=100./1e4*180./np.pi, 
                 diameter=4./1e4*180./np.pi, 
                 wind=(0.,0.),
                 nyquist=4./3600.,
                 nFFT=4096,
                 airmass=1.,
                 parallacticAngle=None):
        '''Specifications for turbulent layer model are:
           variance: total displacement variance (e.g. mas^2)
           outerScale: von Karman outer scale angular size
           diameter: telescope diameter angular size at turbulence height
           wind:     2d wind vector displacement over exposure time
           nyquist:  resolution of the FFT
           nFFT:     size of 2d FFT
           airmass:  airmass of exposure
           parallacticAngle: position angle of direction to zenith (x=0, y=90.)
           
           First argument is in angle-squared (you choose units).
           Others are angles, we will assume degrees where it matters.
           and the arguments of functions will be in this unit.
        '''

        self.chunksize = 4096*64

        # Build the power spectrum
        dk = np.pi / nyquist / (nFFT//2)
        kvals = np.arange(-nFFT//2,nFFT//2) * dk
        # Roll to place DC at 0
        kvals = np.roll(kvals,nFFT//2)
        
        dc = 0  # Index of DC frequency
        ky,kx = np.meshgrid(kvals[:nFFT//2+1],kvals)
        
        # von Karman part, foreshortened as needed
        k0sq = (2*np.pi/outerScale)**2
        if parallacticAngle is None:
            pspec = np.power(kx*kx+ky*ky+k0sq,-11./6.)
        else:
            # Foreshorten along the parallactic direction.
            # Don't worry about overall normalizations since
            # we're renormalizing to zero lag later.
            c = np.cos(parallacticAngle*np.pi/180.)
            s = np.sin(parallacticAngle*np.pi/180.)
            kpar = (kx * c + ky * s)/airmass
            kperp = (ky*c - kx*s)
            pspec = np.power(kpar*kpar+kperp*kperp+k0sq,-11./6.)
            
        # Multiply by telescope aperture factor
        kR = np.sqrt(kx*kx+ky*ky) * (diameter / 2.)
        airy = (j1(kR)/(kR))**2
        # Set DC to fix division by zero
        airy[dc,dc] = 0.25
        pspec *= airy

        # Multiply by sinc function for wind smear
        if np.any(wind):
            # Get component of k along wind direction
            kWind = (kx * (wind[0]/2.) + ky * (wind[1]/2.))
            # Make sure we get k=0 right
            w = np.where(kWind==0, 1., np.sin(kWind) / kWind)
            pspec *= (w*w)
            
        # Do the FFT's
        uu = np.fft.irfft2(kx*kx*pspec)
        uv = np.fft.irfft2(kx*ky*pspec)
        vv = np.fft.irfft2(ky*ky*pspec)
        # Normalize total variance
        norm = np.abs(variance) / (uu[dc,dc] + vv[dc,dc])
        uu *= norm
        uv *= norm
        vv *= norm
        # Roll DC to center+1
        n2 = nFFT//2
        uu = np.roll(uu,n2,(0,1))
        uv = np.roll(uv,n2,(0,1))
        vv = np.roll(vv,n2,(0,1))

        # Make LUT's.  We only need to save half-plane since
        # the correlations functions have inversion symmetry
        x = np.arange(-n2,n2)*nyquist
        self.xMax = x[-1]  # Save table extent
#         print('FFT radius',self.xMax)
        self.uuTable = RectBivariateSpline(x,x[n2:],uu[:,n2:])
        self.uvTable = RectBivariateSpline(x,x[n2:],uv[:,n2:])
        self.vvTable = RectBivariateSpline(x,x[n2:],vv[:,n2:])

        return
        
    
    def _cuv(self,xx,yy):
        # Flip data onto y>=0 half-plane
        ss = np.where(yy>0,1.,-1.)
        x = xx*ss
        y = yy*ss
        return self.uuTable(x,y,grid=False),self.uvTable(x,y,grid=False),self.vvTable(x,y,grid=False)

    def getCuv(self,x,y, workers=24):
        # Return array giving u,v covariance matrix at vector separations
        # (x,y).  output has shape x.shape+(2,2)
        # Includes convolution with wind
        s = x.shape
        # Cut out out-of-bounds points or RectBivariantSpline chokes.
        oob = np.logical_or(np.abs(x)>self.xMax, np.abs(y)>self.xMax)
        if np.any(oob):
            raise ValueError('getCuv requested beyond calculated radius',self.xMax)
        out = np.zeros(s+(2,2),dtype=float)
        # Parcel out each job to a pool of workers in chunks
        
        xx = x.flatten()
        yy = y.flatten()
        xchunks = [xx[i:i+self.chunksize] for i in range(0,xx.shape[0],self.chunksize)]
        ychunks = [yy[i:i+self.chunksize] for i in range(0,xx.shape[0],self.chunksize)]
        #with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            tmp= list(executor.map(self._cuv,xchunks,ychunks))
        out[...,0,0] = np.concatenate([i[0] for i in tmp]).reshape(s)
        out[...,1,1] = np.concatenate([i[2] for i in tmp]).reshape(s)
        out[...,0,1] = np.concatenate([i[1] for i in tmp]).reshape(s)
        #out[...,0,0] = self.uuTable(x.flatten(),y.flatten(),grid=False).reshape(s)
        #out[...,1,1] = self.vvTable(x.flatten(),y.flatten(),grid=False).reshape(s)
        #out[...,0,1] = self.uvTable(x.flatten(),y.flatten(),grid=False).reshape(s)
        out[...,1,0] = out[...,0,1]
        return out

    
# Below are some useful plotting routines

def plotCuv(model,xMax=0.5,N=256):
    '''Make plots of Cuv components.
    model: an instance of TurbulentLayer that makes Cuv's
    xMax:  Range of separation to plot
    N:     pixels per side in images.
    '''
    dx = xMax / (N/2) 
    xx = np.arange(-N/2,N/2)*dx
    x = np.ones((N,N),dtype=float) * xx
    y = np.array(x.transpose())
    cuv = model.getCuv(x,y)
    fig,axes = pl.subplots(2,2,figsize=(10,8))
    i00 = axes[0,0].imshow(cuv[:,:,0,0]+cuv[:,:,1,1],
                     interpolation='nearest',origin='lower',cmap='Spectral',
                    extent=(-xMax,xMax,-xMax,xMax))
    axes[0,0].title.set_text('xi_plus')
    fig.colorbar(i00,ax=axes[0,0])
    
    i01 = axes[0,1].imshow(cuv[:,:,0,0],
                     interpolation='nearest',origin='lower',cmap='Spectral',
                    extent=(-xMax,xMax,-xMax,xMax))
    axes[0,1].title.set_text('Cuu')
    fig.colorbar(i01,ax=axes[0,1])
    
    i10 = axes[1,0].imshow(cuv[:,:,0,1],
                     interpolation='nearest',origin='lower',cmap='Spectral',
                    extent=(-xMax,xMax,-xMax,xMax))
    axes[1,0].title.set_text('Cuv')
    fig.colorbar(i10,ax=axes[1,0])
    
    i11 = axes[1,1].imshow(cuv[:,:,1,1],
                     interpolation='nearest',origin='lower',cmap='Spectral',
                    extent=(-xMax,xMax,-xMax,xMax))
    axes[1,1].title.set_text('Cvv')
    fig.colorbar(i11,ax=axes[1,1])
    
    return


def plotCuts(model,rMin,rMax,dLnR=0.2):
    ''' Make plots of radial run of covariance along major/minor axes
    model: an instance with getCuv method
    rMin,rMax: range to cover on x-axis of semi-log plot
    dLnR:  spacing of points along x axis
    '''

    # Select radii to probe
    rr = np.exp(np.arange(np.log(rMin),np.log(rMax)+dLnR,dLnR))
    # Angles
    theta = np.pi*np.arange(100)/100.
    cc = np.cos(theta)
    ss = np.sin(theta)
    xiplus = []
    major = []
    for r in rr:
        xx = r*cc
        yy = r*ss
        cuv = model.getCuv(xx,yy)
        tr = cuv[:,0,0]+cuv[:,1,1]
        xiplus.append(np.mean(tr))
        major.append(theta[np.argmax(tr)])

    # Plot the circular average
    pl.semilogx(rr,xiplus,'k-', label='xiplus avg')
    
    # Now probe major axis
    major = np.median(major)
    cc = np.cos(major)
    ss = np.sin(major)
    xx = rr * cc
    yy = rr * ss
    
    cuv = model.getCuv(xx,yy)
    pl.semilogx(rr, cuv[:,0,0]+cuv[:,1,1], 'r-', label='xiplus major')
    pl.semilogx(rr, cc*cc*cuv[:,0,0]+ss*ss*cuv[:,1,1]+2*cc*ss*cuv[:,0,1],
                'r--', label='radial major')
    pl.semilogx(rr, ss*ss*cuv[:,0,0]+cc*cc*cuv[:,1,1]-2*cc*ss*cuv[:,0,1],
                'r:', label='xverse major')
    
    # minor axis
    cc,ss = -ss, cc
    xx = rr * cc
    yy = rr * ss
    
    cuv = model.getCuv(xx,yy)
    pl.semilogx(rr, cuv[:,0,0]+cuv[:,1,1], 'b-', label='xiplus minor')
    pl.semilogx(rr, cc*cc*cuv[:,0,0]+ss*ss*cuv[:,1,1]+2*cc*ss*cuv[:,0,1],
                'b--', label='radial minor')
    pl.semilogx(rr, ss*ss*cuv[:,0,0]+cc*cc*cuv[:,1,1]-2*cc*ss*cuv[:,0,1],
                'b:', label='xverse minor')
   
    pl.legend(loc=3,framealpha=0.3)
    pl.grid()  
    pl.xlabel("Radius (degrees)")
    pl.ylabel("Covariance")
    
