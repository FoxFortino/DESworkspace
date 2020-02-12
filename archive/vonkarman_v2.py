# Astrometric covariances from von Karman turbulence

import numpy as np
from matplotlib import pyplot as pl
from scipy.special import j0,j1,jv
from scipy.integrate import quad
from scipy.interpolate import interp1d

class TurbulentLayer:
    # Calculates astrometric turbulence correlation functions from
    # ray-optic model of a Von Karman turbulent layer of atmosphere.
    def __init__(self,
                 variance=100.,
                 outerScale=100./1e4*180./np.pi, 
                 diameter=4./1e4*180./np.pi, 
                 wind=(0.,0.),
                 dWind = 5/3600.,
                 dLnR = 0.1):
        '''Specifications for turbulent layer model are:
           variance: total displacement variance (e.g. mas^2)
           outerScale: von Karman outer scale angular size
           diameter: telescope diameter angular size at turbulence height
           wind:     2d wind vector displacement over exposure time
           dWind:    step size for integrating over wind smear
           dLnR:     interval in ln(R) for building radial lookup tables
           
           First argument is in angle-squared (you choose units).
           Second through fifth are angles; they must be the same unit
           (presumably degrees) and the arguments of functions will
           be in this unit.
        '''
        self.dLnR = dLnR
        self.setVariance(variance)
        self.setScales(outerScale, diameter)
        self.setWind(wind, dWind)
        return
        
    def setVariance(self, variance):
        # Set total displacement variance at zero lag
        self.variance = variance

    def setWind(self,wind,dWind=5./3600.):
        '''Change wind vector: 
        wind: 2d wind total angular displacement during exposure. wind=None
              or wind=(0,0) will shut off wind convolution.
        dWind: step size for wind convolution
        '''
        if wind is None:
            self.hasWind = False
        elif not np.any(np.array(wind)):
            # Zero wind vector
            self.hasWind = False
        else:
            # finite wind vector
            self.hasWind = True
            self.wind = np.array(wind)
            if self.wind.shape != (2,):
                raise ValueError("wind vector is not 2d")
                
        if self.hasWind:
            # set up wind convolution sampling
            self.dWind = dWind
            nsteps = int(np.ceil(np.hypot(self.wind[0],self.wind[1])/self.dWind))
            frac = np.arange(nsteps,dtype=float) / nsteps
            wt = np.concatenate((frac,np.ones(1),frac[::-1]))
            # This is weight to assign to each convolution point
            self.windWeight = wt / np.sum(wt)
            # Now get the displacements associated with each displacement
            # Convolution will extend to +-(wind) with these fractions
            frac = np.concatenate((1-frac,np.zeros(1),frac[::-1]-1))
            # save Nx2 array of displacements
            self.windShift = frac[:,np.newaxis] * self.wind
            
        # Trigger new normalization integral
        self.iNorm = None
            
        return
                
    def setScales(self,outerScale, diameter):
        # Change outer scale and telescope diameter,
        # which requires new LUT's for I0/I2.
        # And w
        self.k0sq = (2*np.pi/outerScale)**2
        self.R = 0.5*diameter
        # Bounds for Bessel integrations
        self.kMin = self.k0sq/1e3
        self.kMax = 20. / self.R

        # Flush any interpolation tables that exist
        self.rLUT = None
        self.iLUT = None

        # Trigger new normalization integral
        self.iNorm = None
            
        return
    
    def renormalize(self):
        # Calculate the normalization integral
        # First set the normalization to unity
        self.iNorm = 1.
        # Then calculate covariance matrix at zero lag
        zero = np.zeros(1,dtype=float)
        cuv = self.getCuv(zero,zero) / self.variance
        self.iNorm = 1. / (cuv[...,0,0]+cuv[...,1,1])
        # Now when we multiply integrals by iNorm, they will
        # yield unit displacement variance.
        return

    def integrateI02(self, r):
        # Do a single pair of Bessel integrals
        # yielding I0, I2 at r.
        def integrand(k):
            return np.power(k,3.)*np.power(k*k+self.k0sq,-11./6.) \
              * (j1(k*self.R)/(k*self.R))**2 * j0(k*r)
        i0 =  quad(integrand,self.kMin,self.kMax,limit=200)[0]
        def integrand(k):
            return np.power(k,3.)*np.power(k*k+self.k0sq,-11./6.) \
              * (j1(k*self.R)/(k*self.R))**2 * jv(2.,k*r)
        i2 =  quad(integrand,self.kMin,self.kMax,limit=200)[0]
        return i0,i2
    
    def buildLUT(self, rMin, rMax):
        # Extend the LUT to include range from rmin to rmax
        # Express new bounds in steps of dLnR
        iBegin = int(np.floor(np.log(rMin)/self.dLnR))
        iEnd = int(np.ceil(np.log(rMax)/self.dLnR)) + 1  # one-past-end
        if iEnd-iBegin<4:
            iBegin = iEnd-4 # Spline table needs minimum 4 points???
        
        freshLUT = False # Set if we make a new LUT

        if self.rLUT is None:
            # Start afresh
            self.iBeginLUT = iBegin
            self.rLUT = np.exp(self.dLnR * np.arange(iBegin,iEnd))
            # Make Nx2 array holding I0,I2
            self.iLUT = np.array( [self.integrateI02(r) for r in self.rLUT])
            
            # Also create values for I0 and I2 at zero, since LUT will be
            # over log(r):
            self.iAtOrigin = np.array(self.integrateI02(0.))
            self.iAtOrigin[1] = 0.  # This must be true.
            freshLUT = True
        
        if iBegin < self.iBeginLUT:
            ###print("Building LUT down to",iBegin)###
            # Do integrals for values below current LUT
            rPre = np.exp(self.dLnR * np.arange(iBegin,self.iBeginLUT))
            iPre = np.array( [self.integrateI02(r) for r in rPre])
            self.iBeginLUT = iBegin
            self.rLUT = np.concatenate((rPre,self.rLUT))
            self.iLUT = np.concatenate((iPre,self.iLUT))
            freshLUT = True
            
        if iEnd > self.iBeginLUT + self.rLUT.shape[0]:
            ###print("Building LUT up to",iEnd)###
            rPost = np.exp(self.dLnR * np.arange(self.iBeginLUT+self.rLUT.shape[0],
                                            iEnd))
            iPost = np.array( [self.integrateI02(r) for r in rPost])
            self.rLUT = np.concatenate((self.rLUT,rPost))
            self.iLUT = np.concatenate((self.iLUT,iPost))
            freshLUT = True           
            
        if freshLUT:
            # Note that the interpolator is set to return the
            # zero-lag value when the input log(r) is out of bounds,
            # which occurs for zero lag.
            ###print(np.log(self.rLUT),self.iLUT)###
            self.lnrLUT = np.log(self.rLUT)
            self.interpolator = interp1d(np.log(self.rLUT), self.iLUT,
                                         axis=0,kind='cubic',
                                         fill_value=self.iAtOrigin,
                                         bounds_error=False)
            ##self.iDiff = self.iLUT[1:,:] - self.iLUT[:-1,:]
            
        return
            
    def getI02(self,r):
        # Return Nx2 array of I0,I2 values at the N values of r provided.
        # Exclude anything at zero when building LUT
        if np.all(r==0.):
            self.buildLUT(0.1,0.1)  # Build at some random value
        else:
            ###print(r)###
            self.buildLUT(np.min(r[r>0]),np.max(r))
        return  self.interpolator(np.log(r))

        # Make my own interpolator that exploits fixed spacing
        #index = np.clip(np.log(r) / self.dLnR - self.iBeginLUT, 0., self.iLUT.shape[0]-1.0001)
        #ii = np.array(np.floor(index),dtype=int)
        #frac = index - ii
        #out = self.iLUT[ii,:] + frac[...,np.newaxis]*self.iDiff[ii,:]
        #return out
    
    def _CuvNoWind(self,x,y):
        # Return arrays giving u,v covariance matrix components at vector separations
        # (x,y).  3 outputs have same shape as x,
        # and give (uu+vv)/2, (vv-uu)/2, and uv components
        # No wind smearing or normalizations applied here, so
        # that all of this code is invariant under wind or variance change.
        rsq = x*x + y*y
        # Alter rsq at origin to avoid divide-by-zero warnings
        rcalc = np.where(rsq>0,rsq,1.)
        c2 = (x*x - y*y) / rcalc
        s2 =  -2 * x * y / rcalc
        i02 = self.getI02(np.sqrt(rsq))
        
        ## Build Cuv - old Way, now do this in getCuv
        #out = np.zeros(x.shape+(2,2),dtype=float)
        #out[...,0,0] = i02[...,0] - c2 * i02[...,1]
        #out[...,1,1] = i02[...,0] + c2 * i02[...,1]
        #out[...,1,0] = s2 * i02[...,1] 
        #out[...,0,1] = out[...,1,0]
        return i02[...,0], c2*i02[...,1], s2*i02[...,1]
    
    def getCuv(self,x,y):
        # Return array giving u,v covariance matrix at vector separations
        # (x,y).  output has shape x.shape+(2,2)
        # Includes convolution with wind
        
        if self.iNorm is None:
            self.renormalize()
            
        out = np.zeros(x.shape+(2,2),dtype=float)
        tr = np.zeros_like(x)
        c = np.zeros_like(x)
        s = np.zeros_like(x)
        if self.hasWind:
            # Break the x,y arrays into chunks and do the
            # wind sum one chunk at a time, in hopes
            # of keeping all of the sums over w in cache.
            xf = x.flatten()
            yf = y.flatten()
            tr = np.zeros_like(xf)
            c = np.zeros_like(xf)
            s = np.zeros_like(xf)
            n = xf.shape[0]
            chunk = (512*512) // len(self.windShift)
            for iStart in range(0,n,chunk):
                iEnd = np.minimum(iStart+chunk,n)
                xx = xf[iStart:iEnd, np.newaxis] + self.windShift[:,0]
                yy = yf[iStart:iEnd, np.newaxis] + self.windShift[:,1]
                tt,cc,ss = self._CuvNoWind(xx,yy)
                tr[iStart:iEnd] = np.dot(tt,self.windWeight)
                c[iStart:iEnd] = np.dot(cc,self.windWeight)
                s[iStart:iEnd] = np.dot(ss,self.windWeight)
                
 
            out[...,0,0] = (tr - c).reshape(x.shape)
            out[...,1,1] = (tr + c).reshape(x.shape)
            out[...,1,0] = s.reshape(x.shape)
            out[...,0,1] = out[...,1,0]
            
            ## Old way:
            ##for i,w in enumerate(self.windWeight):
            ##    out += w * self._CuvNoWind(x+self.windShift[i,0],
            ##                               y+self.windShift[i,1])
        else:
            ## Old way:
            ##out = self._CuvNoWind(x,y)
            out[...,0,0] = tr - c
            out[...,1,1] = tr + c
            out[...,1,0] = s
            out[...,0,1] = s
            
        out *= self.variance * self.iNorm  # Rescale
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
    
def markScales(turb):
    # Make vertical marks at relevant scales for a TurbulentLayer
    # Do this atop a figure created with plotCuts
    ymin,ymax = pl.ylim()
    pl.ylim(ymin,1.2*turb.variance)
    pl.axhline(0,color='k',lw=2)
    arrowprops = {'arrowstyle':'->','color':'m'}
    pl.annotate('diam',(turb.R*2, turb.variance), xytext=(turb.R*2, 1.15*turb.variance),
                color='m',ha='center', arrowprops=arrowprops)
    r = np.hypot(turb.wind[0],turb.wind[1])
    pl.annotate('wind',(r, turb.variance), xytext=(r, 1.15*turb.variance),
                color='m',ha='center', arrowprops=arrowprops)
    r = 2*np.pi/np.sqrt(turb.k0sq)
    pl.annotate('outer',(r, turb.variance), xytext=(r, 1.15*turb.variance),
                color='m',ha='center', arrowprops=arrowprops)
    return

