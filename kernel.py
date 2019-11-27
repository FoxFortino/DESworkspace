import numpy as np
import matplotlib.pyplot as pl

def curl_free_kernel(theta, X1, X2):
    """
    This function generates the full (2N, 2N) covariance matrix.
    
    Parameters:
        theta (dict): Dictionary of the kernel parameters.
        X1 (ndarray): (N, 2) 2d array of astrometric positions (u, v).
        X2 (ndarray): (N, 2) 2d array of astrometric positions (u, v).
        
    Returns:
        K (ndarray): (N, N) 2d array, the kernel.

    XXX Note for Prof:
    self.Xunit is degrees
    self.Yunit is mas

    self.fix_params takes the dictionary of parameters and makes sure that its a dictionary
    """
    
    sigma_s = theta['sigma_s'] 
    sigma_x = theta['sigma_x'] 
    sigma_y = theta['sigma_y'] 
    phi = theta['phi'] 
    
    A = (4 * sigma_s**2 * sigma_x**5 * sigma_y**5) / np.pi
    
    u1, u2 = X1[:, 0], X2[:, 0]
    v1, v2 = X1[:, 1], X2[:, 1]
    
    uu1, uu2 = np.meshgrid(u1, u2)
    vv1, vv2 = np.meshgrid(v1, v2)
    
    du = (uu1 - uu2)
    dv = (vv1 - vv2)
    
    coeff = np.pi * A / (4 * sigma_x**5 * sigma_y**5)
    
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

def gary_kernel(theta, X1, X2):
    '''
    This is a drop-in replacement for the above routine, where I
    have dropped the offset term, and do all the calculations
    with linear algebra routines once the inverse cov matrix of
    the exponential is created.
    '''
    
    sigma_s = theta['sigma_s'] 
    sigma_x = theta['sigma_x'] 
    sigma_y = theta['sigma_y'] 
    phi = theta['phi']
    
    #Construct elements of the inverse covariance matrix
    detC = (sigma_x * sigma_y)**2
    a = 0.5*(sigma_x**2 + sigma_y**2)
    b = 0.5*(sigma_x**2 - sigma_y**2)
    b1 = b * np.cos(2*phi)
    b2 = -b * np.sin(2*phi)
    cInv = np.array( [ [a-b1, -b2],[-b2, a+b1]]) / detC

    dX = X1[:,np.newaxis,:] - X2[np.newaxis,:,:]  # Array is N1 x N2 x 2

    cInvX = np.einsum('kl,ijl',cInv,dX)  # Another N1 x N2 x 2
    
    exponentialFactor = np.exp(-0.5*np.sum(cInvX*dX,axis=2))
    # Multiply the overall prefactor into this scalar array
    exponentialFactor *= sigma_s**2/np.trace(cInv)

    # Start building the master answer, as (N1,N2,2,2) array
    k = np.ones( dX.shape + (2,)) * cInv  # Start with cInv term
    # Now subtract the outer product of cInvX
    k -= cInvX[:,:,:,np.newaxis]*cInvX[:,:,np.newaxis,:]
    
    # And the exponential
    k = k * exponentialFactor[:,:,np.newaxis,np.newaxis]

    # change (N1,N2,2,2) to (2*N1,2*N2) array
    k = np.moveaxis(k,2,1)
    s = k.shape
    k = k.reshape(s[0]*2,s[2]*2)

    return k

    
'''
Willow is 1.6e8 larger for sig(x,y) = 20,10.
This normalization error looks to be equal to 8 * Tr(C) * det(C) 
where C is the covariance matrix of the exponential, or
8 * (sigx^2+sigy^2) * sigx^2*sigy*2.

Other than this, *trace* Kuu+Kvv agree well but Kuu, Kvv, and Kuv
each have errors compared to mine that seem to scale with
something like sin(2*theta - 2*phi) * sin(2*phi)
where theta is the angle of (u,v).
'''

def plotit(img):
    pl.imshow(img.transpose(), cmap='Spectral', interpolation='nearest', origin='lower')
    pl.colorbar()
    return

def makepts(size=64):
    u = np.arange(-size,size,dtype=float)
    v = -np.arange(-size,size,dtype=float)
    z = np.zeros_like(u)

    x1 = np.vstack( (u,z)).transpose()
    x2 = np.vstack( (z,v)).transpose()
    return x1,x2

    
def gary_discrete(theta,size=64):
    '''
    This version calculates the Kuv by taking a finite-difference
    second derivative of the exponential term, as a way to
    check which of the two versions of the algebra above is
    the correct one.
    '''

    sigma_s = theta['sigma_s'] 
    sigma_x = theta['sigma_x'] 
    sigma_y = theta['sigma_y'] 
    phi = theta['phi']
    
    #Construct elements of the inverse covariance matrix
    detC = (sigma_x * sigma_y)**2
    a = 0.5*(sigma_x**2 + sigma_y**2)
    b = 0.5*(sigma_x**2 - sigma_y**2)
    b1 = b * np.cos(2*phi)
    b2 = -b * np.sin(2*phi)
    cInv = np.array( [ [a-b1, -b2],[-b2, a+b1]]) / detC

    u = np.arange(-size,size,dtype=float)
    v = -np.arange(-size,size,dtype=float)
    z = np.zeros_like(u)

    x1 = np.vstack( (u,z)).transpose()
    x2 = np.vstack( (z,v)).transpose()
    dX = x1[:,np.newaxis,:] - x2[np.newaxis,:,:]  # Array is N1 x N2 x 2

    cInvX = np.einsum('kl,ijl',cInv,dX)  # Another N1 x N2 x 2
    
    s = np.exp(-0.5*np.sum(cInvX*dX,axis=2))
    # Multiply the overall prefactor into this scalar array
    s *= sigma_s**2/np.trace(cInv)

    k = np.zeros( dX.shape + (2,)) # Start with cInv term
    k[1:-1,:,0,0] = s[0:-2,:] + s[2:,:] - 2*s[1:-1,:]
    k[:,1:-1,1,1] = s[:,0:-2] + s[:,2:] - 2*s[:,1:-1]
    k[1:-1,1:-1,0,1] = (s[0:-2,0:-2] + s[2:,2:] - s[0:-2,2:]  - s[2:,0:-2])/4.

    return -k
