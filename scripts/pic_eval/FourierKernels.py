import cupy as cp
import cupyx.scipy.special as sp
from scipy import special


def specKernel(NG, L, dx, dim, order=2):
    Ka = cp.arange(1, NG // 2)
    Kb = Ka[::-1]
    K = cp.append(cp.append(Ka, [- NG // 2]), - Kb)
    K = ((2 * cp.pi) / L[0]) * K
    SHat0 = (cp.sin(K * dx[0] / 2) / (K * dx[0] / 2)) ** order
    SHat0 = cp.append([1], SHat0).reshape(NG, 1)
    if(dim == 1):
        return SHat0.reshape(1, NG)
    else:
        SHat1 = (cp.sin(K * dx[1] / 2) / (K * dx[1] / 2)) ** order
        SHat1 = cp.append([1], SHat1)
        return cp.kron(SHat0, SHat1)

    
def circleKernel(NG, L, order=2):
    r = cp.min(L / NG)
    Ja = cp.arange(0, NG[0] // 2)
    Jb = Ja[:0:-1]
    J = (cp.append(cp.append(Ja, [- NG[0] // 2]), - Jb) * 2 * cp.pi / L[0]) ** 2 * cp.ones([NG[1], 1])
    Ka = cp.arange(0, NG[1] // 2)
    Kb = Ka[:0:-1]
    K = (cp.append(cp.append(Ka, [- NG[1] // 2]), - Kb) * 2 * cp.pi / L[1]) ** 2 * cp.ones([NG[0], 1])
    Kabsolute = cp.transpose(cp.sqrt(J + cp.transpose(K)))
    Kabsolute[0,0] = 1  # avoid 0 on denominator
    SHat = (2 * special.j1(r * Kabsolute) / (r * Kabsolute)) ** order
    SHat[0, 0] = 1
    return SHat

def freeSpaceKernelsPIF(NG, L):
    # Prepare for convolution kernel:
    extension = 4
    wm = cp.linspace(- NG * cp.pi / L[0], NG * cp.pi / L[0], extension*NG, endpoint=False) ## 4 times finer than regular Fourier step
    wm1, wm2 = cp.meshgrid(wm, wm)
    s = cp.sqrt(wm1**2 + wm2**2)
    
    ## Construct mollified Green's function
    LT = 1.5 * L[0] ## Truncation window size
    green = (1-sp.j0(LT*s)) / (s**2) - (LT*cp.log(LT)*sp.j1(LT*s)) / s ## Green function in spectral space
    green[extension*NG//2, extension*NG//2] = (LT**2/4 - LT**2*cp.log(LT)/2)
    
    r = L[0] / NG
    J = cp.fft.fftshift(wm) * cp.ones([NG * extension, 1])
    Kabsolute = cp.transpose(cp.sqrt(J**2 + cp.transpose(J)**2))
    Kabsolute[0,0] = 1  # avoid 0 on denominator
    SHat = (2 * sp.j1(r * Kabsolute) / (r * Kabsolute)) ** 2 
    SHat[0, 0] = 1
    SHat = SHat * (L[0] / NG) ** 2 / (r **2)
    
    green1 = SHat * cp.fft.fftshift(green)
    green2 = SHat ** 2 * cp.fft.fftshift(green)
    
    ## Precomputation
    '''
    For optimal performance use precomputation; for optimal accuracy do not use precomputation.
    '''
    T1 = cp.fft.ifftshift(cp.fft.ifft2(green1))
    T1 = T1[extension*NG//4:extension*NG*3//4, extension*NG//4:extension*NG*3//4]
    T1 = cp.fft.fft2(T1) # This is the kernel for potential field
    
    T2 = cp.fft.ifftshift(cp.fft.ifft2(green2))
    T2 = T2[extension*NG//4:extension*NG*3//4, extension*NG//4:extension*NG*3//4]
    T2 = cp.fft.fft2(T2) # This is the kernel for acceleration

    return SHat,J,T1,T2


def freeSpaceKernelsPIC(NG, L):
    # Prepare for convolution kernel:
    extension = 4
    wm = cp.linspace(- NG * cp.pi / L[0], NG * cp.pi / L[0], extension*NG, endpoint=False) ## 4 times finer than regular Fourier step
    wm1, wm2 = cp.meshgrid(wm, wm)
    s = cp.sqrt(wm1**2 + wm2**2)
    J = cp.fft.fftshift(wm) * cp.ones([NG * extension, 1])
    
    ## Construct mollified Green's function
    LT = 1.5 * L[0] ## Truncation window size
    green = (1-sp.j0(LT*s)) / (s**2) - (LT*cp.log(LT)*sp.j1(LT*s)) / s ## Green function in spectral space
    green[extension*NG//2, extension*NG//2] = (LT**2/4 - LT**2*cp.log(LT)/2)
    
    
    ## Precomputation
    '''
    For optimal performance use precomputation; for optimal accuracy do not use precomputation.
    '''
    T1 = cp.fft.ifftshift(cp.fft.ifft2(cp.fft.fftshift(green)))
    T1 = T1[extension*NG//4:extension*NG*3//4, extension*NG//4:extension*NG*3//4]
    T1 = cp.fft.fft2(T1) # This is the kernel for potential field
    
    return J,T1
