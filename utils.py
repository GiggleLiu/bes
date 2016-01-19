'''
Utilities.
'''

from numpy import *
from numpy.linalg import norm,eigh

from rglib.mps.utils import bcast_dot

__all__=['fermi','C2H','EU2C','tokspace','torspace']

def fermi(elist,T=0):
    '''
    Fermi statistics, python implimentation.

    Parameters:
        :elist: float/ndarray, the energy.
        :T: float, the temperature.

    Return:
        float/ndarray, Fermionic disctribution.
    '''
    elist=asarray(elist)
    if T<0.:
        raise ValueError('Negative temperature is not allowed!')
    elif T==0:
        if ndim(elist)!=0:
            f=zeros(elist.shape,dtype='float64')
            f[elist<0]=1.
            f[elist==0]=0.5
            return f
        else:
            if elist>0:
                return 0.
            elif elist==0:
                return 0.5
            else:
                return 1.
    else:
        f=1./(1.+exp(-abs(elist)/T))
        if ndim(elist)!=0:
            posmask=elist>0
            f[posmask]=1.-f[posmask]
        elif elist>0:
            f=1.-f
        return f

def tokspace(mesh,axes):
    '''
    Transform a mesh defined on real-space to k-space.

    Parameters:
        :mesh: ndarray, the input mesh.
        :axes: tuple/list, the axes to perform fourier transition.

    Return:
        The transformed mesh on dk.
    '''
    kmesh=fft.fftn(mesh,axes=axes)
    return kmesh

def torspace(mesh,axes):
    '''
    Transform a mesh defined on k-space to real-space.

    Parameters:
        :mesh: ndarray, the input mesh.
        :axes: tuple/list, the axes to perform fourier transition.

    Return:
        The transformed mesh on dr.
    '''
    rmesh=fft.ifftn(mesh,axes=axes)
    return rmesh

def EU2C(E,U,T=0):
    '''
    Get the expectation matrix in k-space.

    Parameters:
        :E,U: ndarray, the eigenvalues and eigenvectors defined on real space.
        ndim(E)>=2 and ndim(U)=ndim(E)+1.

    Return:
        ndarray, the expectation matrix(the expectation mesh of <ck^\dag,ck>)
    '''
    assert(ndim(E)>=1 and ndim(U)==ndim(E)+1)
    fm=fermi(E,T=T)
    C=bcast_dot((U.conj()*fm[...,newaxis,:]),swapaxes(U,-1,-2))
    return C

def C2H(C,T=1.):
    '''
    Get the entanglement hanmiltonian from expectation matrix.

    Parameters:
        :C: ndarray, the expectation matrix.
        :T: float, the temperature.

    Return:
        ndarray, the hamiltonian.
    '''
    CE,CU=eigh(C)
    print 'Checking for the range fermionic occupation numbers, min -> %s, max -> %s.'%(CE.min(),CE.max())
    assert(all(CE>0) and all(CE<1))
    H=bcast_dot(CU.conj()*(log(1./CE-1)*T)[...,newaxis,:],swapaxes(CU,-1,-2))
    return H


