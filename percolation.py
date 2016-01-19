'''
The utilities for calculation of entanglement percolation.

For the algorithms, check PRL 113, 106801
'''

from numpy import *
from numpy.linalg import norm,eigh
from matplotlib.pyplot import *
import pdb

from tba.lattice import meshgrid_v,c2ind,toreciprocal,KSpace
from tba.hgen import sx,sy,sz,Hmesh

class Percolation(object):
    '''
    Percolation problem setting.

    Attributes:
        :L: 1D array, the super-lattice size, the new unit vectors are (L,L) and (L,-L).
        :size:, 1D array, the number of atoms in a unit cell(for each direction).
        :lattice:, <Lattice>, the lattice(only 2D lattice is allowed).
        :form:, str, the sublattice form, 'x' for unit vectors (L,L)/(L,-L) and '#' for unit vectors (L,0)/(0,L)
    '''
    def __init__(self,L,size,lattice,form='x'):
        assert(len(L)==lattice.dimension and len(size)==lattice.dimension)
        self.L=array(L)
        self.lattice=lattice
        assert(form in ['x','#'])
        self.form=form
        self.resize(size)

    @property
    def size(self):
        '''the size'''
        return self._size

    @property
    def nblock(self):
        '''Number of blocks'''
        return self.lattice.N/self.L

    @property
    def xmesh(self):
        '''The lattice indices of x.'''
        nblock=self.nblock
        xlist=[]
        L=self.L
        if self.form=='x':
            ia1=self.L*array([1,1.])
            ia2=self.L*array([1.,-1])
        elif self.form=='#':
            ia1=self.L*array([1.,0])
            ia2=self.L*array([0,1.])
        xmesh=meshgrid_v([arange(nblock[0]),arange(nblock[1])],(ia1,ia2))
        return int32(xmesh)

    def resize(self,size):
        '''resize the A region.'''
        self._size=array(size)
        globaloffset=-(self.size/2)
        globaloffset=zeros(2,dtype='int32')
        self.offsets=meshgrid_v(siteconfig=[arange(si)+offset for offset,si in zip(globaloffset,self.size)],vecs=array([(0,1),(1,0)]))

    def get_block(self,x):
        '''
        Query the block at x.
        
        Parameters:
            :x: 1D array, the size indices at center.

        Return:
            2D array, a list of size indices.
        '''
        return c2ind(x+self.offsets,N=self.lattice.N).ravel()

    def get_kspace(self):
        '''
        Get the new k space.
        '''
        a=self.lattice.a
        L=self.L
        if self.form=='x':
            a=array([L[0]*a[0]+L[1]*a[1],L[0]*a[0]-L[1]*a[1]])
        else:
            a=array([L[0]*a[0],L[1]*a[1]])
        b=toreciprocal(a)
        ks=KSpace(b,N=self.lattice.N/L)
        ks.special_points['X']=array([b[0]/2.,b[1]/2.,-b[0]/2.,-b[1]/2.])
        ks.special_points['M']=array([(b[0]+b[1])/2.,(b[0]-b[1])/2.,(-b[0]-b[1])/2.,(-b[0]+b[1])/2.])
        return ks

    def show_lattice(self):
        '''display the lattice.'''
        self.lattice.show_sites(plane=(0,1),color='r')
        xmesh=self.xmesh
        sitel=[]
        xl=[]
        for i in xrange(xmesh.shape[0]):
            for j in xrange(xmesh.shape[1]):
                x=xmesh[i,j]%self.lattice.N
                sites=self.lattice.sites[self.get_block(x=x)]
                sitel.append(sites)
                xl.append(x)
        sites=concatenate(sitel,axis=0)
        xs=array(xl)
        scatter(sites[:,0],sites[:,1],color='y')
        scatter(xs[:,0],xs[:,1],color='b')
        legend(['B','A','x'])

    def reduce_r(self,C0):
        '''
        Get the reduced density matrix.

        Parameters:
            :C0: ndarray, the density matrix defined in real space.
        '''
        raise NotImplementedError()
        assert(ndim(C0)==4)
        dimention=self.lattice.dimension
        N=self.lattice.N
        nband=C0.shape[-1]
        ncell=prod(self.size)
        offset=self.offsets.reshape([-1,self.offsets.shape[-1]])
        xmesh=self.xmesh
        C=ndarray(list(xmesh.shape[:-1])+[ncell,ncell]+[nband]*2,dtype=complex128)
        for xi in xrange(xmesh.shape[0]):
            for xj in xrange(xmesh.shape[1]):
                x=xmesh[xi,xj]
                for i in xrange(ncell):
                    for j in xrange(ncell):
                        ds=offset[j]-offset[i]  #offset of lattice indices.
                        C[xi,xj,i,j]=C0[(x[0]+ds[0])%N[0],(x[1]+ds[1])%N[1]]
        C=swapaxes(C,3,4).reshape(list(xmesh.shape[:-1])+[ncell*nband]*2)
        return C

    def reduce_k(self,Cfunc,klist):
        '''
        Get the new C+\dag C type operator(like non-interacting Hamiltonian) for specific k.

        Parameters:
            :Cfunc: function, Cfunc(k) gives the expectation matrix at k.
            :klist: ndarray(Nk,vdim), the k-space.

        Return:
            The reduced operator.
        '''
        nband=Cfunc(zeros(2)).shape[-1]
        ncell=prod(self.size)
        offsets=self.offsets.reshape([-1,self.lattice.dimension])
        slist=offsets[:,0,newaxis]*self.lattice.a[0]+offsets[:,1,newaxis]*self.lattice.a[1]
        #get the equivalent k-points.
        b=toreciprocal(self.lattice.a*self.L[:,newaxis])
        Q=(b[0]+b[1])/2.
        equivk=meshgrid_v([arange(l) for l in self.L],vecs=b).reshape([-1,b.shape[-1]])
        if self.form=='x':
            equivk=concatenate([equivk,equivk+Q])

        #calculate C matrix.
        C=ndarray([len(klist),ncell,ncell,nband,nband],dtype=complex128)
        for ik,k in enumerate(klist):
            print '@%s'%ik
            cs=[Cfunc(k+q) for q in equivk]
            for s1 in xrange(ncell):
                for s2 in xrange(ncell):
                    C[ik,s1,s2]=sum([ci*exp(1j*(k+q).dot(slist[s2]-slist[s1])) for ci,q in zip(cs,equivk)],axis=0)/prod(self.L)/(1. if self.form=='#' else 2.)
        return swapaxes(C,2,3).reshape([len(klist),ncell*nband,ncell*nband])
