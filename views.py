from numpy import *
from numpy.testing import dec,assert_,assert_raises,assert_almost_equal,assert_allclose
from matplotlib.pyplot import *
from scipy.linalg import eigh
import pdb,time

from qhall import QuantumHall,sx,sy,sz
from tba.hgen import sx,sy,sz,Hmesh
from tba.lattice import Square_Lattice,path_k
from percolation import *
from utils import *


def get_models(L,size,form,N):
    '''
    Get a quantum hall model.

    Parameters:
        :L: integer, the block spacing.
        :size: integer, the block size
        :N: integer, the number of sites.
        :form: 'x'/'#', choose the block configuration.
    Return:
        tuple of <QuantumHall>, <Percolation>.
    '''
    L=(L,L)
    size=(size,size)
    mu=0.5
    model=QuantumHall(t=0.5,mu=mu,lamb=0.5,N=[50,50])
    perco=Percolation(L,size,model.hgen.rlattice,form=form)
    return model,perco

def showA(size=5):
    '''
    Show the region of A.
    '''
    model,perco=get_models(L=5,size=size,form='x',N=40)
    ion()
    fig=figure(figsize=(9,7))
    perco.show_lattice()
    #for i in xrange(5):
    #    text(i*10,-3,r'$x=%s$'%i,fontsize=14)
    x0=y0=20
    arrow(x0,y0,5,5,width=0.05,color='k',length_includes_head=True)
    arrow(x0,y0,5,-5,width=0.05,color='k',length_includes_head=True)
    text(x0+1,y0+4,r'$R_1$',color='k',fontsize=16,weight='bold')
    text(x0+1,y0-2,r'$R_2$',color='k',fontsize=16,weight='bold')
    axis('equal')
    ylim(-7,52)
    xlim(-2,55)
    pdb.set_trace()
    savefig('lattice.eps')

def perco_fuliang():
    '''
    Get the bulk entanglement spectrum for Fu Liang's work.
    '''
    model,perco=get_models(L=5,size=5,form='x',N=100)
    hgen=model.hgen
    kspace=perco.get_kspace()
    kpath=path_k([kspace.G,kspace.special_points['X'][0],kspace.M[0],kspace.G],N=50)
    ion()
    cfunc=lambda k:EU2C(*eigh(hgen.Hk(k)),T=0.1)
    ck_red=perco.reduce_k(cfunc,kpath)
    hkmesh=Hmesh(C2H(ck_red,T=1.))
    ekmesh=hkmesh.getemesh()
    kpath.plot(ekmesh,mode='abs')
    ylim(-5,5)
    pdb.set_trace()

def show_bzone(size=5):
    '''
    Show the old and new brillouin zone.
    '''
    model,perco=get_models(L=5,size=size,form='x',N=40)
    ks0=model.hgen.rlattice.kspace
    ks1=perco.get_kspace()
    ion()
    #bzone0=concatenate([ks0.K,ks0.K[:1]],axis=0)
    #bzone1=concatenate([ks1.special_points['M'],ks1.special_points['M'][:1]],axis=0)
    for ksi in [ks0,ks1]:
        bzone=array([zeros(2),ksi.b[0],ksi.b[0]+ksi.b[1],ksi.b[1],zeros(2)])
        plot(bzone[:,0],bzone[:,1],color='k')
        token='b' if ksi is ks0 else r'\tilde{b}'
        offset=[0.1,0.1] if ksi is ks0 else [0.25,-0.15]
        for ib in xrange(2):
            x0=y0=0
            arrow(x0,y0,ksi.b[ib,0],ksi.b[ib,1],width=0.008,color='b',length_includes_head=True)
            text(x0+ksi.b[ib,0]/2.+offset[0],y0+ksi.b[ib,1]/2.+offset[1],r'$%s_%s$'%(token,ib),color='b',fontsize=16,weight='bold')
    #get the equivalent k-points.
    b=toreciprocal(perco.lattice.a*perco.L[:,newaxis])
    equivk=meshgrid_v([arange(l) for l in perco.L],vecs=b).reshape([-1,b.shape[-1]])
    if perco.form=='x':
        Q=(b[0]+b[1])/2.
        equivk=concatenate([equivk,equivk+Q])
    scatter(equivk[:,0],equivk[:,1],s=20,color='r')

    axis('equal')
    pdb.set_trace()
    savefig('bzone.eps')


