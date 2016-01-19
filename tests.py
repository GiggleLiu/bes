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

class ModelTest():
    def __init__(self):
        t=random.random()
        lamb=random.random()
        mu=random.random()
        self.model=QuantumHall(t=t,mu=mu,lamb=lamb)

    def test_Hk(self):
        '''the hamiltonian.'''
        model=self.model
        hgen=model.hgen
        for i in xrange(100):
            k=pi-random.random(2)*2*pi
            kx,ky=k
            hk1=2*model.t*(cos(kx)+cos(ky))*sz-model.mu*sz+2*model.lamb*(sin(kx)*sx+sin(ky)*sy)
            hk2=hgen.Hk(k)
            assert_allclose(hk1,hk2)

    def test_all(self):
        self.test_Hk()

class ExpTest():
    def test_fermi(self):
        #for zero temperature
        elist=[-1,0,1]
        for T in [-1,0,1]:
            if T<0:
                assert_raises(ValueError,fermi,elist=elist,T=T)
            else:
                fl=fermi(elist=elist,T=T)
                for i,e in enumerate(elist):
                    assert_almost_equal(fermi(e,T=T),fl[i])
                    if e==0:
                        assert_(fl[i]==0.5)
                    elif e>0:
                        assert_(fl[i]<0.5)
                    else:
                        assert_(fl[i]>0.5)

    def test_exp_real(self):
        '''
        test for two site model H = t*c1^\dag c2 + t*c2^\dag c1
        '''
        t=random.random()
        Tlist=[0,0.5]
        H=t*sx
        E,U=eigh(H)
        for T in Tlist:
            mval=Fij(E=E,U=U,T=T)
            assert_allclose(mval.diagonal(),0.5*ones(len(mval)))

    def test_exp_k(self):
        t=random.random()
        Tlist=[0,0.5]
        Hk=lambda k:2*t*cos(k)*identity(2)
        klist=arange(1000)*2*pi/1000
        hmesh=Hmesh([Hk(k) for k in klist])
        E,U=hmesh.getemesh(evalvk=True)
        for T in Tlist:
            mval=rFij(E=E,U=U,T=T)
            assert_allclose(mval[0].diagonal(),0.5*ones(mval.shape[1]))

    def test_all(self):
        self.test_fermi()
        self.test_exp_real()
        self.test_exp_k()


class PercoTest(object):
    def __init__(self):
        #lattice=Square_Lattice(N=(100,100),catoms=[zeros(2)])
        mu=0.5
        self.model=QuantumHall(t=0.5,mu=mu,lamb=0.5,N=[50,50])

    def get_perco(self,L,size,form):
        '''
        get a percolation instance.
        '''
        L=(L,L)
        size=(size,size)
        perco=Percolation(L,size,self.model.hgen.rlattice,form=form)
        return perco

    def test_showA(self):
        perco=self.get_perco(L=5,size=5,form='x')
        ion()
        subplot(121)
        perco.show_lattice()
        axis('equal')
        subplot(122)
        perco.resize((3,3))
        perco.show_lattice()
        axis('equal')
        pdb.set_trace()

    def test_symm(self,useC=True):
        perco=self.get_perco(L=5,size=5,form='x')
        hgen=self.model.hgen
        kspace=perco.get_kspace()
        kpath=path_k([kspace.G,kspace.special_points['X'][0],kspace.M[0],kspace.G],N=40)
        ion()
        if useC:
            cfunc=lambda k:EU2C(*eigh(hgen.Hk(k)),T=0.1)
            ck_red=perco.reduce_k(cfunc,kpath)
            hkmesh=Hmesh(C2H(ck_red,T=1.))
        else:
            hfunc=lambda k:hgen.Hk(k)
            hk_red=perco.reduce_k(hfunc,kpath)
            hkmesh=Hmesh(hk_red)
        ekmesh=hkmesh.getemesh()
        kpath.plot(ekmesh,mode='abs',ls='--' if useC else '-')
        ylim(-5,5)
        pdb.set_trace()

    def test_bandrecover(self,useC=True):
        perco=self.get_perco(L=1,size=1,form='#')
        T=1.
        hgen=self.model.hgen
        kspace=perco.get_kspace()
        kpath=path_k([kspace.G,kspace.special_points['X'][0],kspace.M[0],kspace.G],N=50)
        if useC:
            cfunc=lambda k:EU2C(*eigh(hgen.Hk(k)),T=T)
            ck_red=perco.reduce_k(cfunc,kpath)
            hkmesh=Hmesh(C2H(ck_red))
        else:
            hfunc=lambda k:hgen.Hk(k)
            hk_red=perco.reduce_k(hfunc,kpath)
            hkmesh=Hmesh(hk_red)
        ekmesh=hkmesh.getemesh()
        ekmesh0=Hmesh([hgen.Hk(k) for k in kpath]).getemesh()
        #kpath.plot(ekmesh,mode='abs')
        #kpath.plot(ekmesh0,mode='abs',color='r',ls='--')
        assert_allclose(ekmesh,ekmesh0,atol=1e-4)

    def test_all(self):
        for useC in [True,False]:
            self.test_bandrecover(useC=useC)
            self.test_symm(useC=useC)
        self.test_showA()

#ModelTest().test_all()
#ExpTest().test_all()
#PercoTest().test_all()
PercoTest().test_showA()
PercoTest().test_symm(useC=True)
