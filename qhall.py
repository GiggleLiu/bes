'''
Calculation of bulk entanglement spectrum for quantum Hall insulator.
'''

from numpy import *
import pdb,time

from tba.hgen import KHGenerator,SpaceConfig,op_simple_hopping,op_simple_onsite,op_from_mats,sx,sy,sz
from tba.lattice import Square_Lattice

class QuantumHall(object):
    '''
    The model of Quantum Hall insulator.
    The hamiltonian is, H = t*(cos kx + cos ky - mu)*sz + lamb*(sin kx*sx + sin ky*sy).
    
    Attributes:
        :t: float, strength of the hopping term.
        :lamb: float, strength of the SOC term.
        :mu: float, the chemical potential.
    '''
    def __init__(self,t,lamb,mu):
        self.t=t
        self.lamb=lamb
        self.mu=mu
        spaceconfig=SpaceConfig([1,2,1,1],kspace=True)
        hgen=KHGenerator(spaceconfig,propergauge=False)

        #define the operator of the system
        hgen.register_params({
            't1':self.t,
            '-mu':-self.mu,
            '-ilamb':-1j*self.lamb,
            })

        #define a structure and initialize bonds.
        rlattice=Square_Lattice(N=(100,100),catoms=[zeros(2)])
        hgen.uselattice(rlattice)

        b1s=rlattice.cbonds[1]  #the nearest neighbor
        b0s=rlattice.cbonds[0]  #the onsite term.

        #add the hopping term.
        op_t1=op_from_mats(label='hop1',spaceconfig=spaceconfig,mats=[sz]*len(b1s),bonds=b1s)
        hgen.register_operator(op_t1,param='t1')
        op_mu=op_from_mats(label='n',spaceconfig=spaceconfig,mats=[sz])
        hgen.register_operator(op_mu,param='-mu')
        bondx=b1s.query(bondv=array([1,0]))
        _bondx=b1s.query(bondv=-array([1,0]))
        bondy=b1s.query(bondv=array([0,1]))
        _bondy=b1s.query(bondv=-array([0,1]))
        op_l=op_from_mats(label='lamb',spaceconfig=spaceconfig,mats=[sx,-sx,sy,-sy],bonds=[bondx[0],_bondx[0],bondy[0],_bondy[0]])
        hgen.register_operator(op_l,param='-ilamb')

        self.hgen=hgen

