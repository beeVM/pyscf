#!/usr/bin/env python

'''
EOM-IP-GCCSD for H2O
'''

import numpy as np
from pyscf import lib, gto, scf, cc

mol = gto.Mole()
mol.verbose = 5
mol.unit = 'A'
mol.atom = '''O 0.00000000	 0.00000000	 0.06580905;
              H 0.00000000	 0.75738081	-0.52221858;
              H 0.00000000	-0.75738081	-0.52221858'''
mol.basis = 'cc-pvtz'
mol.spin = 0
mol.build()

# Hartree-Fock
mf = scf.RHF(mol)
mf.max_cycle=100
mf.conv_tol=1e-12
mf.conv_tol_grad=1e-12
mf.verbose = 7
mf.kernel()

#CCSD calculation
mycc = cc.RCCSD(mf)
mycc.verbose = 6
mycc.ccsd()

#IP-EOM-CCSD for 3 root
eip,vip = mycc.ipccsd(nroots=3)

