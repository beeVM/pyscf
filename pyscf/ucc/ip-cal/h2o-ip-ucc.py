#!/usr/bin/env python

'''
Ground-state, EOM-EE-GCCSD for HF
'''

import numpy as np
from pyscf import lib, gto, scf, cc
from pyscf.ucc import ucc, ucc_ppt

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
#mf.chkfile='h2o.chk'
mf.kernel()

# Read MO information from h2o.chk
#mol = lib.chkfile.load_mol('h2o.chk')
#mf = scf.RHF(mol)
#mf.__dict__.update(lib.chkfile.load('h2o.chk','scf'))

# transform RHF to GHF
nmf = scf.addons.convert_to_ghf(mf)
#
# UGCCSD
mycc = ucc.UnitaryGCCSD(nmf)#,frozen=frozen)
mycc.verbose = 5
mycc.max_cycle=100
mycc.conv_tol=1e-8
mycc.conv_tol_normt=1e-8
#mycc.diis_file = 'h2o_gquccsd_diis.h5'
ecc, t1, t2 = mycc.kernel()

# Restore UCC ground state calculation
#mycc.diis = lib.diis.restore('h2o_gquccsd_diis.h5')
#uccvec = mycc.diis.extrapolate()
#t1, t2 = mycc.vector_to_amplitudes(uccvec)
#ecc, t1, t2 = mycc.kernel(t1,t2)

# IP-UCC
mycc.max_cycle=100
mycc.conv_tol=1e-5
mycc.conv_tol_normt=1e-5
mycc.verbose = 5
e,v = mycc.ipccsd(nroots=8)

for i in e: 
    print("%f" %(i*27.2113819))

