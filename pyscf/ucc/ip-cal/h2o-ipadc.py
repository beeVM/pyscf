#!/usr/bin/env python

'''
IP-RADC calculations for closed-shell H2O 
'''

from pyscf import gto, scf, adc

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
#mf.chkfile='co.chk'
mf.kernel()

myadc = adc.ADC(mf)

#IP-RADC(3) for 1 root
myadc.verbose = 6
myadc.method = "adc(3)"
eip,vip,pip,xip = myadc.kernel(nroots=3)

