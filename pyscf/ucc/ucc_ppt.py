#!/usr/bin/env python
# Copyright 2014-2021 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import numpy as np

from pyscf import lib
from pyscf.lib import logger, module_method
from pyscf.cc import ccsd
from pyscf.cc import eom_rccsd
from pyscf.cc import gccsd
from pyscf import ucc
from pyscf.ucc import gintermediates as imds

einsum = lib.einsum

def kernel(eom,nroots=1,guess=None, eris=None, verbose=None):

    #UnitaryGCCSD.method = UnitaryGCCSD.method.lower()
    #if UnitaryGCCSD.method not in ("UCC3", "qUCCSD"):
    #    raise NotImplementedError(UnitaryGCCSD.method)

    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(adc.stdout, adc.verbose)
    if adc.verbose >= logger.WARN:
        adc.check_sanity()
    adc.dump_flags()

    if eris is None:
        eris = eom._PhysicistERIs()

    imds = eom.get_imds(eris)
    matvec, diag = eom.gen_matvec(imds, eris)

    guess = eom.get_init_guess(nroots, diag, ascending=True)

    conv, e, vec = lib.linalg_helper.davidson_nosym1(
        lambda xs : [matvec(x) for x in xs],
        guess, diag, nroots=nroots, verbose=log, tol=eom.conv_tol,
        max_cycle=eom.max_cycle, max_space=eom.max_space, tol_residual=eom.tol_residual)

    header = ("\n*************************************************************"
              "\n            UCC-PPT calculation summary"
              "\n*************************************************************")
    logger.info(eom, header)

    for n in range(nroots):
        print_string = ('%s root %d  |  Energy (Eh) = %14.10f  |  Energy (eV) = %12.8f  ' %
                        (eom.method, n, e[n], e[n]*27.2114))
        print_string += ("|  conv = %s" % conv[n])
        logger.info(eom, print_string)

    log.timer('UCC', *cput0)

    return conv, e.real, vec


########################################
# IP-UCC-EPT
########################################

def vector_to_amplitudes_ip(vector, nmo, nocc):
    nvir = nmo - nocc
    r1 = vector[:nocc].copy()
    r2 = np.zeros((nocc,nocc,nvir), dtype=vector.dtype)
    idx, idy = np.tril_indices(nocc, -1)
    r2[idx,idy] = vector[nocc:].reshape(nocc*(nocc-1)//2,nvir)
    r2[idy,idx] =-vector[nocc:].reshape(nocc*(nocc-1)//2,nvir)
    return r1, r2

def amplitudes_to_vector_ip(r1, r2):
    nocc = r1.size
    return np.hstack((r1, r2[np.tril_indices(nocc, -1)].ravel()))

def ipccsd_matvec(eom, vector, imds=None, diag=None):
    if imds is None: imds = eom.make_imds()
    nocc = eom.nocc
    nmo = eom.nmo
    r1, r2 = eom.vector_to_amplitudes(vector, nmo, nocc)

    # 1h-1h block
    Hr1 = -einsum('mi,m->i', imds.Hoo, r1)
    # 1h-2h1p block
    Hooov = np.asarray(imds.Hovoo).transpose(2,3,0,1)
    Hr1 += -0.5*einsum('nmie,mne->i', Hooov.conj(), r2)
    # 2h1p-1h block
    Hr2 =  -einsum('kbij,k->jib', imds.Hovoo, r1)
    # 2h1p-2h1p block
    foo = imds.eris.fock[:nocc,:nocc]
    fvv = imds.eris.fock[nocc:,nocc:]
    Hr2 +=  einsum('ae,ije->ija', fvv, r2)
    tmp1 = -einsum('mi,mja->ija', foo, r2)
    Hr2 += tmp1 - tmp1.transpose(1,0,2)
    Hr2 += 0.5*einsum('mnij,mna->ija', imds.Hoooo, r2)
    tmp2 = einsum('maei,mje->ija', imds.eris.ovvo, r2)
    Hr2 += tmp2 - tmp2.transpose(1,0,2)

    vector = eom.amplitudes_to_vector(Hr1, Hr2)
    return vector

def ipccsd_diag(eom, imds=None):
    if imds is None: imds = eom.make_imds()
    t1, t2 = imds.t1, imds.t2
    nocc, nvir = t1.shape

    Hr1 = -np.diag(imds.Hoo)
    Hr2 = np.zeros((nocc,nocc,nvir), dtype=t1.dtype)
    for i in range(nocc):
        for j in range(nocc):
            for a in range(nvir):
                Hr2[i,j,a] += imds.Hvv[a,a]
                Hr2[i,j,a] += -imds.Hoo[i,i]
                Hr2[i,j,a] += -imds.Hoo[j,j]
                Hr2[i,j,a] += 0.5*(imds.Hoooo[i,j,i,j]-imds.Hoooo[j,i,i,j])
                Hr2[i,j,a] += imds.Hovvo[i,a,a,i]
                Hr2[i,j,a] += imds.Hovvo[j,a,a,j]
                Hr2[i,j,a] += 0.5*(np.dot(imds.Hoovv[i,j,:,a], t2[i,j,a,:]) -
                                   np.dot(imds.Hoovv[j,i,:,a], t2[i,j,a,:]))

    vector = eom.amplitudes_to_vector(Hr1, Hr2)
    return vector


def ipccsd(eom, nroots=1, koopmans=False, guess=None, eris=None, imds=None):
    '''Calculate (N-1)-electron charged excitations via IP-EOM-CCSD.

    Kwargs:
        nroots : int
            Number of roots (eigenvalues) requested
        koopmans : bool
            Calculate Koopmans'-like (quasiparticle) excitations only, targeting via
            overlap.
        guess : list of ndarray
            List of guess vectors to use for targeting via overlap.
    '''
    eom.converged, eom.e, eom.v \
            = eom_rccsd.kernel(eom, nroots, koopmans, guess, eris=eris, imds=imds)
    return eom.e, eom.v

class EOMIP(eom_rccsd.EOMIP):
    kernel = ipccsd
    ipccsd = ipccsd
    matvec = ipccsd_matvec
    get_diag = ipccsd_diag

    amplitudes_to_vector = staticmethod(amplitudes_to_vector_ip)
    vector_to_amplitudes = module_method(vector_to_amplitudes_ip,
                                         absences=['nmo', 'nocc'])

    def vector_size(self):
        nocc = self.nocc
        nvir = self.nmo - nocc
        return nocc + nocc*(nocc-1)//2*nvir

    def make_imds(self, eris=None):
        imds = _IMDS(self._cc, eris)
        imds.make_ip()
        return imds


########################################
# EE-UCC-PPT
########################################

vector_to_amplitudes_ee = ccsd.vector_to_amplitudes_s4
amplitudes_to_vector_ee = ccsd.amplitudes_to_vector_s4

def eeccsd_matvec(eom, vector, imds=None, diag=None):
    # Ref: J. Liu, A. Asthana, L. Cheng, D. Mukherjee, J. Chem. Phys. 148, 244110(2018). 
    # Ref: J. Liu, L. Cheng. J. Chem. Phys. 155, 174102(2021). 
    if imds is None: imds = eom.make_imds()
    nocc = eom.nocc
    nmo = eom.nmo
    r1, r2 = eom.vector_to_amplitudes(vector, nmo, nocc)

    # Hbar_SS
    Hr1  = einsum('ab,ib->ia', imds.Hvv, r1)
    Hr1 -= einsum('ji,ja->ia', imds.Hoo, r1)
    Hr1 += einsum('jabi,jb->ia', imds.Hovvo, r1)
    # Hbar_SD
    Hr1 -= 0.5*einsum('iemn,mnae->ia', imds.Hovoo.conj(), r2)
    Hr1 += 0.5*einsum('feam,imfe->ia', imds.Hvvvo.conj(), r2)
    #three body term
    Iij = 0.5*einsum('ikab,jkab->ij',imds.t2.conj(),r2)  
    eris_ovoo = np.asarray(imds.eris.ooov).transpose(2,3,0,1)
    Hr1 -= einsum('kj,jaki->ia',Iij,eris_ovoo.conj())
    Iab = -0.5*einsum('ijac,ijbc->ab',r2,imds.t2.conj())
    eris_vvvo = np.asarray(imds.eris.ovvv).transpose(3,2,1,0)
    Hr1 -= einsum('cb,baci->ia',Iab,eris_vvvo.conj())
    # Hbar_DS
    tmpab = -einsum('kaji,kb->ijab',imds.Hovoo,r1)
    tmpij = einsum('abcj,ic->ijab',imds.Hvvvo,r1) 
    #three body term
    Iij = einsum('ikja,ka->ij',imds.eris.ooov,r1)
    tmpij -= einsum('kj,ikab->ijab',Iij,imds.t2)
    eris_vovv = np.asarray(imds.eris.ovvv).transpose(1,0,3,2)
    Iab = einsum('aibc,ic->ab',eris_vovv.conj(),r1)
    tmpab += einsum('bc,ijac->ijab',Iab,imds.t2)
    Hr2 = tmpab - tmpab.transpose(0,1,3,2)
    Hr2 += tmpij - tmpij.transpose(1,0,2,3)
    #Hbar_DD
    nocc, nvir = imds.t1.shape
    foo = imds.eris.fock[:nocc,:nocc]
    fvv = imds.eris.fock[nocc:,nocc:]
    tmpab = einsum('be,ijae->ijab',fvv,r2)
    Hr2 += tmpab - tmpab.transpose(0,1,3,2)
    tmpij = -einsum('kj,ikab->ijab',foo,r2)
    Hr2 += tmpij - tmpij.transpose(1,0,2,3)
    tmpijab = einsum('mbej,imae->ijab', imds.eris.ovvo, r2)
    tmpijab = tmpijab - tmpijab.transpose(1,0,2,3)
    tmpijab = tmpijab - tmpijab.transpose(0,1,3,2)
    Hr2 += tmpijab
    Hr2 += 0.5*einsum('mnij,mnab->ijab', imds.Hoooo, r2)
    Hr2 += 0.5*einsum('abef,ijef->ijab', imds.Hvvvv, r2)

    vector = eom.amplitudes_to_vector(Hr1, Hr2)
    return vector

def eeccsd_diag(eom, imds=None):
    if imds is None: imds = eom.make_imds()
    t1, t2 = imds.t1, imds.t2
    nocc, nvir = t1.shape

    Hr1 = np.zeros((nocc,nvir), dtype=t1.dtype)
    Hr2 = np.zeros((nocc,nocc,nvir,nvir), dtype=t1.dtype)
    for i in range(nocc):
        for a in range(nvir):
            Hr1[i,a] = imds.Hvv[a,a] - imds.Hoo[i,i] + imds.Hovvo[i,a,a,i]
    for a in range(nvir):
        tmp = 0.5*(np.einsum('ijeb,ijbe->ijb', imds.Hoovv, t2) -
                   np.einsum('jieb,ijbe->ijb', imds.Hoovv, t2))
        Hr2[:,:,:,a] += imds.Hvv[a,a] + tmp
        Hr2[:,:,a,:] += imds.Hvv[a,a] + tmp
        _Hvvvva = np.array(imds.Hvvvv[a])
        for b in range(a):
            Hr2[:,:,a,b] += 0.5*(_Hvvvva[b,a,b]-_Hvvvva[b,b,a])
        for i in range(nocc):
            tmp = imds.Hovvo[i,a,a,i]
            Hr2[:,i,:,a] += tmp
            Hr2[i,:,:,a] += tmp
            Hr2[:,i,a,:] += tmp
            Hr2[i,:,a,:] += tmp
    for i in range(nocc):
        tmp = 0.5*(np.einsum('kjab,jkab->jab', imds.Hoovv, t2) -
                   np.einsum('kjba,jkab->jab', imds.Hoovv, t2))
        Hr2[:,i,:,:] += -imds.Hoo[i,i] + tmp
        Hr2[i,:,:,:] += -imds.Hoo[i,i] + tmp
        for j in range(i):
            Hr2[i,j,:,:] += 0.5*(imds.Hoooo[i,j,i,j]-imds.Hoooo[j,i,i,j])

    vector = eom.amplitudes_to_vector(Hr1, Hr2)
    return vector


def eeccsd(eom, nroots=1, koopmans=False, guess=None, eris=None, imds=None):
    '''Calculate N-electron neutral excitations via EOM-EE-CCSD.

    Kwargs:
        nroots : int
            Number of roots (eigenvalues) requested
        koopmans : bool
            Calculate Koopmans'-like (1p1h) excitations only, targeting via
            overlap.
        guess : list of ndarray
            List of guess vectors to use for targeting via overlap.
    '''
    #return eom_rccsd.eomee_ccsd_singlet(eom, nroots, koopmans, guess, eris, imds)
    eom.converged, eom.e, eom.v \
            = eom_rccsd.kernel(eom, nroots, koopmans, guess, eris=eris,
                    imds=imds, diag=None)
    return eom.e, eom.v


class EOMEE(eom_rccsd.EOMEE):

    kernel = eeccsd
    eeccsd = eeccsd
    matvec = eeccsd_matvec
    get_diag = eeccsd_diag

    def gen_matvec(self, imds=None, **kwargs):
        if imds is None: imds = self.make_imds()
        diag = self.get_diag(imds)
        matvec = lambda xs: [self.matvec(x, imds) for x in xs]
        return matvec, diag

    amplitudes_to_vector = staticmethod(amplitudes_to_vector_ee)
    vector_to_amplitudes = module_method(vector_to_amplitudes_ee,
                                         absences=['nmo', 'nocc'])

    def vector_size(self):
        '''size of the vector based on spin-orbital basis'''
        nocc = self.nocc
        nvir = self.nmo - nocc
        return nocc*nvir + nocc*(nocc-1)//2*nvir*(nvir-1)//2

    def make_imds(self, eris=None):
        imds = _IMDS(self._cc, eris)
        imds.make_ee()
        return imds

gccsd.GCCSD.EOMIP = lib.class_as_method(EOMIP)
gccsd.GCCSD.EOMEE = lib.class_as_method(EOMEE)

class _IMDS:
    # Exactly the same as RCCSD IMDS except
    # -- rintermediates --> gintermediates
    # -- Loo, Lvv, cc_Fov --> Foo, Fvv, Fov
    # -- One less 2-virtual intermediate
    def __init__(self, cc, eris=None):
        self.verbose = cc.verbose
        self.stdout = cc.stdout
        self.t1 = cc.t1
        self.t2 = cc.t2
        if eris is None:
            eris = cc.ao2mo()
        self.eris = eris
        self._made_shared = False
        self.made_ip_imds = False
        self.made_ea_imds = False
        self.made_ee_imds = False

    def _make_shared(self):
        cput0 = (logger.process_clock(), logger.perf_counter())

        t1, t2, eris = self.t1, self.t2, self.eris
        self.Hoo = imds.Hoo(t1, t2, eris)
        self.Hvv = imds.Hvv(t1, t2, eris)

        # 2 virtuals
        self.Hovvo = imds.Hovvo(t1, t2, eris)
        self.Hoovv = eris.oovv

        self._made_shared = True
        logger.timer_debug1(self, 'UCC-EPT/PPT shared intermediates', *cput0)
        return self

    def make_ip(self):
        if not self._made_shared:
            self._make_shared()

        cput0 = (logger.process_clock(), logger.perf_counter())

        t1, t2, eris = self.t1, self.t2, self.eris

        # 0 or 1 virtuals
        self.Hvv = imds.Hvv(t1, t2, eris)
        self.Hoooo = imds.Hoooo(t1, t2, eris)
        self.Hovoo = imds.Hovoo(t1, t2, eris)

        self.made_ip_imds = True
        logger.timer_debug1(self, 'UCC-EPT IP intermediates', *cput0)
        return self

    def make_ee(self):
        if not self._made_shared:
            self._make_shared()

        cput0 = (logger.process_clock(), logger.perf_counter())

        t1, t2, eris = self.t1, self.t2, self.eris

        self.Hoooo = imds.Hoooo(t1, t2, eris)
        self.Hovoo = imds.Hovoo(t1, t2, eris)
        self.Hvvvv = imds.Hvvvv(t1, t2, eris)
        self.Hvvvo = imds.Hvvvo(t1, t2, eris)

        self.made_ee_imds = True
        logger.timer(self, 'UCC-PPT shared intermediates', *cput0)
        return self
