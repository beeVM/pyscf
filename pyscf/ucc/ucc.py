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

# Author: Junzi Liu <latrix1247@gmail.com>

import numpy as np
from functools import reduce

from pyscf import lib
from pyscf import ao2mo
from pyscf import scf
from pyscf.lib import logger
from pyscf.cc import ccsd
from pyscf.cc import addons
from pyscf.ucc import commutator as cmt
from pyscf.cc.addons import spatial2spin, spin2spatial
from pyscf import __config__

MEMORYMIN = getattr(__config__, 'cc_ccsd_memorymin', 2000)

#einsum = np.einsum
einsum = lib.einsum

# This is unrestricted (U)CCSD in spin-orbital form.

def update_amps(cc, t1, t2, eris):
    assert (isinstance(eris, _PhysicistsERIs))
    nocc, nvir = t1.shape
    fock = eris.fock

    fov = fock[:nocc,nocc:]
    mo_e_o = eris.mo_energy[:nocc]
    mo_e_v = eris.mo_energy[nocc:] + cc.level_shift

    #UCC3 T1 amplitude equation
    t1new = cmt.t1_amp_UCC3(t1,t2,eris)
    #UCC3 T2 amplitude equation
    t2new = cmt.t2_amp_UCC3(t1,t2,eris)

    #qUCCSD T1 amplitude equation
    #t1new = cmt.t1_single_commutator(t1,t2,eris)
    #t1new += cmt.t1_double_commutator(t1,t2,eris)
    #qUCCSD T2 amplitude equation
    #t2new = cmt.t2_single_commutator(t1,t2,eris)
    #t2new += cmt.t2_double_commutator(t1,t2,eris)

    eia = mo_e_o[:,None] - mo_e_v
    eijab = lib.direct_sum('ia,jb->ijab', eia, eia)
    t1new /= eia
    t2new /= eijab

    return t1new, t2new

def energy(cc, t1, t2, eris):
    # Bernoulli expansion
    e = cmt.bern_energy_double_commutator(t1,t2,eris)
    #e += cmt.bern_energy_triple_commutator(t1,t2,eris)

    if abs(e.imag) > 1e-4:
       logger.warn(cc, 'Non-zero imaginary part found in Unitary-CCSD energy %s', e)
    return e.real

vector_to_amplitudes = ccsd.vector_to_amplitudes_s4
amplitudes_to_vector = ccsd.amplitudes_to_vector_s4

def amplitudes_from_rccsd(t1, t2, orbspin=None):
    '''Convert spatial orbital T1,T2 to spin-orbital T1,T2'''
    return spatial2spin(t1, orbspin), spatial2spin(t2, orbspin)

class UnitaryGCCSD(ccsd.CCSDBase):

    conv_tol = getattr(__config__, 'ucc_ucc_UnitaryCCSD_conv_tol', 1e-7)
    conv_tol_normt = getattr(__config__, 'ucc_ucc_UnitaryCCSD_conv_tol_normt', 1e-7)

    def __init__(self, mf, frozen=None, mo_coeff=None, mo_occ=None):
        ccsd.CCSDBase.__init__(self, mf, frozen, mo_coeff, mo_occ)

    def init_amps(self, eris=None):
        if eris is None:
            eris = self.ao2mo(self.mo_coeff)
        mo_e = eris.mo_energy
        nocc = self.nocc
        eia = mo_e[:nocc,None] - mo_e[None,nocc:]
        eijab = lib.direct_sum('ia,jb->ijab', eia, eia)
        t1 = eris.fock[:nocc,nocc:] / eia
        eris_oovv = np.array(eris.oovv)
        t2 = eris_oovv.conj() / eijab
        #self.emp2 = 0.25*einsum('ijab,ijab', t2, eris_oovv.conj()).real
        self.emp2 = 0.25*einsum('ijab,ijab', t2, eris_oovv).real
        logger.info(self, 'Init t2, MP2 energy = %.15g', self.emp2)
        return self.emp2, t1, t2

    def mp_amps(self,eris=None):
        if eris is None:
            eris = self.ao2mo(self.mo_coeff)
        nocc = self.nocc
        #fock = eris.fock
        nvir = self.nmo - nocc
        mo_e = eris.mo_energy
        eia = mo_e[:nocc,None] - mo_e[None,nocc:]
        eijab = lib.direct_sum('ia,jb->ijab', eia, eia)
        eris_oovv = np.asarray(eris.oovv)
        t2 = eris_oovv.conj()/eijab
        t1 = np.zeros((nocc,nvir))
        #t1 = eris.fock[:nocc,nocc:] / eia
        emp2 = 0.25*einsum('ijab,ijab', t2, eris_oovv).real
        logger.info(self, 'Init t2, MP2 energy = %.15g', emp2)
        return emp2, t1, t2


    energy = energy
    update_amps = update_amps

    def kernel(self, t1=None, t2=None, eris=None, mbpt2=False):
        '''Ground-state unitary coupled-cluster theory

        Kwargs:
            mbpt2 : bool
                Use one-shot MBPT2 approximation to UCCSD.
        '''
        if mbpt2:
            from pyscf.mp import gmp2
            pt = gmp2.GMP2(self._scf, self.frozen, self.mo_coeff, self.mo_occ)
            self.e_corr, self.t2 = pt.kernel(eris=eris)
            nocc, nvir = self.t2.shape[1:3]
            self.t1 = np.zeros((nocc,nvir))
            return self.e_corr, self.t1, self.t2

        # Initialize orbspin so that we can attach it to t1, t2
        if getattr(self.mo_coeff, 'orbspin', None) is None:
            orbspin = scf.ghf.guess_orbspin(self.mo_coeff)
            if not np.any(orbspin == -1):
                self.mo_coeff = lib.tag_array(self.mo_coeff, orbspin=orbspin)

        self.e_corr, self.t1, self.t2 = ccsd.CCSDBase.ccsd(self, t1, t2, eris)
        if getattr(eris, 'orbspin', None) is not None:
            self.t1 = lib.tag_array(self.t1, orbspin=eris.orbspin)
            self.t2 = lib.tag_array(self.t2, orbspin=eris.orbspin)
        return self.e_corr, self.t1, self.t2

    def amplitudes_to_vector(self, t1, t2, out=None):
        return amplitudes_to_vector(t1, t2, out)

    def vector_to_amplitudes(self, vec, nmo=None, nocc=None):
        if nocc is None: nocc = self.nocc
        if nmo is None: nmo = self.nmo
        return vector_to_amplitudes(vec, nmo, nocc)

    def vector_size(self, nmo=None, nocc=None):
        if nocc is None: nocc = self.nocc
        if nmo is None: nmo = self.nmo
        nvir = nmo - nocc
        return nocc * nvir + nocc*(nocc-1)//2*nvir*(nvir-1)//2

    def amplitudes_from_rccsd(self, t1, t2, orbspin=None):
        return amplitudes_from_rccsd(t1, t2, orbspin)

    def ipccsd(self, nroots=1, koopmans=False, guess=None, eris=None):
        from pyscf.ucc import ucc_ppt
        return ucc_ppt.EOMIP(self).kernel(nroots, koopmans, guess, eris)

    def eeccsd(self, nroots=1, koopmans=False, guess=None, eris=None):
        from pyscf.ucc import ucc_ppt
        return ucc_ppt.EOMEE(self).kernel(nroots, koopmans, guess, eris)

    def ao2mo(self, mo_coeff=None):
        nmo = self.nmo
        mem_incore = nmo**4*2 * 8/1e6
        mem_now = lib.current_memory()[0]
        if (self._scf._eri is not None and
            (mem_incore+mem_now < self.max_memory) or self.mol.incore_anyway):
            return _make_eris_incore(self, mo_coeff)
        elif getattr(self._scf, 'with_df', None):
            raise NotImplementedError
        else:
            return _make_eris_outcore(self, mo_coeff)

    def eomip_method(self):
        from pyscf.cc import eom_gccsd
        return eom_gccsd.EOMIP(self)

    def eomee_method(self):
        from pyscf.cc import eom_gccsd
        return eom_gccsd.EOMEE(self)

    def amplitudes_from_ccsd(self, t1, t2):
        '''Convert spatial orbital T1,T2 to spin-orbital T1,T2'''
        return self.spatial2spin(t1), self.spatial2spin(t2)

    def spatial2spin(self, tx, orbspin=None):
        if orbspin is None:
            orbspin = getattr(self.mo_coeff, 'orbspin', None)
            if orbspin is not None:
                orbspin = orbspin[self.get_frozen_mask()]
        return spatial2spin(tx, orbspin)

    def spin2spatial(self, tx, orbspin=None):
        if orbspin is None:
            orbspin = getattr(self.mo_coeff, 'orbspin', None)
            if orbspin is not None:
                orbspin = orbspin[self.get_frozen_mask()]
        return spin2spatial(tx, orbspin)

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('')
        log.info('******** %s ********', self.__class__)
        log.info('UCCSD nocc = %s, nmo = %s', self.nocc, self.nmo)
        if self.frozen != 0:
            log.info('frozen orbitals %s', self.frozen)
        log.info('max_cycle = %d', self.max_cycle)
        log.info('direct = %d', self.direct)
        log.info('conv_tol = %g', self.conv_tol)
        log.info('conv_tol_normt = %s', self.conv_tol_normt)
        log.info('diis_space = %d', self.diis_space)
        #log.info('diis_file = %s', self.diis_file)
        log.info('diis_start_cycle = %d', self.diis_start_cycle)
        log.info('diis_start_energy_diff = %g', self.diis_start_energy_diff)
        log.info('max_memory %d MB (current use %d MB)',
                 self.max_memory, lib.current_memory()[0])
        if (log.verbose >= logger.DEBUG1 and
            self.__class__ == UCCSD):
            nocc = self.nocc
            nvir = self.nmo - self.nocc
            flops = _fp(nocc, nvir)
            log.debug1('total FLOPs %s', flops)
        return self

CCSD = UnitaryGCCSD


class _PhysicistsERIs:
    '''<pq||rs> = <pq|rs> - <pq|sr>'''
    def __init__(self, mol=None):
        self.mol = mol
        self.mo_coeff = None
        self.nocc = None
        self.fock = None
        self.orbspin = None

        self.oooo = None
        self.ooov = None
        self.oovv = None
        self.ovvo = None
        self.ovov = None
        self.ovvv = None
        self.vvvv = None

    def _common_init_(self, mycc, mo_coeff=None):
        if mo_coeff is None:
            mo_coeff = mycc.mo_coeff
        mo_idx = ccsd.get_frozen_mask(mycc)
        if getattr(mo_coeff, 'orbspin', None) is not None:
            self.orbspin = mo_coeff.orbspin[mo_idx]
            mo_coeff = lib.tag_array(mo_coeff[:,mo_idx], orbspin=self.orbspin)
        else:
            orbspin = scf.ghf.guess_orbspin(mo_coeff)
            mo_coeff = mo_coeff[:,mo_idx]
            if not np.any(orbspin == -1):
                self.orbspin = orbspin[mo_idx]
                mo_coeff = lib.tag_array(mo_coeff, orbspin=self.orbspin)
        self.mo_coeff = mo_coeff

        # Note: Recomputed fock matrix since SCF may not be fully converged.
        dm = mycc._scf.make_rdm1(mycc.mo_coeff, mycc.mo_occ)
        vhf = mycc._scf.get_veff(mycc.mol, dm)
        fockao = mycc._scf.get_fock(dm=dm)
        self.fock = reduce(np.dot, (mo_coeff.conj().T, fockao, mo_coeff))

        self.nocc = mycc.nocc
        self.mol = mycc.mol

        mo_e = self.mo_energy = self.fock.diagonal().real
        gap = abs(mo_e[:self.nocc,None] - mo_e[None,self.nocc:]).min()
        if gap < 1e-5:
            logger.warn(mycc, 'HOMO-LUMO gap %s too small for Unitary-CCSD', gap)
        return self

def _make_eris_incore(mycc, mo_coeff=None, ao2mofn=None):
    eris = _PhysicistsERIs()
    eris._common_init_(mycc, mo_coeff)
    nocc = eris.nocc
    nao, nmo = eris.mo_coeff.shape

    if callable(ao2mofn):
        eri = ao2mofn(eris.mo_coeff).reshape([nmo]*4)
    else:
        assert(eris.mo_coeff.dtype == np.double)
        mo_a = eris.mo_coeff[:nao//2]
        mo_b = eris.mo_coeff[nao//2:]
        orbspin = eris.orbspin
        if orbspin is None:
            eri  = ao2mo.kernel(mycc._scf._eri, mo_a)
            eri += ao2mo.kernel(mycc._scf._eri, mo_b)
            eri1 = ao2mo.kernel(mycc._scf._eri, (mo_a,mo_a,mo_b,mo_b))
            eri += eri1
            eri += eri1.T
        else:
            mo = mo_a + mo_b
            eri = ao2mo.kernel(mycc._scf._eri, mo)
            if eri.size == nmo**4:  # if mycc._scf._eri is a complex array
                eri = eri.reshape(nmo**2, nmo**2)
                sym_forbid = (orbspin[:,None] != orbspin).ravel()
            else:  # 4-fold symmetry
                sym_forbid = (orbspin[:,None] != orbspin)[np.tril_indices(nmo)]
            eri[sym_forbid,:] = 0
            eri[:,sym_forbid] = 0

        if eri.dtype == np.double:
            eri = ao2mo.restore(1, eri, nmo)

    eri = eri.reshape(nmo,nmo,nmo,nmo)
    eri = eri.transpose(0,2,1,3) - eri.transpose(0,2,3,1)

    eris.oooo = eri[:nocc,:nocc,:nocc,:nocc].copy()
    eris.ooov = eri[:nocc,:nocc,:nocc,nocc:].copy()
    eris.oovv = eri[:nocc,:nocc,nocc:,nocc:].copy()
    eris.ovov = eri[:nocc,nocc:,:nocc,nocc:].copy()
    eris.ovvo = eri[:nocc,nocc:,nocc:,:nocc].copy()
    eris.ovvv = eri[:nocc,nocc:,nocc:,nocc:].copy()
    eris.vvvv = eri[nocc:,nocc:,nocc:,nocc:].copy()
    return eris

def _make_eris_outcore(mycc, mo_coeff=None):
    from pyscf.scf.ghf import GHF
    assert isinstance(mycc._scf, GHF)
    cput0 = (logger.process_clock(), logger.perf_counter())
    log = logger.Logger(mycc.stdout, mycc.verbose)

    eris = _PhysicistsERIs()
    eris._common_init_(mycc, mo_coeff)
    nocc = eris.nocc
    nao, nmo = eris.mo_coeff.shape
    nvir = nmo - nocc
    assert (eris.mo_coeff.dtype == np.double)
    mo_a = eris.mo_coeff[:nao//2]
    mo_b = eris.mo_coeff[nao//2:]
    orbspin = eris.orbspin

    feri = eris.feri = lib.H5TmpFile()
    dtype = np.result_type(eris.mo_coeff).char
    eris.oooo = feri.create_dataset('oooo', (nocc,nocc,nocc,nocc), dtype)
    eris.ooov = feri.create_dataset('ooov', (nocc,nocc,nocc,nvir), dtype)
    eris.oovv = feri.create_dataset('oovv', (nocc,nocc,nvir,nvir), dtype)
    eris.ovov = feri.create_dataset('ovov', (nocc,nvir,nocc,nvir), dtype)
    eris.ovvo = feri.create_dataset('ovvo', (nocc,nvir,nvir,nocc), dtype)
    eris.ovvv = feri.create_dataset('ovvv', (nocc,nvir,nvir,nvir), dtype)

    if orbspin is None:
        orbo_a = mo_a[:,:nocc]
        orbv_a = mo_a[:,nocc:]
        orbo_b = mo_b[:,:nocc]
        orbv_b = mo_b[:,nocc:]

        max_memory = mycc.max_memory-lib.current_memory()[0]
        blksize = min(nocc, max(2, int(max_memory*1e6/8/(nmo**3*2))))
        max_memory = max(MEMORYMIN, max_memory)

        fswap = lib.H5TmpFile()
        ao2mo.kernel(mycc.mol, (orbo_a,mo_a,mo_a,mo_a), fswap, 'aaaa',
                     max_memory=max_memory, verbose=log)
        ao2mo.kernel(mycc.mol, (orbo_a,mo_a,mo_b,mo_b), fswap, 'aabb',
                     max_memory=max_memory, verbose=log)
        ao2mo.kernel(mycc.mol, (orbo_b,mo_b,mo_a,mo_a), fswap, 'bbaa',
                     max_memory=max_memory, verbose=log)
        ao2mo.kernel(mycc.mol, (orbo_b,mo_b,mo_b,mo_b), fswap, 'bbbb',
                     max_memory=max_memory, verbose=log)

        for p0, p1 in lib.prange(0, nocc, blksize):
            tmp  = np.asarray(fswap['aaaa'][p0*nmo:p1*nmo])
            tmp += np.asarray(fswap['aabb'][p0*nmo:p1*nmo])
            tmp += np.asarray(fswap['bbaa'][p0*nmo:p1*nmo])
            tmp += np.asarray(fswap['bbbb'][p0*nmo:p1*nmo])
            tmp = lib.unpack_tril(tmp).reshape(p1-p0,nmo,nmo,nmo)
            eris.oooo[p0:p1] = (tmp[:,:nocc,:nocc,:nocc].transpose(0,2,1,3) -
                                tmp[:,:nocc,:nocc,:nocc].transpose(0,2,3,1))
            eris.ooov[p0:p1] = (tmp[:,:nocc,:nocc,nocc:].transpose(0,2,1,3) -
                                tmp[:,nocc:,:nocc,:nocc].transpose(0,2,3,1))
            eris.ovvv[p0:p1] = (tmp[:,nocc:,nocc:,nocc:].transpose(0,2,1,3) -
                                tmp[:,nocc:,nocc:,nocc:].transpose(0,2,3,1))
            eris.oovv[p0:p1] = (tmp[:,nocc:,:nocc,nocc:].transpose(0,2,1,3) -
                                tmp[:,nocc:,:nocc,nocc:].transpose(0,2,3,1))
            eris.ovov[p0:p1] = (tmp[:,:nocc,nocc:,nocc:].transpose(0,2,1,3) -
                                tmp[:,nocc:,nocc:,:nocc].transpose(0,2,3,1))
            eris.ovvo[p0:p1] = (tmp[:,nocc:,nocc:,:nocc].transpose(0,2,1,3) -
                                tmp[:,:nocc,nocc:,nocc:].transpose(0,2,3,1))
            tmp = None
        cput0 = log.timer_debug1('transforming ovvv', *cput0)

        eris.vvvv = feri.create_dataset('vvvv', (nvir,nvir,nvir,nvir), dtype)
        tril2sq = lib.square_mat_in_trilu_indices(nvir)
        fswap = lib.H5TmpFile()
        ao2mo.kernel(mycc.mol, (orbv_a,orbv_a,orbv_a,orbv_a), fswap, 'aaaa',
                     max_memory=max_memory, verbose=log)
        ao2mo.kernel(mycc.mol, (orbv_a,orbv_a,orbv_b,orbv_b), fswap, 'aabb',
                     max_memory=max_memory, verbose=log)
        ao2mo.kernel(mycc.mol, (orbv_b,orbv_b,orbv_a,orbv_a), fswap, 'bbaa',
                     max_memory=max_memory, verbose=log)
        ao2mo.kernel(mycc.mol, (orbv_b,orbv_b,orbv_b,orbv_b), fswap, 'bbbb',
                     max_memory=max_memory, verbose=log)
        for p0, p1 in lib.prange(0, nvir, blksize):
            off0 = p0*(p0+1)//2
            off1 = p1*(p1+1)//2
            tmp  = np.asarray(fswap['aaaa'][off0:off1])
            tmp += np.asarray(fswap['aabb'][off0:off1])
            tmp += np.asarray(fswap['bbaa'][off0:off1])
            tmp += np.asarray(fswap['bbbb'][off0:off1])

            if p0 > 0:
                c = tmp[ tril2sq[p0:p1,:p0] - off0 ]
                c = lib.unpack_tril(c.reshape((p1-p0)*p0,-1))
                eris.vvvv[p0:p1,:p0] = c.reshape(p1-p0,p0,nvir,nvir)
            c = tmp[ tril2sq[:p1,p0:p1] - off0 ]
            c = lib.unpack_tril(c.reshape((p1-p0)*p1,-1))
            eris.vvvv[:p1,p0:p1] = c.reshape(p1,p1-p0,nvir,nvir)
            tmp = None

        for p0, p1 in lib.prange(0, nvir, blksize):
            tmp = np.asarray(eris.vvvv[p0:p1])
            eris.vvvv[p0:p1] = tmp.transpose(0,2,1,3) - tmp.transpose(0,2,3,1)
        cput0 = log.timer_debug1('transforming vvvv', *cput0)

    else:  # with orbspin
        mo = mo_a + mo_b
        orbo = mo[:,:nocc]
        orbv = mo[:,nocc:]

        max_memory = mycc.max_memory-lib.current_memory()[0]
        blksize = min(nocc, max(2, int(max_memory*1e6/8/(nmo**3*2))))
        max_memory = max(MEMORYMIN, max_memory)

        fswap = lib.H5TmpFile()
        ao2mo.kernel(mycc.mol, (orbo,mo,mo,mo), fswap,
                     max_memory=max_memory, verbose=log)
        sym_forbid = orbspin[:,None] != orbspin

        for p0, p1 in lib.prange(0, nocc, blksize):
            tmp = np.asarray(fswap['eri_mo'][p0*nmo:p1*nmo])
            tmp = lib.unpack_tril(tmp).reshape(p1-p0,nmo,nmo,nmo)
            tmp[sym_forbid[p0:p1]] = 0
            tmp[:,:,sym_forbid] = 0

            eris.oooo[p0:p1] = (tmp[:,:nocc,:nocc,:nocc].transpose(0,2,1,3) -
                                tmp[:,:nocc,:nocc,:nocc].transpose(0,2,3,1))
            eris.ooov[p0:p1] = (tmp[:,:nocc,:nocc,nocc:].transpose(0,2,1,3) -
                                tmp[:,nocc:,:nocc,:nocc].transpose(0,2,3,1))
            eris.ovvv[p0:p1] = (tmp[:,nocc:,nocc:,nocc:].transpose(0,2,1,3) -
                                tmp[:,nocc:,nocc:,nocc:].transpose(0,2,3,1))
            eris.oovv[p0:p1] = (tmp[:,nocc:,:nocc,nocc:].transpose(0,2,1,3) -
                                tmp[:,nocc:,:nocc,nocc:].transpose(0,2,3,1))
            eris.ovov[p0:p1] = (tmp[:,:nocc,nocc:,nocc:].transpose(0,2,1,3) -
                                tmp[:,nocc:,nocc:,:nocc].transpose(0,2,3,1))
            eris.ovvo[p0:p1] = (tmp[:,nocc:,nocc:,:nocc].transpose(0,2,1,3) -
                                tmp[:,:nocc,nocc:,nocc:].transpose(0,2,3,1))
            tmp = None
        cput0 = log.timer_debug1('transforming ovvv', *cput0)

        eris.vvvv = feri.create_dataset('vvvv', (nvir,nvir,nvir,nvir), dtype)
        sym_forbid = (orbspin[nocc:,None]!=orbspin[nocc:])[np.tril_indices(nvir)]
        tril2sq = lib.square_mat_in_trilu_indices(nvir)

        fswap = lib.H5TmpFile()
        ao2mo.kernel(mycc.mol, orbv, fswap, max_memory=max_memory, verbose=log)
        for p0, p1 in lib.prange(0, nvir, blksize):
            off0 = p0*(p0+1)//2
            off1 = p1*(p1+1)//2
            tmp = np.asarray(fswap['eri_mo'][off0:off1])
            tmp[sym_forbid[off0:off1]] = 0
            tmp[:,sym_forbid] = 0

            if p0 > 0:
                c = tmp[ tril2sq[p0:p1,:p0] - off0 ]
                c = lib.unpack_tril(c.reshape((p1-p0)*p0,-1))
                eris.vvvv[p0:p1,:p0] = c.reshape(p1-p0,p0,nvir,nvir)
            c = tmp[ tril2sq[:p1,p0:p1] - off0 ]
            c = lib.unpack_tril(c.reshape((p1-p0)*p1,-1))
            eris.vvvv[:p1,p0:p1] = c.reshape(p1,p1-p0,nvir,nvir)
            tmp = None

        for p0, p1 in lib.prange(0, nvir, blksize):
            tmp = np.asarray(eris.vvvv[p0:p1])
            eris.vvvv[p0:p1] = tmp.transpose(0,2,1,3) - tmp.transpose(0,2,3,1)
        cput0 = log.timer_debug1('transforming vvvv', *cput0)

    return eris

def _fp(nocc, nvir):
    '''Total float points'''
    return (nocc**3*nvir**2*2 + nocc**2*nvir**3*2 +     # Ftilde
            nocc**4*nvir*2 * 2 + nocc**4*nvir**2*2 +    # Wijkl
            nocc*nvir**4*2 * 2 +                        # Wabcd
            nocc**2*nvir**3*2 + nocc**3*nvir**2*2 +
            nocc**3*nvir**3*2 + nocc**3*nvir**3*2 +
            nocc**2*nvir**3*2 + nocc**3*nvir**2*2 +     # Wiabj
            nocc**2*nvir**3*2 + nocc**3*nvir**2*2 +     # t1
            nocc**3*nvir**2*2 * 2 + nocc**4*nvir**2*2 +
            nocc*(nocc+1)/2*nvir**4*2 +                 # vvvv
            nocc**2*nvir**3*2 * 2 + nocc**3*nvir**2*2 * 2 +     # t2
            nocc**3*nvir**3*2 +
            nocc**3*nvir**3*2 * 2 + nocc**3*nvir**2*2 * 4)      # Wiabj

