#!/usr/bin/env python
# Copyright 2014-2018 The PySCF Developers. All Rights Reserved.
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
from pyscf import lib

einsum = lib.einsum

def bern_energy_double_commutator(t1,t2,eris):
    e = 0.25*einsum('ijab,ijab', t2, eris.oovv)
    e += (1.0/6.0)*einsum('ia,jb,ijab', t1, t1, eris.oovv)

    if abs(e.imag) > 1e-4:
        logger.warn(cc, 'Non-zero imaginary part found in UCC double commutator energy %s', e)
    return e

def bern_energy_triple_commutator(t1,t2,eris):
    #fourth order term
    #1
    wphhp = einsum('ijab,ikac->jbkc',eris.oovv,t2)
    wpphh = -einsum('jbkc,jlbd->klcd',wphhp,t2)
    #2
    ftaa = einsum('ijac,ijab->cb',t2,eris.oovv)
    wpphh += (1.0/2.0)*einsum('cb,klbd->klcd',ftaa,t2)
    #3
    ftii = einsum('ijab,ikab->jk',eris.oovv,t2)
    wpphh += (1.0/2.0)*einsum('jk,jlcd->klcd',ftii,t2)
    #4
    whhhh = einsum('ijab,klab->ijkl',eris.oovv,t2)
    wpphh += -(1.0/8.0)*einsum('ijkl,ijcd->klcd',whhhh,t2)
    #
    e = (1.0/6.0)*einsum('klcd,klcd',wpphh,t2.conj())

    #fifth order term
    #5
    ftii = einsum('jika,ia->jk',eris.ooov,t1)
    wpphh = einsum('jk,jlcd->klcd',ftii,t2)
    #6
    ftaa = einsum('icab,ia->cb',eris.ovvv,t1)
    wpphh += -einsum('cb,jkbd->jkcd',ftaa,t2)
    #7
    wphhp = einsum('ijka,ib->bkaj',np.array(eris.ooov).conj(),t1.conj())
    wpphh += 2.0*einsum('bkaj,klbc->jlac',wphhp,t2)
    #8
    wphhp = einsum('icba,ja->cjbi',np.array(eris.ovvv).conj(),t1.conj())
    wpphh += -2.0*einsum('cjbi,jkcd->ikbd',wphhp,t2)
    #9,10
    tai = einsum('lc,ilbc->ib',t1.conj(),t2)
    #9
    wpphh += (1.0/3.0)*einsum('jkia,ib->jkba',np.array(eris.ooov).conj(),tai)
    #10
    wpphh += -(1.0/3.0)*einsum('icab,jc->ijab',np.array(eris.ovvv).conj(),tai)
    #11
    whhhh = einsum('kjia,la->ilkj',np.array(eris.ooov).conj(),t1.conj())
    wpphh += -0.5*einsum('ilkj,libc->jkbc',whhhh,t2)
    #12
    wpppp = einsum('icba,id->badc',np.array(eris.ovvv).conj(),t1.conj())
    wpphh += 0.5*einsum('badc,jkdc->jkba',wpppp,t2)

    #sixth order term
    #14
    wii = einsum('kc,ic->ki',t1.conj(),t1)
    wpphh += (1.0/3.0)*einsum('ki,kjab->ijab',wii,np.array(eris.oovv).conj())
    #15
    waa = einsum('ka,kc->ac',t1,t1.conj())
    wpphh += (1.0/3.0)*einsum('ac,ijcb->ijab',waa,np.array(eris.oovv).conj())
    #17
    wpphh += (4.0/3.0)*einsum('iabj,kb,ic->kjca',eris.ovvo,t1,t1)
    #18
    wpphh += -(1.0/3.0)*einsum('klij,ka,lb->ijab',eris.oooo,t1,t1)
    #19
    wpphh += -(1.0/3.0)*einsum('abcd,ic,jd->ijab',eris.vvvv,t1,t1)
    #
    e += 0.5*einsum('jkcd,jkcd',wpphh,t2.conj())

    #13 (sixth order term)
    wia = einsum('ijab,ia->jb',eris.oovv,t1)
    tai = -(1.0/12.0)*einsum('jkbc,jb->kc',t2,wia)

    #20
    wii = einsum('ijka,ja->ik',eris.ooov,t1)
    tai += (1.0/3.0)*einsum('ik,ib->kb',wii,t1)
    #21
    waa = einsum('iacb,ic->ab',eris.ovvv,t1.conj())
    tai += -(1.0/3.0)*einsum('ab,jb->ja',waa,t1)
    #
    e += 2.0*einsum('ja,ja',tai,t1.conj())

    if abs(e.imag) > 1e-4:
        logger.warn(cc, 'Non-zero imaginary part found in UCC triple commutator energy %s', e)
    return e

def t1_single_commutator(t1,t2,eris):
    t1new = 0.5*einsum('jabc,ijcb->ia',eris.ovvv,t2)
    t1new +=-0.5*einsum('kjib,kjab->ia',eris.ooov,t2)
    t1new += 0.5*einsum('ijab,jb->ia',np.array(eris.oovv).conj(),t1.conj())
    t1new += einsum('jabi,jb->ia',eris.ovvo,t1)
    
    return t1new

def t1_double_commutator(t1,t2,eris):
    #1
    ftii = einsum('kjcb,ljcb->kl',t2.conj(),t2)
    t1new = -0.5*einsum('kl,kila->ia',ftii,np.array(eris.ooov).conj())
    #2
    ftaa = einsum('jkbc,jkbd->cd',t2.conj(),t2)
    t1new += 0.5*einsum('cd,icad->ia',ftaa,np.array(eris.ovvv).conj())
    #5+6
    ftia = -0.25*einsum('kjlb,kjcb->lc',np.array(eris.ooov).conj(),t2.conj())
    ftia += 0.25*einsum('jkbd,jcbd->kc',t2.conj(),np.array(eris.ovvv).conj())
    t1new += einsum('lc,ilac->ia',ftia,t2)
    #8
    whhhh = einsum('kjcb,ilbc->kjli',t2.conj(),t2)
    t1new += +0.25*einsum('kjla,kjli->ia',np.array(eris.ooov).conj(),whhhh)
    #7
    whhhp = einsum('jkbd,icbd->jkic',t2.conj(),np.array(eris.ovvv).conj())
    t1new += -0.25*einsum('jkic,jkac->ia',whhhp,t2)
    #3+4
    tmp = einsum('jkbc,klca->jbla',t2.conj(),t2)
    t1new += -einsum('ijlb,jbla->ia',np.array(eris.ooov).conj(),tmp)
    t1new +=  einsum('jdba,jbid->ia',np.array(eris.ovvv).conj(),tmp)

    #quadratic 
    #a
    t1ia = einsum('jkbc,jb->kc',eris.oovv,t1)
    t1new += (5.0/12.0)*einsum('kc,ikac->ia',t1ia,t2)
    #d
    t1ia = einsum('jkbc,jb->kc',t2.conj(),t1)
    t1new += (1.0/3.0)*einsum('kc,ikac->ia',t1ia,np.array(eris.oovv).conj())
    #b+e
    ftii = (1.0/3.0)*einsum('jkbc,jibc->ki',eris.oovv,t2)
    ftii += (1.0/6.0)*einsum('kjcb,jibc->ki',t2.conj(),np.array(eris.oovv).conj())
    t1new += -einsum('ki,ka->ia',ftii,t1)
    #c+f
    ftaa = (1.0/3.0)*einsum('jkba,jkbc->ac',t2,eris.oovv)
    ftaa += (1.0/6.0)*einsum('jkab,jkcb->ac',np.array(eris.oovv).conj(),t2.conj())
    t1new += -einsum('ic,ac->ia',t1,ftaa)
    #g
    t2ijab = einsum('ijcd,abcd->ijab',t2,eris.vvvv)
    t1new += 0.25*einsum('ijab,jb->ia',t2ijab,t1.conj())
    #h
    t2ijab = einsum('ljki,ljba->kiba',eris.oooo,t2)
    t1new += 0.25*einsum('kiba,kb->ia',t2ijab,t1.conj())
    #i
    tmp = einsum('jcbi,jkba->ciak',eris.ovvo,t2)
    t1new += -0.5*einsum('ciak,kc->ia',tmp,t1.conj())
    #j
    t1new += -0.5*einsum('akci,kc->ia',tmp,t1.conj())
    #l+m
    ftaa = einsum('jabc,jb->ac',eris.ovvv,t1)
    ftaa += 0.5*einsum('jcba,jb->ac',np.array(eris.ovvv).conj(),t1.conj())  
    t1new += einsum('ac,ic->ia',ftaa,t1)
    #k+n
    ftii = -einsum('kjib,jb->ki',eris.ooov,t1)
    ftii += -0.5*einsum('ijkb,jb->ki',np.array(eris.ooov).conj(),t1.conj())
    t1new += einsum('ki,ka->ia',ftii,t1)
    #o
    ftaa = einsum('jb,jc->bc',t1,t1.conj())
    t1new += 0.5*einsum('ibac,bc->ia',np.array(eris.ovvv).conj(),ftaa)
    #p
    ftii = einsum('jb,kb->jk',t1.conj(),t1)
    t1new += -0.5*einsum('jika,jk->ia',np.array(eris.ooov).conj(),ftii)

    return t1new


def t2_single_commutator(t1,t2,eris):
    t2new = np.array(eris.oovv).conj()
    tmp = einsum('icab,jc->ijab',np.array(eris.ovvv).conj(),t1)
    t2new += tmp - tmp.transpose(1,0,2,3)
    del tmp
    #
    tmp = einsum('ijkb,ka->ijab',np.array(eris.ooov).conj(),t1)
    tmpijab = tmp - tmp.transpose(0,1,3,2)
    t2new += -tmpijab
    del tmpijab,tmp
    #
    t2new += 0.5*einsum('klab,klij->ijab',t2,eris.oooo)
    t2new += 0.5*einsum('ijcd,abcd->ijab',t2,eris.vvvv)
    tmp = einsum('kaci,kjcb->ijab',eris.ovvo,t2)
    tmpijab = tmp - tmp.transpose(1,0,2,3)
    t2new += tmpijab - tmpijab.transpose(0,1,3,2)
    del tmpijab, tmp

    return t2new


def t2_double_commutator(t1,t2,eris):
    #11
    whhhh = einsum('klcd,ijcd->klij',eris.oovv,t2)
    t2new = (1.0/6.0)*einsum('klij,klab->ijab',whhhh,t2)
    #4
    whhhh = einsum('klab,ijab->klij',t2.conj(),np.array(eris.oovv).conj())
    t2new += (1.0/12.0)*einsum('klij,klab->ijab',whhhh,t2)
    #5
    whhhh = einsum('klcd,ijcd->klij',t2.conj(),t2)
    t2new += (1.0/12.0)*einsum('klij,klab->ijab',whhhh,np.array(eris.oovv).conj())
    #10
    tmp1 = einsum('klcd,ikac->ilad',eris.oovv,t2)
    tmp2 = einsum('ilad,ljdb->ijab',tmp1,t2)
    tmpijab = tmp2 - tmp2.transpose(1,0,2,3)
    t2new += (1.0/3.0)*(tmpijab - tmpijab.transpose(0,1,3,2))
    del tmp1, tmp2, tmpijab
    #3
    tmp1 = einsum('ilad,lkdc->ikac',np.array(eris.oovv).conj(),t2.conj())
    tmp2 = einsum('ikac,kjcb->ijab',tmp1,t2)
    #tmp1 = einsum('klcd,ikac->ilad',eris.oovv,t2)
    #tmp2 = einsum('ilad,ijab->ljdb',tmp1.conj(),t2)
    tmpijab = tmp2 - tmp2.transpose(1,0,2,3)
    t2new += (1.0/3.0)*(tmpijab - tmpijab.transpose(0,1,3,2))
    del tmp1, tmp2, tmpijab
    #13
    ftii = einsum('klcd,kjcd->lj',eris.oovv,t2)
    tmp = einsum('lj,ilab->ijab',ftii,t2)
    tmpijab = tmp - tmp.transpose(1,0,2,3)
    t2new += -(1.0/3.0)*tmpijab
    del tmp, tmpijab
    #7
    ftii = einsum('klcd,kjcd->lj',t2.conj(),t2)
    tmp = einsum('lj,ilab->ijab',ftii,np.array(eris.oovv).conj())
    tmpijab = tmp - tmp.transpose(1,0,2,3)
    t2new += -(1.0/6.0)*tmpijab
    del tmp, tmpijab
    #9
    ftii = einsum('klcd,kjcd->lj',t2.conj(),np.array(eris.oovv).conj())
    tmp = einsum('lj,ilab->ijab',ftii,t2)
    tmpijab = tmp - tmp.transpose(1,0,2,3)
    t2new += -(1.0/6.0)*tmpijab
    del tmp, tmpijab
    #12
    ftaa = einsum('lkcb,lkcd->bd',t2,eris.oovv)
    tmp = einsum('ijad,bd->ijab',t2,ftaa)
    tmpijab = tmp - tmp.transpose(0,1,3,2)
    t2new += -(1.0/3.0)*tmpijab
    del tmp, tmpijab
    #6
    ftaa = einsum('lkcb,lkcd->bd',t2,t2.conj())
    tmp = einsum('ijad,bd->ijab',np.array(eris.oovv).conj(),ftaa)
    tmpijab = tmp - tmp.transpose(0,1,3,2)
    t2new += -(1.0/6.0)*tmpijab
    del tmp, tmpijab
    #8
    ftaa = einsum('lkcb,lkcd->bd',np.array(eris.oovv).conj(),t2.conj())
    tmp = einsum('ijad,bd->ijab',t2,ftaa)
    tmpijab = tmp - tmp.transpose(0,1,3,2)
    t2new += -(1.0/6.0)*tmpijab
    del tmp, tmpijab

    #quadratic 
    #23
    ftii = einsum('jlkc,lc->jk',eris.ooov,t1)
    tmp = -einsum('jk,ikab->ijab',ftii.conj(),t2)
    t2new += tmp - tmp.transpose(1,0,2,3)
    del tmp
    #24
    ftaa = einsum('ldcb,lc->db',eris.ovvv,t1)
    tmp = einsum('db,ijad->ijab',ftaa.conj(),t2)
    t2new += tmp - tmp.transpose(0,1,3,2)
    del tmp
    #26
    ftia = einsum('ljcd,lc->jd',t2,t1.conj())
    tmp = 0.5*einsum('idab,jd->ijab',np.array(eris.ovvv).conj(),ftia)
    t2new += tmp - tmp.transpose(1,0,2,3)
    del tmp
    #25
    tmp = -0.5*einsum('jika,kb->ijab',np.array(eris.ooov).conj(),ftia)
    t2new += tmp - tmp.transpose(0,1,3,2)
    del tmp
    #27
    whpph = einsum('idac,lc->idal',eris.ovvv,t1)
    tmp = einsum('idal,ljdb->ijab',whpph.conj(),t2)
    tmpijab = tmp - tmp.transpose(1,0,2,3)
    t2new += tmpijab - tmpijab.transpose(0,1,3,2)
    del tmpijab, tmp
    #28
    wphhp = einsum('lika,lc->cika',eris.ooov,t1)
    tmp = -einsum('cika,kjcb->ijab',wphhp.conj(),t2)
    tmpijab = tmp - tmp.transpose(1,0,2,3)
    t2new += tmpijab - tmpijab.transpose(0,1,3,2)
    del tmpijab
    #29
    whhhh = einsum('ijkc,lc->ijkl',eris.ooov,t1)
    t2new += einsum('ijkl,klab->ijab',whhhh.conj(),t2)
    #30
    wpppp = einsum('ldba,lc->cdba',eris.ovvv,t1)
    t2new += -einsum('cdba,jicd->jiba',wpppp.conj(),t2)
    #21
    ftii = einsum('lkjc,kc->lj',eris.ooov,t1)
    tmp = -einsum('lj,ilab->ijab',ftii,t2)
    t2new += tmp - tmp.transpose(1,0,2,3)
    #22
    ftaa = einsum('kbcd,kc->bd',eris.ovvv,t1)
    tmp = einsum('bd,ijad->ijab',ftaa,t2)
    t2new += tmp - tmp.transpose(0,1,3,2)
    #18
    whpph = einsum('kbcd,jd->kbcj',eris.ovvv,t1)
    tmp = einsum('kbcj,ikac->ijab',whpph,t2)
    tmpijab = tmp - tmp.transpose(1,0,2,3)
    t2new += tmpijab - tmpijab.transpose(0,1,3,2)
    del tmpijab,tmp
    #17
    wphhp = einsum('lkjc,lb->bkjc',eris.ooov,t1)
    tmp = -einsum('bkjc,ikac->ijab',wphhp,t2)
    tmpijab = tmp - tmp.transpose(1,0,2,3)
    t2new += tmpijab - tmpijab.transpose(0,1,3,2)
    del tmpijab,tmp
    #20
    whhhh = einsum('lkic,jc->lkij',eris.ooov,t1)
    whhhh = whhhh - whhhh.transpose(0,1,3,2)
    t2new += 0.5*einsum('lkij,lkab->ijab',whhhh,t2)
    #19
    wpppp = einsum('kacd,kb->bacd',eris.ovvv,t1)
    wpppp = wpppp - wpppp.transpose(1,0,2,3)
    t2new += -0.5*einsum('bacd,jicd->jiba',wpppp,t2)
    #15
    tmp = einsum('klij,ka->alij',eris.oooo,t1)
    r2 = 0.5*einsum('alij,lb->ijab',tmp,t1)
    t2new += r2 - r2.transpose(0,1,3,2)
    #14
    tmp = einsum('abcd,ic->abid',eris.vvvv,t1)
    r2 = 0.5*einsum('abid,jd->ijab',tmp,t1)
    t2new += r2 - r2.transpose(1,0,2,3)
    #1
    ftaa = einsum('kb,kc->bc',t1,t1.conj())
    tmp = -(1.0/3.0)*einsum('ijac,bc->ijab',eris.oovv,ftaa)
    t2new += tmp - tmp.transpose(0,1,3,2)
    #2
    ftii = einsum('kc,jc->kj',t1.conj(),t1)
    tmp = -(1.0/3.0)*einsum('ikab,kj->ijab',eris.oovv,ftii)
    t2new += tmp - tmp.transpose(1,0,2,3)
    #16
    tmp1 = einsum('kaci,kb->baci',eris.ovvo,t1)
    tmp2 = -einsum('baci,jc->jiba',tmp1,t1)
    tmpijab = tmp2 - tmp2.transpose(1,0,2,3)
    t2new += tmpijab - tmpijab.transpose(0,1,3,2)
    del tmpijab,tmp1,tmp2

    return t2new

def t1_amp_UCC3(t1,t2,eris):
    #single commutator
    t1new = 0.5*einsum('jabc,ijcb->ia',eris.ovvv,t2)
    t1new +=-0.5*einsum('kjib,kjab->ia',eris.ooov,t2)
    t1new += 0.5*einsum('ijab,jb->ia',np.array(eris.oovv).conj(),t1.conj())
    t1new += einsum('jabi,jb->ia',eris.ovvo,t1)
    #third order term in double commtator
    #1
    ftii = einsum('kjcb,ljcb->kl',t2.conj(),t2)
    t1new += -0.5*einsum('kl,kila->ia',ftii,np.array(eris.ooov).conj())
    #2
    ftaa = einsum('jkbc,jkbd->cd',t2.conj(),t2)
    t1new += 0.5*einsum('cd,icad->ia',ftaa,np.array(eris.ovvv).conj())
    #5+6
    ftia = -0.25*einsum('kjlb,kjcb->lc',np.array(eris.ooov).conj(),t2.conj())
    ftia += 0.25*einsum('jkbd,jcbd->kc',t2.conj(),np.array(eris.ovvv).conj())
    t1new += einsum('lc,ilac->ia',ftia,t2)
    #8
    whhhh = einsum('kjcb,ilbc->kjli',t2.conj(),t2)
    t1new += +0.25*einsum('kjla,kjli->ia',np.array(eris.ooov).conj(),whhhh)
    #7
    whhhp = einsum('jkbd,icbd->jkic',t2.conj(),np.array(eris.ovvv).conj())
    t1new += -0.25*einsum('jkic,jkac->ia',whhhp,t2)
    #3+4
    tmp = einsum('jkbc,klca->jbla',t2.conj(),t2)
    t1new += -einsum('ijlb,jbla->ia',np.array(eris.ooov).conj(),tmp)
    t1new +=  einsum('jdba,jbid->ia',np.array(eris.ovvv).conj(),tmp)
    del tmp

    return t1new 

def t2_amp_UCC3(t1,t2,eris):
    #single commutator
    t2new = np.array(eris.oovv).conj()
    tmp = einsum('icab,jc->ijab',np.array(eris.ovvv).conj(),t1)
    t2new += tmp - tmp.transpose(1,0,2,3)
    del tmp
    #
    tmp = einsum('ijkb,ka->ijab',np.array(eris.ooov).conj(),t1)
    tmpijab = tmp - tmp.transpose(0,1,3,2)
    t2new += -tmpijab
    del tmpijab,tmp
    #
    t2new += 0.5*einsum('klab,klij->ijab',t2,eris.oooo)
    t2new += 0.5*einsum('ijcd,abcd->ijab',t2,eris.vvvv)
    tmp = einsum('kaci,kjcb->ijab',eris.ovvo,t2)
    tmpijab = tmp - tmp.transpose(1,0,2,3)
    t2new += tmpijab - tmpijab.transpose(0,1,3,2)
    del tmpijab, tmp
    #third order term in double commtator
    #11
    whhhh = einsum('klcd,ijcd->klij',eris.oovv,t2)
    t2new += (1.0/6.0)*einsum('klij,klab->ijab',whhhh,t2)
    #4
    whhhh = einsum('klab,ijab->klij',t2.conj(),np.array(eris.oovv).conj())
    t2new += (1.0/12.0)*einsum('klij,klab->ijab',whhhh,t2)
    #5
    whhhh = einsum('klcd,ijcd->klij',t2.conj(),t2)
    t2new += (1.0/12.0)*einsum('klij,klab->ijab',whhhh,np.array(eris.oovv).conj())
    #10
    tmp1 = einsum('klcd,ikac->ilad',eris.oovv,t2)
    tmp2 = einsum('ilad,ljdb->ijab',tmp1,t2)
    tmpijab = tmp2 - tmp2.transpose(1,0,2,3)
    t2new += (1.0/3.0)*(tmpijab - tmpijab.transpose(0,1,3,2))
    del tmp1, tmp2, tmpijab
    #3
    tmp1 = einsum('ilad,lkdc->ikac',np.array(eris.oovv).conj(),t2.conj())
    tmp2 = einsum('ikac,kjcb->ijab',tmp1,t2)
    #tmp1 = einsum('klcd,ikac->ilad',eris.oovv,t2)
    #tmp2 = einsum('ilad,ijab->ljdb',tmp1.conj(),t2)
    tmpijab = tmp2 - tmp2.transpose(1,0,2,3)
    t2new += (1.0/3.0)*(tmpijab - tmpijab.transpose(0,1,3,2))
    del tmp1, tmp2, tmpijab
    #13
    ftii = einsum('klcd,kjcd->lj',eris.oovv,t2)
    tmp = einsum('lj,ilab->ijab',ftii,t2)
    tmpijab = tmp - tmp.transpose(1,0,2,3)
    t2new += -(1.0/3.0)*tmpijab
    del tmp, tmpijab
    #7
    ftii = einsum('klcd,kjcd->lj',t2.conj(),t2)
    tmp = einsum('lj,ilab->ijab',ftii,np.array(eris.oovv).conj())
    tmpijab = tmp - tmp.transpose(1,0,2,3)
    t2new += -(1.0/6.0)*tmpijab
    del tmp, tmpijab
    #9
    ftii = einsum('klcd,kjcd->lj',t2.conj(),np.array(eris.oovv).conj())
    tmp = einsum('lj,ilab->ijab',ftii,t2)
    tmpijab = tmp - tmp.transpose(1,0,2,3)
    t2new += -(1.0/6.0)*tmpijab
    del tmp, tmpijab
    #12
    ftaa = einsum('lkcb,lkcd->bd',t2,eris.oovv)
    tmp = einsum('ijad,bd->ijab',t2,ftaa)
    tmpijab = tmp - tmp.transpose(0,1,3,2)
    t2new += -(1.0/3.0)*tmpijab
    del tmp, tmpijab
    #6
    ftaa = einsum('lkcb,lkcd->bd',t2,t2.conj())
    tmp = einsum('ijad,bd->ijab',np.array(eris.oovv).conj(),ftaa)
    tmpijab = tmp - tmp.transpose(0,1,3,2)
    t2new += -(1.0/6.0)*tmpijab
    del tmp, tmpijab
    #8
    ftaa = einsum('lkcb,lkcd->bd',np.array(eris.oovv).conj(),t2.conj())
    tmp = einsum('ijad,bd->ijab',t2,ftaa)
    tmpijab = tmp - tmp.transpose(0,1,3,2)
    t2new += -(1.0/6.0)*tmpijab
    del tmp, tmpijab

    return t2new
