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

from pyscf import lib
import numpy as np

einsum = lib.einsum

def Hoo(t1,t2,eris):
    nocc, nvir = t1.shape
   #Single commutator + foo
    tmpoo = 0.25*einsum('ikab,jkab->ij',eris.oovv,t2,optimize=True) 
    tmpoo += einsum('ikja,ka->ij',eris.ooov,t1,optimize=True)
    tmpoo_t = tmpoo.transpose(1,0)
    Hij = eris.fock[:nocc,:nocc] + tmpoo + tmpoo_t.conj()
    del tmpoo, tmpoo_t
   #Double commutator
    #Third order terms
    tmp = einsum('klbc,ical->kbia',t2.conj(),eris.ovvo,optimize=True)
    tmpoo = 0.5*einsum('kbia,jkab->ij',tmp,t2,optimize=True) 
    #
    tmp = einsum('imkl,klab->imab',eris.oooo,t2.conj(),optimize=True)
    tmpoo += 0.125*einsum('imab,jmab->ij',tmp,t2,optimize=True)
    tmpoo_t = tmpoo.transpose(1,0)
    Hij += tmpoo + tmpoo_t.conj()
    del tmp, tmpoo, tmpoo_t
    #
    tmp = einsum('lkba,mkba->lm',t2.conj(),t2,optimize=True)
    Hij -= 0.5*einsum('lm,imjl->ij',tmp,eris.oooo,optimize=True)
    del tmp
    #
    tmp = einsum('klab,klac->bc',t2,t2.conj(),optimize=True)
    Hij -= 0.5*einsum('bc,icbj->ij',tmp,eris.ovvo,optimize=True)
    del tmp
   #------above terms are in UCC3---------

    #D.7
    tmp_oovv = -einsum('ilka,kb->ilba',eris.ooov,t1.conj(),optimize=True)
    tmpoo = 0.5*einsum('ilba,jlba->ij',tmp_oovv,t2,optimize=True)
    #D.8
    tmp_ov = einsum('lb,klab->ka',t1.conj(),t2,optimize=True)
    tmpoo += 0.5*einsum('ikja,ka->ij',eris.ooov,tmp_ov,optimize=True) 
    #D.9####################
    tmp_oovv = einsum('ibac,kb->ikac',eris.ovvv,t1.conj(),optimize=True) 
    tmpoo += 0.25*einsum('ikac,jkac->ij',tmp_oovv,t2,optimize=True)
    del tmp_oovv,tmp_ov
    #---
    #tmp_ooov = 0.5*einsum('ibac,jkac->ijkb',eris.ovvv,t2,optimize=True)
    #tmpoo = 0.5*einsum('ijkb,kb->ij',tmp_ooov,t1.conj(),optimize=True)
    ########################

    #D.10
    tmp_ov = einsum('ikab,kb->ia',eris.oovv,t1,optimize=True)
    tmpoo += (5.0/12.0)*einsum('ia,ja->ij',tmp_ov,t1,optimize=True)
    del tmp_ov
    #D.11
    tmp_ov = einsum('ibak,kb->ia',eris.ovvo,t1.conj(),optimize=True)
    tmpoo += 0.5*einsum('ia,ja->ij',tmp_ov,t1,optimize=True)
    del tmp_ov
    #---
    tmpoo_t = tmpoo.transpose(1,0)
    Hij += tmpoo + tmpoo_t.conj() 

    #D.12
    Hij -= einsum('ikjl,ka,la->ij',eris.oooo,t1,t1.conj(),optimize=True)
    #D.13
    Hij -= einsum('iabj,ka,kb->ij',eris.ovvo,t1.conj(),t1,optimize=True)

    return Hij

def Hvv(t1,t2,eris):
   #
    nocc, nvir = t1.shape
   #Single commutator + fvv
    tmpvv = -0.25*einsum('jica,jicb->ab',t2,eris.oovv,optimize=True)
    tmpvv += einsum('iacb,ic->ab',eris.ovvv,t1,optimize=True)
    tmpvv_t = tmpvv.transpose(1,0)
    Hab = eris.fock[nocc:,nocc:] + tmpvv + tmpvv_t.conj()
    del tmpvv, tmpvv_t

   #Double commtator
    tmp = einsum('ijcd,kdbj->ickb',t2.conj(),eris.ovvo,optimize=True)
    tmpvv = -0.5*einsum('ikca,ickb->ab',t2,tmp,optimize=True)
    #
    tmp = einsum('ijfd,fdbc->ijbc',t2.conj(),eris.vvvv,optimize=True)
    tmpvv -= 0.125*einsum('ijac,ijbc->ab',t2,tmp,optimize=True)
    tmpvv_t = tmpvv.transpose(1,0)
    Hab += tmpvv + tmpvv_t.conj()
    del tmpvv, tmpvv_t
    #
    tmpvv = einsum('ijfc,ijfd->cd',t2,t2.conj(),optmize=True)
    Hab += 0.5*einsum('cd,adbc->ab',tmpvv,eris.vvvv,optimize=True)
    del tmpvv
    #
    tmpoo = einsum('jidc,kidc->jk',t2.conj(),t2,optimize=True)
    Hab += 0.5*einsum('jk,kabj->ab',tmpoo,eris.ovvo,optimize=True)
    del tmpoo
   #------above terms are in UCC3---------

    #E.7
    tmp_oovv = einsum('icbd,jc->ijbd',eris.ovvv,t1.conj(),optimize=True)
    tmpvv = -0.5*einsum('jida,ijbd->ab',t2,tmp_oovv,optimize=True)
    #E.8
    tmp_ov = einsum('jd,ijcd->ic',t1.conj(),t2,optimize=True)
    tmpvv += 0.5*einsum('iacb,ic->ab',eris.ovvv,tmp_ov,optimize=True) 
    #E.9
    tmp_oovv = -einsum('kijb,jc->kicb',eris.ooov,t1.conj(),optimize=True) 
    tmpvv += -0.25*einsum('kica,kicb->ab',t2,tmp_oovv,optimize=True)
    del tmp_oovv,tmp_ov
    #E.10
    tmp_ov = einsum('ijbc,jc->ib',eris.oovv,t1,optimize=True)
    tmpvv -= (5.0/12.0)*einsum('ia,ib->ab',t1,tmp_ov,optimize=True)
    #E.11
    tmp_ov = einsum('icbj,jc->ib',eris.ovvo,t1.conj(),optimize=True)
    tmpvv -= 0.5*einsum('ia,ib->ab',t1,tmp_ov,optimize=True)
    del tmp_ov
    #-----
    tmpvv_t = tmpvv.transpose(1,0)
    Hab += tmpvv + tmpvv_t.conj()

    #E.12
    tmpvv = einsum('ic,id->cd',t1,t1.conj(),optimize=True)
    Hab += einsum('adbc,cd->ab',eris.vvvv,tmpvv,optimize=True)
    del tmpvv
    #E.13
    Hab += einsum('jabi,jc,ic->ab',eris.ovvo,t1,t1.conj(),optimize=True)

    return Hab

def Hovvo(t1,t2,eris):
   #Single commutator + <ia||bj>
    tmpiabj = 0.5*einsum('ikbc,kjca->iabj',t2.conj(),np.asarray(eris.oovv).conj(),optimize=True)
    tmpiabj += einsum('iabc,jc->iabj',eris.ovvv,t1,optimize=True)
    tmpiabj -= einsum('kijb,ka->iabj',eris.ooov,t1,optimize=True)
    tmpjbai = tmpiabj.transpose(3,2,1,0)
    Hiabj = np.asarray(eris.ovvo) + tmpiabj + tmpjbai.conj()
    del tmpiabj, tmpjbai
   #Double commutator
    #F.4+F.5
    tmpijab = einsum('imkl,klbc->imbc',eris.oooo,t2.conj(),optimize=True)
    tmpijab += einsum('ikce,cebd->ikbd',t2.conj(),eris.vvvv,optimize=True)
    tmpiabj = 0.25*einsum('ikbd,jkad->iabj',tmpijab,t2,optimize=True)
    del tmpijab
    ##F.4
    ##tmpijab = einsum('imkl,klbc->imbc',eris.oooo,t2.conj(),optimize=True)
    ##tmpiabj = 0.25*einsum('imbc,jmac->iabj',tmpijab,t2,optimize=True)
    ##del tmpijab
    ##F.5
    ##tmpijab = einsum('ikce,cebd->ikbd',t2.conj(),eris.vvvv,optimize=True)
    ##tmpiabj += 0.25*einsum('ikbd,jkad->iabj',tmpijab,t2,optimize=True)
    #F.6 
    tmpijab = einsum('ikdc,lcbk->ildb',t2.conj(),eris.ovvo,optimize=True)
    tmpiabj -= 0.5*einsum('ildb,ljda->iabj',tmpijab,t2,optimize=True)
    #F.7
    tmpiabj -= 0.5*einsum('libc,ljca->iabj',tmpijab,t2,optimize=True)
    del tmpijab
    #F.8
    tmpab = -0.5*einsum('klca,klcd->ad',t2,t2.conj(),optimize=True)
    tmpiabj += 0.5*einsum('ad,idbj->iabj',tmpab,eris.ovvo,optimize=True)
    del tmpab
    #F.9
    tmpij = 0.5*einsum('lkdc,jkdc->lj',t2.conj(),t2,optimize=True)
    tmpiabj -= 0.5*einsum('iabl,lj->iabj',eris.ovvo,tmpij,optimize=True)
    del tmpij
    #F.10
    tmp = einsum('klbd,jlcd->kcbj',t2.conj(),t2,optimize=True)
    tmpiabj += einsum('iack,kcbj->iabj',eris.ovvo,tmp,optimize=True)
    tmpjbai = tmpiabj.transpose(3,2,1,0)
    Hiabj += tmpiabj + tmpjbai.conj()
    del tmpiabj
    #F.11
    Hiabj += einsum('labk,iklj->iabj',tmp,eris.oooo,optimize=True)
    #F.12
    Hiabj += einsum('icdj,adcb->iabj',tmp,eris.vvvv,optimize=True)
    del tmp
    #F.13
    tmpoooo = 0.5*einsum('ilcd,kjcd->ilkj',t2.conj(),t2,optimize=True)
    Hiabj += einsum('ilkj,kabl->iabj',tmpoooo,eris.ovvo,optimize=True)
    del tmpoooo
    #F.14
    tmpvvvv = 0.5*einsum('klad,klcb->adcb',t2,t2.conj(),optimize=True)
    Hiabj += einsum('adcb,icdj->iabj',tmpvvvv,eris.ovvo,optimize=True)
    del tmpvvvv
   #------above terms are in UCC3---------

    #F.15
    tmpov = einsum('klac,lc->ka',t2,t1.conj(),optimize=True)
    tmpiabj = -0.5*einsum('kijb,ka->iabj',eris.ooov,tmpov,optimize=True)
    #F.16
    tmpiabj += 0.5*einsum('iabc,jc->iabj',eris.ovvv,tmpov,optimize=True)
    del tmpov

    #F.17
    tmpoovv = einsum('likb,kc->licb',eris.ooov,t1.conj(),optimize=True)
    tmpiabj += -0.5*einsum('licb,ljca->iabj',tmpoovv,t2,optimize=True)
    #F.20
    tmpiabj -= 0.5*einsum('ilbc,ljca->iabj',tmpoovv,t2,optimize=True)
    del tmpoovv
    #F.18
    tmpoovv = einsum('idbc,kd->ikbc',eris.ovvv,t1.conj(),optimize=True)
    tmpiabj += 0.5*einsum('ikbc,kjca->iabj',tmpoovv,t2,optimize=True)
    #F.19
    tmpiabj += 0.5*einsum('kicb,jkac->iabj',tmpoovv,t2,optimize=True)
    del tmpoovv
    #F.21
    tmpovoo = einsum('ikjc,klca->iajl',eris.ooov,t2,optimize=True)
    tmpiabj += 0.5*einsum('iajl,lb->iabj',tmpovoo,t1.conj(),optimize=True)
    del tmpovoo
    #F.22
    tmpvvvo = -einsum('kabc,kjdc->dabj',eris.ovvv,t2,optimize=True)
    tmpiabj += 0.5*einsum('dabj,id->iabj',tmpvvvo,t1.conj(),optimize=True)
    del tmpvvvo
    #F.23
    tmpvvov = 0.5*einsum('klac,kljb->acjb',t2,eris.ooov,optimize=True)
    tmpiabj += 0.5*einsum('acjb,ic->iabj',tmpvvov,t1.conj(),optimize=True)
    #F.24
    tmpovoo = 0.5*einsum('iadc,jkcd->iakj',eris.ovvv,t2,optimize=True)
    tmpiabj -= 0.5*einsum('iakj,kb->iabj',tmpovoo,t1.conj(),optimize=True)
    #----
    tmpjbai = tmpiabj.transpose(3,2,1,0)
    Hiabj += tmpiabj + tmpjbai.conj()

    #F.25
    tmpiabj = -(2.0/3.0)*einsum('ikbc,jc,ka->iabj',eris.oovv,t1,t1,optimize=True)
    #F.26
    tmpvv = -einsum('ka,kc->ac',t1,t1.conj(),optimize=True)
    tmpiabj += 0.5*einsum('icbj,ac->iabj',eris.ovvo,tmpvv,optimize=True)
    del tmpvv
    #F.27
    tmpoo = einsum('kc,jc->kj',t1.conj(),t1,optimize=True)
    tmpiabj -= 0.5*einsum('iabk,kj->iabj',eris.ovvo,tmpoo,optimize=True)
    del tmpoo
    #F.30
    tmpiabj += einsum('kcbj,ka,ic->iabj',eris.ovvo,t1,t1.conj(),optmize=True)
    #----
    tmpjbai = tmpiabj.transpose(3,2,1,0)
    Hiabj += tmpiabj + tmpjbai.conj()
    #
    #F.28
    Hiabj += einsum('iklj,lb,ka->iabj',eris.oooo,t1.conj(),t1,optimize=True)
    #F.29
    Hiabj += einsum('adcb,jc,id->iabj',eris.vvvv,t1,t1.conj(),optimize=True)

    return Hiabj

def Hvvvo(t1,t2,eris): 
    eris_vvvo = np.asarray(eris.ovvv).transpose(3,2,1,0)
    #
    tmpabci = einsum('jadc,ijbd->abci',eris.ovvv,t2,optimize=True)
    Habci = tmpabci - tmpabci.transpose(1,0,2,3)
    Habci += eris_vvvo.conj()
    del tmpabci
    
    Habci += 0.5*einsum('kjic,jkab->abci',eris.ooov,t2,optimize=True)
   #------above terms are in UCC3---------

    Habci -= 0.5*einsum('jc,jiab->abci',t1.conj(),eris.oovv,optimize=True)
    
    Habci += einsum('abcd,id->abci',eris.vvvv,t1,optimize=True)
    
    tmp = einsum('jaci,jb->abci',eris.ovvo,t1,optimize=True)
    Habci += tmp - tmp.transpose(1,0,2,3)
    del tmp
    return Habci

def Hovoo(t1,t2,eris): 
    eris_ovoo = np.asarray(eris.ooov).transpose(2,3,0,1)
    #
    tmpiajk = einsum('iljb,klab->iajk',eris.ooov,t2,optimize=True)
    Hiajk = tmpiajk - tmpiajk.transpose(0,1,3,2)
    Hiajk += eris_ovoo.conj()
    del tmpiajk
    #
    Hiajk += 0.5*einsum('iabc,jkbc->iajk',eris.ovvv,t2,optimize=True)
   #------above terms are in UCC3---------
    
    Hiajk += 0.5*einsum('ib,jkba->iajk',t1.conj(),eris.oovv,optimize=True)
    #
    Hiajk -= einsum('iljk,la->iajk',eris.oooo,t1,optimize=True)
    #
    tmp = -einsum('iabj,kb->iajk',eris.ovvo,t1,optimize=True)
    Hiajk += tmp - tmp.transpose(0,1,3,2)
    del tmp
    return Hiajk

def Hoooo(t1,t2,eris):
    Hijkl = np.asarray(eris.oooo)
    return Hijkl

def Hvvvv(t1,t2,eris):
    Habcd = np.asarray(eris.vvvv)
    return Habcd

####################################################################
