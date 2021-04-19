/***********************************************************************
 *
 * Copyright (C) 2002,2003,2004,2005,2006,2007,2008 Carsten Urbach,
 * 2014 Mario Schroeck
 *
 * This file is part of tmLQCD.
 *
 * tmLQCD is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * tmLQCD is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.	See the
 * GNU General Public License for more deta_BSMils.
 * 
 * You should have received a copy of the GNU General Public License
 * along with tmLQCD.	If not, see <http://www.gnu.org/licenses/>.
 *
 *******************************************************************************/

/*******************************************************************************
 *
 * Action of a Dirac operator (Frezzotti-Rossi BSM toy model) on a bispinor field
 *
 *******************************************************************************/
#ifdef HAVE_CONFIG_H
# include<tmlqcd_config.h>
#endif

#ifdef TM_USE_BSM
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "global.h"
#include "su3.h"
#include "sse.h"
#include "boundary.h"
#ifdef TM_USE_MPI
# include "xchange/xchange.h"
#endif
#include "update_backward_gauge.h"
#include "block.h"
#include "operator/D_psi_BSM.h"
#include "operator/D_psi_BSM3.h"
#include "solver/dirac_operator_eigenvectors.h"
#include "buffers/utils.h"
#include "linalg_eo.h"

#include "operator/clovertm_operators.h"
#include "operator/clover_leaf.h"


static bispinor *tempor;

void init_D_psi_BSM3(){


     tempor=(bispinor *)calloc(VOLUMEPLUSRAND,sizeof(bispinor));
}
void free_D_psi_BSM3(){

     free(tempor);
}

static inline void tm3_add(bispinor * const out, const bispinor * const in, const double sign)
{
  /*out+=s*i\gamma_5 \tau_3 mu3 *in
   * sign>0 for D+i\gamma_5\tau_3
   * sign<0 for D_dag-i\gamma_5\tau_3
   */
  const double s = (sign < 0) ? -1. : 1. ;

  /* out_up += s * i \gamma_5 \mu3 * in_up */
  _vector_add_i_mul(out->sp_up.s0,  s*mu03_BSM, in->sp_up.s0);
  _vector_add_i_mul(out->sp_up.s1,  s*mu03_BSM, in->sp_up.s1);
  _vector_add_i_mul(out->sp_up.s2, -s*mu03_BSM, in->sp_up.s2);
  _vector_add_i_mul(out->sp_up.s3, -s*mu03_BSM, in->sp_up.s3);


  /* out_dn +=- s * i \gamma_5 \mu3 * in_dn */
  _vector_add_i_mul(out->sp_dn.s0, -s*mu03_BSM, in->sp_dn.s0);
  _vector_add_i_mul(out->sp_dn.s1, -s*mu03_BSM, in->sp_dn.s1);
  _vector_add_i_mul(out->sp_dn.s2,  s*mu03_BSM, in->sp_dn.s2);
  _vector_add_i_mul(out->sp_dn.s3,  s*mu03_BSM, in->sp_dn.s3);

}
static inline void tm1_add(bispinor * const out, const bispinor * const in, const double sign)
{
  /*out+=s*i\gamma_5 \tau_1 mu1 *in
   * sign>0 for D+i\gamma_5\tau_1
   * sign<0 for D_dag-i\gamma_5\tau_1
   */
  const double s = (sign < 0) ? -1. : 1.;

  /* out_up += s * i \gamma_5 \mu1 * in_dn */
  _vector_add_i_mul(out->sp_up.s0,  s*mu01_BSM, in->sp_dn.s0);
  _vector_add_i_mul(out->sp_up.s1,  s*mu01_BSM, in->sp_dn.s1);
  _vector_add_i_mul(out->sp_up.s2, -s*mu01_BSM, in->sp_dn.s2);
  _vector_add_i_mul(out->sp_up.s3, -s*mu01_BSM, in->sp_dn.s3);


  /* out_dn += s * i \gamma_5 \mu1 * in_up */
  _vector_add_i_mul(out->sp_dn.s0,  s*mu01_BSM, in->sp_up.s0);
  _vector_add_i_mul(out->sp_dn.s1,  s*mu01_BSM, in->sp_up.s1);
  _vector_add_i_mul(out->sp_dn.s2, -s*mu01_BSM, in->sp_up.s2);
  _vector_add_i_mul(out->sp_dn.s3, -s*mu01_BSM, in->sp_up.s3);

}


/* operation out(x) += Fabs(y)*in(x)
 * Fabs(y) := [ \phi_0(y)**2 + \sum_j \phi_j(y)**2 ] * c
 * this operator acts locally on a site x, pass pointers accordingly.
 * out: the resulting bispinor, out += F*in
 * in:  the input bispinor at site x
 * phi: pointer to the four scalars phi0,...,phi3 at site y, y = x or x+-\mu
 * c:    constant double
 */

static inline void Fabsadd(bispinor * const out, const bispinor * const in, const scalar * const phi, const double c) {
  const double c_phisq = c*(phi[0]*phi[0]+phi[1]*phi[1]+phi[2]*phi[2]+phi[3]*phi[3]);

  // flavour 1:
  // out_up += c(\phi_0 \phi_0 + \phi_1 \phi_1 + \phi_2 \phi_2+ \phi_3 \phi_3)* in_up
  _vector_add_mul(out->sp_up.s0, c_phisq, in->sp_up.s0);
  _vector_add_mul(out->sp_up.s1, c_phisq, in->sp_up.s1);
  _vector_add_mul(out->sp_up.s2, c_phisq, in->sp_up.s2);
  _vector_add_mul(out->sp_up.s3, c_phisq, in->sp_up.s3);

  // flavour 2:
  // out_dn += c(\phi_0 \phi_0 + \phi_1 \phi_1 + \phi_2 \phi_2+ \phi_3 \phi_3)* in_dn
  _vector_add_mul(out->sp_dn.s0, c_phisq, in->sp_dn.s0);
  _vector_add_mul(out->sp_dn.s1, c_phisq, in->sp_dn.s1);
  _vector_add_mul(out->sp_dn.s2, c_phisq, in->sp_dn.s2);
  _vector_add_mul(out->sp_dn.s3, c_phisq, in->sp_dn.s3);

}



/* operation out(x) += F(y)*in(x)
 * F(y) := [ \phi_0(y) + i \gamma_5 \tau^j \phi_j(y) ] * c
 * this operator acts locally on a site x, pass pointers accordingly.
 * out: the resulting bispinor, out += F*in
 * in:  the input bispinor at site x
 * phi: pointer to the four scalars phi0,...,phi3 at site y, y = x or x+-\mu
 * c:    constant double
 *
 * sign = +1 -> Fadd
 * sign = -1 -> Fbaradd
 */

static inline void Fadd(bispinor * const out, const bispinor * const in, const scalar * const phi, const double c, const double sign) {
#ifdef TM_USE_OMP
#define static
#endif
  static spinor tmp;
#ifdef TM_USE_OMP
#undef static
#endif
  
  const double s = (sign < 0) ? -1. : 1.;

  // flavour 1:
  // tmp_up = \phi_0 * in_up
  _vector_mul(tmp.s0, phi[0], in->sp_up.s0);
  _vector_mul(tmp.s1, phi[0], in->sp_up.s1);
  _vector_mul(tmp.s2, phi[0], in->sp_up.s2);
  _vector_mul(tmp.s3, phi[0], in->sp_up.s3);
  
  // tmp_up += s * i \gamma_5 \phi_1 * in_dn
  _vector_add_i_mul(tmp.s0,  s*phi[1], in->sp_dn.s0);
  _vector_add_i_mul(tmp.s1,  s*phi[1], in->sp_dn.s1);
  _vector_add_i_mul(tmp.s2, -s*phi[1], in->sp_dn.s2);
  _vector_add_i_mul(tmp.s3, -s*phi[1], in->sp_dn.s3);
  
  // tmp_up += s * \gamma_5 \phi_2 * in_dn
  _vector_add_mul(tmp.s0,  s*phi[2], in->sp_dn.s0);
  _vector_add_mul(tmp.s1,  s*phi[2], in->sp_dn.s1);
  _vector_add_mul(tmp.s2, -s*phi[2], in->sp_dn.s2);
  _vector_add_mul(tmp.s3, -s*phi[2], in->sp_dn.s3);
  
  // tmp_up += s * i \gamma_5 \phi_3 * in_up
  _vector_add_i_mul(tmp.s0,  s*phi[3], in->sp_up.s0);
  _vector_add_i_mul(tmp.s1,  s*phi[3], in->sp_up.s1);
  _vector_add_i_mul(tmp.s2, -s*phi[3], in->sp_up.s2);
  _vector_add_i_mul(tmp.s3, -s*phi[3], in->sp_up.s3);
  
  // out_up += c * tmp;
  _vector_add_mul(out->sp_up.s0,c,tmp.s0);
  _vector_add_mul(out->sp_up.s1,c,tmp.s1);
  _vector_add_mul(out->sp_up.s2,c,tmp.s2);
  _vector_add_mul(out->sp_up.s3,c,tmp.s3);
  
  
  // flavour 2:
  // tmp_dn = \phi_0 * in_dn
  _vector_mul(tmp.s0, phi[0], in->sp_dn.s0);
  _vector_mul(tmp.s1, phi[0], in->sp_dn.s1);
  _vector_mul(tmp.s2, phi[0], in->sp_dn.s2);
  _vector_mul(tmp.s3, phi[0], in->sp_dn.s3);
  
  // tmp_dn += s * i \gamma_5 \phi_1 * in_up
  _vector_add_i_mul(tmp.s0,  s*phi[1], in->sp_up.s0);
  _vector_add_i_mul(tmp.s1,  s*phi[1], in->sp_up.s1);
  _vector_add_i_mul(tmp.s2, -s*phi[1], in->sp_up.s2);
  _vector_add_i_mul(tmp.s3, -s*phi[1], in->sp_up.s3);
  
  // tmp_dn -= s * \gamma_5 \phi_2 * in_up
  _vector_add_mul(tmp.s0, -s*phi[2], in->sp_up.s0);
  _vector_add_mul(tmp.s1, -s*phi[2], in->sp_up.s1);
  _vector_add_mul(tmp.s2,  s*phi[2], in->sp_up.s2);
  _vector_add_mul(tmp.s3,  s*phi[2], in->sp_up.s3);
  
  // tmp_dn -= s * i \gamma_5 \phi_3 * in_dn
  _vector_add_i_mul(tmp.s0, -s*phi[3], in->sp_dn.s0);
  _vector_add_i_mul(tmp.s1, -s*phi[3], in->sp_dn.s1);
  _vector_add_i_mul(tmp.s2,  s*phi[3], in->sp_dn.s2);
  _vector_add_i_mul(tmp.s3,  s*phi[3], in->sp_dn.s3);
  
  // out_dn += c * tmp;
  _vector_add_mul(out->sp_dn.s0,c,tmp.s0);
  _vector_add_mul(out->sp_dn.s1,c,tmp.s1);
  _vector_add_mul(out->sp_dn.s2,c,tmp.s2);
  _vector_add_mul(out->sp_dn.s3,c,tmp.s3);
}

static inline void bispinor_times_phase_times_u(bispinor * restrict const us, const _Complex double phase,
						su3 const * restrict const u, bispinor const * restrict const s)
{
#ifdef TM_USE_OMP
#define static
#endif
  static su3_vector chi;
#ifdef TM_USE_OMP
#undef static
#endif

  _su3_multiply(chi, (*u), s->sp_up.s0);
  _complex_times_vector(us->sp_up.s0, phase, chi);

  _su3_multiply(chi, (*u), s->sp_up.s1);
  _complex_times_vector(us->sp_up.s1, phase, chi);

  _su3_multiply(chi, (*u), s->sp_up.s2);
  _complex_times_vector(us->sp_up.s2, phase, chi);

  _su3_multiply(chi, (*u), s->sp_up.s3);
  _complex_times_vector(us->sp_up.s3, phase, chi);

  _su3_multiply(chi, (*u), s->sp_dn.s0);
  _complex_times_vector(us->sp_dn.s0, phase, chi);

  _su3_multiply(chi, (*u), s->sp_dn.s1);
  _complex_times_vector(us->sp_dn.s1, phase, chi);

  _su3_multiply(chi, (*u), s->sp_dn.s2);
  _complex_times_vector(us->sp_dn.s2, phase, chi);

  _su3_multiply(chi, (*u), s->sp_dn.s3);
  _complex_times_vector(us->sp_dn.s3, phase, chi);

  return;
}


static inline void bispinor_times_phase_times_inverse_u(bispinor * restrict const us, const _Complex double phase,
							su3 const * restrict const u, bispinor const * restrict const s)
{
#ifdef TM_USE_OMP
#define static
#endif
  static su3_vector chi;
#ifdef TM_USE_OMP
#undef static
#endif

  _su3_inverse_multiply(chi, (*u), s->sp_up.s0);
  _complexcjg_times_vector(us->sp_up.s0, phase, chi);

  _su3_inverse_multiply(chi, (*u), s->sp_up.s1);
  _complexcjg_times_vector(us->sp_up.s1, phase, chi);

  _su3_inverse_multiply(chi, (*u), s->sp_up.s2);
  _complexcjg_times_vector(us->sp_up.s2, phase, chi);

  _su3_inverse_multiply(chi, (*u), s->sp_up.s3);
  _complexcjg_times_vector(us->sp_up.s3, phase, chi);

  _su3_inverse_multiply(chi, (*u), s->sp_dn.s0);
  _complexcjg_times_vector(us->sp_dn.s0, phase, chi);

  _su3_inverse_multiply(chi, (*u), s->sp_dn.s1);
  _complexcjg_times_vector(us->sp_dn.s1, phase, chi);

  _su3_inverse_multiply(chi, (*u), s->sp_dn.s2);
  _complexcjg_times_vector(us->sp_dn.s2, phase, chi);

  _su3_inverse_multiply(chi, (*u), s->sp_dn.s3);
  _complexcjg_times_vector(us->sp_dn.s3, phase, chi);

  return;
}

static inline void bispinor_times_real_times_inverse_u(bispinor * restrict const us, const double realnum,
                                                        su3 const * restrict const u, bispinor const * restrict const s)
{
#ifdef TM_USE_OMP
#define static
#endif
  static su3_vector chi;
#ifdef TM_USE_OMP
#undef static
#endif


  _su3_inverse_multiply(chi, (*u), s->sp_up.s0);
  _vector_null( us->sp_up.s0 );
  _vector_add_mul( us->sp_up.s0, realnum, chi );

  _su3_inverse_multiply(chi, (*u), s->sp_up.s1);
  _vector_null( us->sp_up.s1 );
  _vector_add_mul( us->sp_up.s1, realnum, chi );

  _su3_inverse_multiply(chi, (*u), s->sp_up.s2);
  _vector_null( us->sp_up.s2 );
  _vector_add_mul( us->sp_up.s2, realnum, chi );

  _su3_inverse_multiply(chi, (*u), s->sp_up.s3);
  _vector_null( us->sp_up.s3 );
  _vector_add_mul( us->sp_up.s3, realnum, chi );

  _su3_inverse_multiply(chi, (*u), s->sp_dn.s0);
  _vector_null( us->sp_dn.s0 );
  _vector_add_mul( us->sp_dn.s0, realnum, chi );

  _su3_inverse_multiply(chi, (*u), s->sp_dn.s1);
  _vector_null( us->sp_dn.s1 );
  _vector_add_mul( us->sp_dn.s1, realnum, chi );

  _su3_inverse_multiply(chi, (*u), s->sp_dn.s2);
  _vector_null( us->sp_dn.s2 );
  _vector_add_mul( us->sp_dn.s2, realnum, chi );

  _su3_inverse_multiply(chi, (*u), s->sp_dn.s3);
  _vector_null( us->sp_dn.s3 );
  _vector_add_mul( us->sp_dn.s3, realnum, chi );

  return;
}


//This takes care of -rho/4*F(\tilde x)*Z(tilde)(tilde x) psi(tilde x)
//One has to set the location of the field beforehand correctly here they are just vectors
//in spinor color and flavor space
//note the -factor is implemented through 0.5 in both phasef and phase -->>altogether a factor of 1/4
static inline void padd_chitildebreak(bispinor * restrict const tmpr , bispinor const * restrict const s,
                         su3 const * restrict const u, const int inv, const _Complex double phase,
                         const double phaseF, const scalar * const phi, const scalar * const phip,
                         const double sign) {

#ifdef TM_USE_OMP
#define static
#endif
  static bispinor us;
#ifdef TM_USE_OMP
#undef static
#endif

  // us = phase*u*s
  ( inv == 0 ) ? bispinor_times_phase_times_u(&us, phase, u, s) : bispinor_times_phase_times_inverse_u(&us, phase, u, s);

  // FIXME: signs of terms local and non-local in phi should be different
  // tmpr += F*us
  Fadd(tmpr, &us, phi,  phaseF, sign);
  Fadd(tmpr, &us, phip, phaseF, sign);

  // tmpr += b*us

  return;
}


static inline void p0add_wilsonclover( bispinor * restrict const tmpr , bispinor const * restrict const sp,
                         su3 const * restrict const u, const int inv, const _Complex double phase,
                         const double sign) {
#ifdef TM_USE_OMP
#define static
#endif
// FIXME check and fix sign here once and for all ;) 
  const int sign_gamma = (inv==1) ? -sign : sign ;
  static su3_vector halfwilson1;
  static su3_vector halfwilson2;
  static su3_vector chi;
  static su3_vector results1;
  static su3_vector results2;
#ifdef TM_USE_OMP
#undef static
#endif
  _vector_null( halfwilson1 );
  _vector_null( halfwilson2 );
  if(sign_gamma == 1){
//Performing the multiplication on the first half of a halfspinor
//shrink the fermion vector from four spin component to two
//first component
    _vector_add(     halfwilson1, sp->sp_up.s0, sp->sp_up.s2);
    _vector_add(     halfwilson2, sp->sp_dn.s0, sp->sp_dn.s2);
    if(inv == 1){
      _su3_inverse_multiply(chi, (*u), halfwilson1);
      _complexcjg_times_vector(results1, phase, chi);
      _su3_inverse_multiply(chi, (*u), halfwilson2);
      _complexcjg_times_vector(results2, phase, chi);
    }
    else{
      _su3_multiply(chi, (*u), halfwilson1);
      _complex_times_vector(results1, phase, chi);
      _su3_multiply(chi, (*u), halfwilson2);
      _complex_times_vector(results2, phase, chi);
    }

    _vector_add_assign( tmpr->sp_up.s0, results1);
    _vector_add_assign( tmpr->sp_up.s2, results1);
    _vector_add_assign( tmpr->sp_dn.s0, results2);
    _vector_add_assign( tmpr->sp_dn.s2, results2);

//Performing the multiplication on the first half of a halfspinor
//shrink the fermion vector from four spin component to two
//second component
    _vector_add(     halfwilson1, sp->sp_up.s1, sp->sp_up.s3 );
    _vector_add(     halfwilson2, sp->sp_dn.s1, sp->sp_dn.s3 );

    if(inv == 1){
      _su3_inverse_multiply(chi, (*u), halfwilson1);
      _complexcjg_times_vector(results1, phase, chi);
      _su3_inverse_multiply(chi, (*u), halfwilson2);
      _complexcjg_times_vector(results2, phase, chi);
    }
    else{
      _su3_multiply(chi, (*u), halfwilson1);
      _complex_times_vector(results1, phase, chi);
      _su3_multiply(chi, (*u), halfwilson2);
      _complex_times_vector(results2, phase, chi);
    }
    _vector_add_assign( tmpr->sp_up.s1, results1);
    _vector_add_assign( tmpr->sp_up.s3, results1);
    _vector_add_assign( tmpr->sp_dn.s1, results2);
    _vector_add_assign( tmpr->sp_dn.s3, results2);
  }//end of if sign_gamma==1
  else{
//Performing the multiplication on the first half of a halfspinor
//shrink the fermion vector from four spin component to two
//first component
    _vector_assign(     halfwilson1, sp->sp_up.s0);
    _vector_assign(     halfwilson2, sp->sp_dn.s0);
    _vector_sub_assign( halfwilson1, sp->sp_up.s2);
    _vector_sub_assign( halfwilson2, sp->sp_dn.s2);
//multiply the shrinked fermion vector with the gauge 
    if(inv == 1){
      _su3_inverse_multiply(chi, (*u), halfwilson1);
      _complexcjg_times_vector(results1, phase, chi);
      _su3_inverse_multiply(chi, (*u), halfwilson2);
      _complexcjg_times_vector(results2, phase, chi);
    }
    else{
      _su3_multiply(chi, (*u), halfwilson1);
      _complex_times_vector(results1, phase, chi);
      _su3_multiply(chi, (*u), halfwilson2);
      _complex_times_vector(results2, phase, chi);
    }
//expand it
    _vector_add_assign( tmpr->sp_up.s0, results1);
    _vector_sub_assign( tmpr->sp_up.s2, results1);
    _vector_add_assign( tmpr->sp_dn.s0, results2);
    _vector_sub_assign( tmpr->sp_dn.s2, results2);

//Performing the multiplication on the first half of a halfspinor
//shrink the fermion vector from four spin component to two
//second component
    _vector_assign(     halfwilson1, sp->sp_up.s1);
    _vector_assign(     halfwilson2, sp->sp_dn.s1);
    _vector_sub_assign( halfwilson1, sp->sp_up.s3);
    _vector_sub_assign( halfwilson2, sp->sp_dn.s3);
//multiply the shrinked fermion vector with the gauge 
    if(inv == 1){
      _su3_inverse_multiply(chi, (*u), halfwilson1);
      _complexcjg_times_vector(results1, phase, chi);
      _su3_inverse_multiply(chi, (*u), halfwilson2);
      _complexcjg_times_vector(results2, phase, chi);
    }
    else{
      _su3_multiply(chi, (*u), halfwilson1);
      _complex_times_vector(results1, phase, chi);
      _su3_multiply(chi, (*u), halfwilson2);
      _complex_times_vector(results2, phase, chi);
    }
//expand it
    _vector_add_assign( tmpr->sp_up.s1, results1);
    _vector_sub_assign( tmpr->sp_up.s3, results1);
    _vector_add_assign( tmpr->sp_dn.s1, results2);
    _vector_sub_assign( tmpr->sp_dn.s3, results2);
  }
  return;
      
}
static inline void p1add_wilsonclover( bispinor * restrict const tmpr , bispinor const * restrict const sp,
                         su3 const * restrict const u, const int inv, const _Complex double phase,
                         const double sign) {
#ifdef TM_USE_OMP
#define static
#endif
  const int sign_gamma = (inv==1) ? -sign : sign ;
  static su3_vector halfwilson1;
  static su3_vector halfwilson2;
  static su3_vector chi;
  static su3_vector results1;
  static su3_vector results2;
#ifdef TM_USE_OMP
#undef static
#endif
  _vector_null( halfwilson1 );
  _vector_null( halfwilson2 );
  if(sign_gamma == 1){
//Performing the multiplication on the first half of a halfspinor
//shrink the fermion vector from four spin component to two
//first component
    _vector_assign(     halfwilson1, sp->sp_up.s0);
    _vector_assign(     halfwilson2, sp->sp_dn.s0);
    _vector_add_i_assign( halfwilson1, sp->sp_up.s3);
    _vector_add_i_assign( halfwilson2, sp->sp_dn.s3);
    if(inv == 1){
      _su3_inverse_multiply(chi, (*u), halfwilson1);
      _complexcjg_times_vector(results1, phase, chi);
      _su3_inverse_multiply(chi, (*u), halfwilson2);
      _complexcjg_times_vector(results2, phase, chi);
    }
    else{
      _su3_multiply(chi, (*u), halfwilson1);
      _complex_times_vector(results1, phase, chi);
      _su3_multiply(chi, (*u), halfwilson2);
      _complex_times_vector(results2, phase, chi);
    }

    _vector_add_assign(   tmpr->sp_up.s0, results1);
    _vector_sub_i_assign( tmpr->sp_up.s3, results1);
    _vector_add_assign(   tmpr->sp_dn.s0, results2);
    _vector_sub_i_assign( tmpr->sp_dn.s3, results2);

//Performing the multiplication on the first half of a halfspinor
//shrink the fermion vector from four spin component to two
//second component
    _vector_assign(     halfwilson1, sp->sp_up.s1);
    _vector_assign(     halfwilson2, sp->sp_dn.s1);
    _vector_add_i_assign( halfwilson1, sp->sp_up.s2);
    _vector_add_i_assign( halfwilson2, sp->sp_dn.s2);

    if(inv == 1){
      _su3_inverse_multiply(chi, (*u), halfwilson1);
      _complexcjg_times_vector(results1, phase, chi);
      _su3_inverse_multiply(chi, (*u), halfwilson2);
      _complexcjg_times_vector(results2, phase, chi);
    }
    else{
      _su3_multiply(chi, (*u), halfwilson1);
      _complex_times_vector(results1, phase, chi);
      _su3_multiply(chi, (*u), halfwilson2);
      _complex_times_vector(results2, phase, chi);
    }
    _vector_add_assign( tmpr->sp_up.s1, results1);
    _vector_sub_i_assign( tmpr->sp_up.s2, results1);
    _vector_add_assign( tmpr->sp_dn.s1, results2);
    _vector_sub_i_assign( tmpr->sp_dn.s2, results2);
  }//end of if sign_gamma==1
  else{
//Performing the multiplication on the first half of a halfspinor
//shrink the fermion vector from four spin component to two
//first component
    _vector_assign(       halfwilson1, sp->sp_up.s0);
    _vector_assign(       halfwilson2, sp->sp_dn.s0);
    _vector_sub_i_assign( halfwilson1, sp->sp_up.s3);
    _vector_sub_i_assign( halfwilson2, sp->sp_dn.s3);
//multiply the shrinked fermion vector with the gauge 
    if(inv == 1){
      _su3_inverse_multiply(chi, (*u), halfwilson1);
      _complexcjg_times_vector(results1, phase, chi);
      _su3_inverse_multiply(chi, (*u), halfwilson2);
      _complexcjg_times_vector(results2, phase, chi);
    }
    else{
      _su3_multiply(chi, (*u), halfwilson1);
      _complex_times_vector(results1, phase, chi);
      _su3_multiply(chi, (*u), halfwilson2);
      _complex_times_vector(results2, phase, chi);
    }
//expand it
    _vector_add_assign(   tmpr->sp_up.s0, results1);
    _vector_add_i_assign( tmpr->sp_up.s3, results1);
    _vector_add_assign(   tmpr->sp_dn.s0, results2);
    _vector_add_i_assign( tmpr->sp_dn.s3, results2);

//Performing the multiplication on the first half of a halfspinor
//shrink the fermion vector from four spin component to two
//second component
    _vector_assign(     halfwilson1, sp->sp_up.s1);
    _vector_assign(     halfwilson2, sp->sp_dn.s1);
    _vector_sub_i_assign( halfwilson1, sp->sp_up.s2);
    _vector_sub_i_assign( halfwilson2, sp->sp_dn.s2);
//multiply the shrinked fermion vector with the gauge 
    if(inv == 1){
      _su3_inverse_multiply(chi, (*u), halfwilson1);
      _complexcjg_times_vector(results1, phase, chi);
      _su3_inverse_multiply(chi, (*u), halfwilson2);
      _complexcjg_times_vector(results2, phase, chi);
    }
    else{
      _su3_multiply(chi, (*u), halfwilson1);
      _complex_times_vector(results1, phase, chi);
      _su3_multiply(chi, (*u), halfwilson2);
      _complex_times_vector(results2, phase, chi);
    }
//expand it
    _vector_add_assign(   tmpr->sp_up.s1, results1);
    _vector_add_i_assign( tmpr->sp_up.s2, results1);
    _vector_add_assign(   tmpr->sp_dn.s1, results2);
    _vector_add_i_assign( tmpr->sp_dn.s2, results2);
  }


  return;

}


static inline void p2add_wilsonclover( bispinor * restrict const tmpr , bispinor const * restrict const sp,
                         su3 const * restrict const u, const int inv, const _Complex double phase,
                         const double sign) {
#ifdef TM_USE_OMP
#define static
#endif
  const int sign_gamma = (inv==1) ? -sign : sign ;
  static su3_vector halfwilson1;
  static su3_vector halfwilson2;
  static su3_vector chi;
  static su3_vector results1;
  static su3_vector results2;
#ifdef TM_USE_OMP
#undef static
#endif
  _vector_null( halfwilson1 );
  _vector_null( halfwilson2 );
  if(sign_gamma == 1){
//Performing the multiplication on the first half of a halfspinor
//shrink the fermion vector from four spin component to two
//first component
    _vector_assign(     halfwilson1, sp->sp_up.s0);
    _vector_assign(     halfwilson2, sp->sp_dn.s0);
    _vector_add_assign( halfwilson1, sp->sp_up.s3);
    _vector_add_assign( halfwilson2, sp->sp_dn.s3);
    if(inv == 1){
      _su3_inverse_multiply(chi, (*u), halfwilson1);
      _complexcjg_times_vector(results1, phase, chi);
      _su3_inverse_multiply(chi, (*u), halfwilson2);
      _complexcjg_times_vector(results2, phase, chi);
    }
    else{
      _su3_multiply(chi, (*u), halfwilson1);
      _complex_times_vector(results1, phase, chi);
      _su3_multiply(chi, (*u), halfwilson2);
      _complex_times_vector(results2, phase, chi);
    }

    _vector_add_assign(   tmpr->sp_up.s0, results1);
    _vector_add_assign(   tmpr->sp_up.s3, results1);
    _vector_add_assign(   tmpr->sp_dn.s0, results2);
    _vector_add_assign(   tmpr->sp_dn.s3, results2);

//Performing the multiplication on the first half of a halfspinor
//shrink the fermion vector from four spin component to two
//second component
    _vector_assign(     halfwilson1, sp->sp_up.s1);
    _vector_assign(     halfwilson2, sp->sp_dn.s1);
    _vector_sub_assign( halfwilson1, sp->sp_up.s2);
    _vector_sub_assign( halfwilson2, sp->sp_dn.s2);

    if(inv == 1){
      _su3_inverse_multiply(chi, (*u), halfwilson1);
      _complexcjg_times_vector(results1, phase, chi);
      _su3_inverse_multiply(chi, (*u), halfwilson2);
      _complexcjg_times_vector(results2, phase, chi);
    }
    else{
      _su3_multiply(chi, (*u), halfwilson1);
      _complex_times_vector(results1, phase, chi);
      _su3_multiply(chi, (*u), halfwilson2);
      _complex_times_vector(results2, phase, chi);
    }
    _vector_add_assign( tmpr->sp_up.s1, results1);
    _vector_sub_assign( tmpr->sp_up.s2, results1);
    _vector_add_assign( tmpr->sp_dn.s1, results2);
    _vector_sub_assign( tmpr->sp_dn.s2, results2);
  }//end of if sign_gamma==1
  else{
//Performing the multiplication on the first half of a halfspinor
//shrink the fermion vector from four spin component to two
//first component
    _vector_assign(       halfwilson1, sp->sp_up.s0);
    _vector_assign(       halfwilson2, sp->sp_dn.s0);
    _vector_sub_assign(   halfwilson1, sp->sp_up.s3);
    _vector_sub_assign(   halfwilson2, sp->sp_dn.s3);
//multiply the shrinked fermion vector with the gauge 
    if(inv == 1){
      _su3_inverse_multiply(chi, (*u), halfwilson1);
      _complexcjg_times_vector(results1, phase, chi);
      _su3_inverse_multiply(chi, (*u), halfwilson2);
      _complexcjg_times_vector(results2, phase, chi);
    }
    else{
      _su3_multiply(chi, (*u), halfwilson1);
      _complex_times_vector(results1, phase, chi);
      _su3_multiply(chi, (*u), halfwilson2);
      _complex_times_vector(results2, phase, chi);
    }
//expand it
    _vector_add_assign(   tmpr->sp_up.s0, results1);
    _vector_sub_assign(   tmpr->sp_up.s3, results1);
    _vector_add_assign(   tmpr->sp_dn.s0, results2);
    _vector_sub_assign(   tmpr->sp_dn.s3, results2);

//Performing the multiplication on the first half of a halfspinor
//shrink the fermion vector from four spin component to two
//second component
    _vector_assign(     halfwilson1, sp->sp_up.s1);
    _vector_assign(     halfwilson2, sp->sp_dn.s1);
    _vector_add_assign( halfwilson1, sp->sp_up.s2);
    _vector_add_assign( halfwilson2, sp->sp_dn.s2);
//multiply the shrinked fermion vector with the gauge 
    if(inv == 1){
      _su3_inverse_multiply(chi, (*u), halfwilson1);
      _complexcjg_times_vector(results1, phase, chi);
      _su3_inverse_multiply(chi, (*u), halfwilson2);
      _complexcjg_times_vector(results2, phase, chi);
    }
    else{
      _su3_multiply(chi, (*u), halfwilson1);
      _complex_times_vector(results1, phase, chi);
      _su3_multiply(chi, (*u), halfwilson2);
      _complex_times_vector(results2, phase, chi);
    }
//expand it
    _vector_add_assign(   tmpr->sp_up.s1, results1);
    _vector_add_assign(   tmpr->sp_up.s2, results1);
    _vector_add_assign(   tmpr->sp_dn.s1, results2);
    _vector_add_assign(   tmpr->sp_dn.s2, results2);
  }

  return;

}



static inline void p3add_wilsonclover( bispinor * restrict const tmpr , bispinor const * restrict const sp,
                         su3 const * restrict const u, const int inv, const _Complex double phase,
                         const double sign) {
#ifdef TM_USE_OMP
#define static
#endif
  const int sign_gamma = (inv==1) ? -sign : sign ;
  static su3_vector halfwilson1;
  static su3_vector halfwilson2;
  static su3_vector chi;
  static su3_vector results1;
  static su3_vector results2;
#ifdef TM_USE_OMP
#undef static
#endif
  if(sign_gamma == 1){
//Performing the multiplication on the first half of a halfspinor
//shrink the fermion vector from four spin component to two
//first component
    _vector_assign(       halfwilson1, sp->sp_up.s0);
    _vector_assign(       halfwilson2, sp->sp_dn.s0);
    _vector_add_i_assign( halfwilson1, sp->sp_up.s2);
    _vector_add_i_assign( halfwilson2, sp->sp_dn.s2);
    if(inv == 1){
      _su3_inverse_multiply(chi, (*u), halfwilson1);
      _complexcjg_times_vector(results1, phase, chi);
      _su3_inverse_multiply(chi, (*u), halfwilson2);
      _complexcjg_times_vector(results2, phase, chi);
    }
    else{
      _su3_multiply(chi, (*u), halfwilson1);
      _complex_times_vector(results1, phase, chi);
      _su3_multiply(chi, (*u), halfwilson2);
      _complex_times_vector(results2, phase, chi);
    }

    _vector_add_assign(     tmpr->sp_up.s0, results1);
    _vector_add_assign(     tmpr->sp_dn.s0, results2);
    _vector_sub_i_assign(   tmpr->sp_up.s2, results1);
    _vector_sub_i_assign(   tmpr->sp_dn.s2, results2);

//Performing the multiplication on the first half of a halfspinor
//shrink the fermion vector from four spin component to two
//second component
    _vector_assign(       halfwilson1, sp->sp_up.s1);
    _vector_assign(       halfwilson2, sp->sp_dn.s1);
    _vector_sub_i_assign( halfwilson1, sp->sp_up.s3);
    _vector_sub_i_assign( halfwilson2, sp->sp_dn.s3);

    if(inv == 1){
      _su3_inverse_multiply(chi, (*u), halfwilson1);
      _complexcjg_times_vector(results1, phase, chi);
      _su3_inverse_multiply(chi, (*u), halfwilson2);
      _complexcjg_times_vector(results2, phase, chi);
    }
    else{
      _su3_multiply(chi, (*u), halfwilson1);
      _complex_times_vector(results1, phase, chi);
      _su3_multiply(chi, (*u), halfwilson2);
      _complex_times_vector(results2, phase, chi);
    }
    _vector_add_assign(   tmpr->sp_up.s1, results1);
    _vector_add_i_assign( tmpr->sp_up.s3, results1);
    _vector_add_assign(   tmpr->sp_dn.s1, results2);
    _vector_add_i_assign( tmpr->sp_dn.s3, results2);
  }//end of if sign_gamma==1
  else{
//Performing the multiplication on the first half of a halfspinor
//shrink the fermion vector from four spin component to two
//first component
    _vector_assign(         halfwilson1, sp->sp_up.s0);
    _vector_assign(         halfwilson2, sp->sp_dn.s0);
    _vector_sub_i_assign(   halfwilson1, sp->sp_up.s2);
    _vector_sub_i_assign(   halfwilson2, sp->sp_dn.s2);
//multiply the shrinked fermion vector with the gauge 
    if(inv == 1){
      _su3_inverse_multiply(chi, (*u), halfwilson1);
      _complexcjg_times_vector(results1, phase, chi);
      _su3_inverse_multiply(chi, (*u), halfwilson2);
      _complexcjg_times_vector(results2, phase, chi);
    }
    else{
      _su3_multiply(chi, (*u), halfwilson1);
      _complex_times_vector(results1, phase, chi);
      _su3_multiply(chi, (*u), halfwilson2);
      _complex_times_vector(results2, phase, chi);
    }
//expand it
    _vector_add_assign(     tmpr->sp_up.s0, results1);
    _vector_add_i_assign(   tmpr->sp_up.s2, results1);
    _vector_add_assign(     tmpr->sp_dn.s0, results2);
    _vector_add_i_assign(   tmpr->sp_dn.s2, results2);

//Performing the multiplication on the first half of a halfspinor
//shrink the fermion vector from four spin component to two
//second component
    _vector_assign(       halfwilson1, sp->sp_up.s1);
    _vector_assign(       halfwilson2, sp->sp_dn.s1);
    _vector_add_i_assign( halfwilson1, sp->sp_up.s3);
    _vector_add_i_assign( halfwilson2, sp->sp_dn.s3);
//multiply the shrinked fermion vector with the gauge 
    if(inv == 1){
      _su3_inverse_multiply(chi, (*u), halfwilson1);
      _complexcjg_times_vector(results1, phase, chi);
      _su3_inverse_multiply(chi, (*u), halfwilson2);
      _complexcjg_times_vector(results2, phase, chi);
    }
    else{
      _su3_multiply(chi, (*u), halfwilson1);
      _complex_times_vector(results1, phase, chi);
      _su3_multiply(chi, (*u), halfwilson2);
      _complex_times_vector(results2, phase, chi);
    }
//expand it
    _vector_add_assign(       tmpr->sp_up.s1, results1);
    _vector_sub_i_assign(     tmpr->sp_up.s3, results1);
    _vector_add_assign(       tmpr->sp_dn.s1, results2);
    _vector_sub_i_assign(     tmpr->sp_dn.s3, results2);
  }

  return;

}

/* D_psi_BSM3 acts on bispinor fields 
 * version meant for production uses two 
 * different gauge fields: smeared one
 * for the fermionic kinetic term and 
 * unsmeared for the chitilde breaking
 * terms */
void D_psi_BSM3(bispinor * const P, bispinor * const Q){
  if(P==Q){
    printf("Error in D_psi_BSM (D_psi_BSM.c):\n");
    printf("Arguments must be different bispinor fields\n");
    printf("Program aborted\n");
    exit(1);
  }
#ifdef TM_USE_MPI
  generic_exchange(Q, sizeof(bispinor));
#endif

#ifdef TM_USE_OMP
#pragma omp parallel
  {
#endif

    int ix,iy;                       /* x, x+-\mu */
    su3 * restrict up,* restrict um; /* U_\mu(x), U_\mu(x-\mu) */
    bispinor * restrict rr;          /* P(x) */
    bispinor const * restrict s;     /* Q(x) */
    bispinor const * restrict sp;    /* Q(x+\mu) */
    bispinor const * restrict sm;    /* Q(x-\mu) */
    scalar phi[4];                   /* phi_i(x) */
    scalar phip[4][4];               /* phi_i(x+mu) = phip[mu][i] */
    scalar phim[4][4];               /* phi_i(x-mu) = phim[mu][i] */
    const su3 *w1,*w2,*w3;
    _Complex double rho1, rho2;
    rho1 = 0.5*(1. +  mu03_BSM * I);
    rho2 = conj(rho1);




    /************************ loop over all lattice sites *************************/

#ifdef TM_USE_OMP
#pragma omp for
#endif
    for (ix=0;ix<VOLUME;ix++)
      {
        rr = (bispinor *) P + ix;
        s  = (bispinor *) Q + ix;

        /* prefatch scalar fields */
        phi[0] = g_scalar_field[0][ix];
        phi[1] = g_scalar_field[1][ix];
        phi[2] = g_scalar_field[2][ix];
        phi[3] = g_scalar_field[3][ix];

        for( int mu=0; mu<4; mu++ )
          {
            phip[mu][0] = g_scalar_field[0][g_iup[ix][mu]];
            phip[mu][1] = g_scalar_field[1][g_iup[ix][mu]];
            phip[mu][2] = g_scalar_field[2][g_iup[ix][mu]];
            phip[mu][3] = g_scalar_field[3][g_iup[ix][mu]];

            phim[mu][0] = g_scalar_field[0][g_idn[ix][mu]];
            phim[mu][1] = g_scalar_field[1][g_idn[ix][mu]];
            phim[mu][2] = g_scalar_field[2][g_idn[ix][mu]];
            phim[mu][3] = g_scalar_field[3][g_idn[ix][mu]];
          }


        /* the local part (not local in phi) */

        _spinor_null(rr->sp_up);
        _spinor_null(rr->sp_dn);
        
        // FIXME check normalisation of twisted mass and csw terms
        // Note that here what is implemented is i*1/2*c_sw*sigma_mu,nu F_mu,nu
        // thus we need to apply and additional factor of 1/2 preparing the 
        // clover term, this is done when we initialize the clover term in 
        // operator.c for executable invert and in contractions_BSM.c for exec
        // contractions_BSM
        if(csw_BSM > 0) {
         (assign_mul_one_sw_pm_imu_site_lexic)(ix, &(rr->sp_up), &(s->sp_up), +mu03_BSM);
         (assign_mul_one_sw_pm_imu_site_lexic)(ix, &(rr->sp_dn), &(s->sp_dn), -mu03_BSM);
        }
        else {
          _complex_times_vector(rr->sp_up.s0, rho1, s->sp_up.s0);
          _complex_times_vector(rr->sp_up.s1, rho1, s->sp_up.s1);
          _complex_times_vector(rr->sp_up.s2, rho2, s->sp_up.s2);
          _complex_times_vector(rr->sp_up.s3, rho2, s->sp_up.s3);

          _complex_times_vector(rr->sp_dn.s0, rho2, s->sp_dn.s0);
          _complex_times_vector(rr->sp_dn.s1, rho2, s->sp_dn.s1);
          _complex_times_vector(rr->sp_dn.s2, rho1, s->sp_dn.s2);
          _complex_times_vector(rr->sp_dn.s3, rho1, s->sp_dn.s3);

        }

        // FIXME split r-term into two pieces (perhaps), apply 3r here, 1r via assign_mul_one_sw_pm_imu_site_lexic
        // tmpr += (3*r_BSM+m0_BSM)*s
        // This part takes care of the constant factor coming from the 
        // second derivative in the wilson term 4r0 delta_xy
        // + the additional wilson bare mass m_0
        // Here we already took a factor of 1. in the clover 
        // term account, thus we only have to add a factor of 3
        /* tmpr += (3*r_BSM+m0_BSM)*s */
        _vector_add_mul(rr->sp_up.s0, (3.5+m0_BSM), s->sp_up.s0);
        _vector_add_mul(rr->sp_up.s1, (3.5+m0_BSM), s->sp_up.s1);
        _vector_add_mul(rr->sp_up.s2, (3.5+m0_BSM), s->sp_up.s2);
        _vector_add_mul(rr->sp_up.s3, (3.5+m0_BSM), s->sp_up.s3);

        _vector_add_mul(rr->sp_dn.s0, (3.5+m0_BSM), s->sp_dn.s0);
        _vector_add_mul(rr->sp_dn.s1, (3.5+m0_BSM), s->sp_dn.s1);
        _vector_add_mul(rr->sp_dn.s2, (3.5+m0_BSM), s->sp_dn.s2);
        _vector_add_mul(rr->sp_dn.s3, (3.5+m0_BSM), s->sp_dn.s3);


        /* tmpr += (\eta_BSM+2*\rho_BSM) * F(x)*Q(x) */
        Fadd(rr, s, phi, eta_BSM+2.0*rho_BSM, +1.);

        /* tmpr += \sum_\mu (\rho_BSM/4) * F(x+-\mu)*Q */
        for( int mu=0; mu<4; mu++ ) {
          Fadd(rr, s, phip[mu], 0.25*rho_BSM, +1.);
          Fadd(rr, s, phim[mu], 0.25*rho_BSM, +1.);
        }
        Fabsadd(rr,s,phi,c5phi_BSM);

        /* tmpr+=i\gamma_5\tau_1 mu0 *Q */
        if( fabs(mu01_BSM) > 1.e-10 )
          tm1_add(rr, s, 1);

        /* tmpr+=i\gamma_5\tau_3 mu0 *Q */
        /* if( fabs(mu03_BSM) > 1.e-10 )
          tm3_add(rr, s, 1); */

        /* the hopping part:
         * tmpr += +1/2 \sum_\mu (1-gamma_\mu - \rho_BSM/2*F(x) - \rho_BSM/2*F(x+-\mu))*U_{+-\mu}(x)*Q(x+-\mu)
         ******************************* direction +0 *********************************/
        iy=g_iup[ix][0];
        sp = (bispinor *) Q +iy;
        up=&g_smeared_gauge_field[ix][0];
        p0add_wilsonclover(rr, sp, up, 0, -0.5*phase_0, 1);
        up=&g_gauge_field[ix][0];
        padd_chitildebreak(rr, sp, up, 0, -0.5*phase_0, 0.5*rho_BSM, phi, phip[0], +1.);

        /******************************* direction -0 *********************************/
        iy=g_idn[ix][0];
        sm = (bispinor *) Q +iy;
        um=&g_smeared_gauge_field[iy][0];
        p0add_wilsonclover(rr, sm, um, 1, -0.5*phase_0, 1);
        um=&g_gauge_field[iy][0];
        padd_chitildebreak(rr, sm, um, 1, -0.5*phase_0, 0.5*rho_BSM, phi, phim[0], +1.);

        /******************************* direction +1 *********************************/
        iy=g_iup[ix][1];
        sp = (bispinor *) Q +iy;
        up=&g_smeared_gauge_field[ix][1];
        p1add_wilsonclover(rr, sp, up, 0, -0.5*phase_1, 1);
        up=&g_gauge_field[ix][1];
        padd_chitildebreak(rr, sp, up, 0, -0.5*phase_1, 0.5*rho_BSM, phi, phip[1], +1.);


        /******************************* direction -1 *********************************/
        iy=g_idn[ix][1];
        sm = (bispinor *) Q +iy;
        um=&g_smeared_gauge_field[iy][1];
        p1add_wilsonclover(rr, sm, um, 1, -0.5*phase_1, 1);
        um=&g_gauge_field[iy][1];
        padd_chitildebreak(rr, sm, um, 1, -0.5*phase_1, 0.5*rho_BSM, phi, phim[1], +1.);


        /******************************* direction +2 *********************************/
        iy=g_iup[ix][2];
        sp = (bispinor *) Q +iy;
        up=&g_smeared_gauge_field[ix][2];
        p2add_wilsonclover(rr, sp, up, 0, -0.5*phase_2, 1);
        up=&g_gauge_field[ix][2];
        padd_chitildebreak(rr, sp, up, 0, -0.5*phase_2, 0.5*rho_BSM, phi, phip[2], +1.);


        /******************************* direction -2 *********************************/
        iy=g_idn[ix][2];
        sm = (bispinor *) Q +iy;
        um=&g_smeared_gauge_field[iy][2];
        p2add_wilsonclover(rr, sm, um, 1, -0.5*phase_2, 1);
        um=&g_gauge_field[iy][2];
        padd_chitildebreak(rr, sm, um, 1, -0.5*phase_2, 0.5*rho_BSM, phi, phim[2], +1.);

        /******************************* direction +3 *********************************/
        iy=g_iup[ix][3];
        sp = (bispinor *) Q +iy;
        up=&g_smeared_gauge_field[ix][3];
        p3add_wilsonclover(rr, sp, up, 0, -0.5*phase_3, 1);
        up=&g_gauge_field[ix][3];
        padd_chitildebreak(rr, sp, up, 0, -0.5*phase_3, 0.5*rho_BSM, phi, phip[3], +1.);

        /******************************* direction -3 *********************************/
        iy=g_idn[ix][3];
        sm = (bispinor *) Q +iy;
        um=&g_smeared_gauge_field[iy][3];
        p3add_wilsonclover(rr, sm, um, 1, -0.5*phase_3, 1);
        um=&g_gauge_field[iy][3];
        padd_chitildebreak(rr, sm, um, 1, -0.5*phase_3, 0.5*rho_BSM, phi, phim[3], +1.);
      }
#ifdef TM_USE_OMP
  } /* OpenMP closing brace */
#endif
}




/* D_psi_BSM acts on bispinor fields */
void D_psi_dagger_BSM3(bispinor * const P, bispinor * const Q){
  if(P==Q){
    printf("Error in D_psi_BSM (D_psi_BSM.c):\n");
    printf("Arguments must be different bispinor fields\n");
    printf("Program aborted\n");
    exit(1);
  }
#ifdef TM_USE_MPI
  generic_exchange(Q, sizeof(bispinor));
#endif
  
#ifdef TM_USE_OMP
#pragma omp parallel
  {
#endif
    
    int ix,iy;                       // x, x+-\mu
    su3 * restrict up,* restrict um; // U_\mu(x), U_\mu(x-\mu)
    bispinor * restrict rr;          // P(x)
    bispinor const * restrict s;     // Q(x)
    bispinor const * restrict sp;    // Q(x+\mu)
    bispinor const * restrict sm;    // Q(x-\mu)
    scalar phi[4];                   // phi_i(x)
    scalar phip[4][4];               // phi_i(x+mu) = phip[mu][i]
    scalar phim[4][4];               // phi_i(x-mu) = phim[mu][i] 
    _Complex double rho1, rho2;
    rho1 = 0.5*( 1. +  mu03_BSM * I);
    rho2 = conj(rho1);

    
    /************************ loop over all lattice sites *************************/

#ifdef TM_USE_OMP
#pragma omp for
#endif
    for (ix = 0; ix < VOLUME; ix++) {
      rr = (bispinor *) P + ix;
      s  = (bispinor *) Q + ix;
      
      // prefatch scalar fields
      phi[0] = g_scalar_field[0][ix];
      phi[1] = g_scalar_field[1][ix];
      phi[2] = g_scalar_field[2][ix];
      phi[3] = g_scalar_field[3][ix];
      
      for(int mu = 0; mu < 4; mu++ ) {
        phip[mu][0] = g_scalar_field[0][g_iup[ix][mu]];
        phip[mu][1] = g_scalar_field[1][g_iup[ix][mu]];
        phip[mu][2] = g_scalar_field[2][g_iup[ix][mu]];
        phip[mu][3] = g_scalar_field[3][g_iup[ix][mu]];
        
        phim[mu][0] = g_scalar_field[0][g_idn[ix][mu]];
        phim[mu][1] = g_scalar_field[1][g_idn[ix][mu]];
        phim[mu][2] = g_scalar_field[2][g_idn[ix][mu]];
        phim[mu][3] = g_scalar_field[3][g_idn[ix][mu]];
      }

      // the local part (not local in phi)

       _spinor_null(rr->sp_up);
       _spinor_null(rr->sp_dn);

       if(csw_BSM > 0) {
         (assign_mul_one_sw_pm_imu_site_lexic)(ix, &(rr->sp_up), &(s->sp_up), -mu03_BSM);
         (assign_mul_one_sw_pm_imu_site_lexic)(ix, &(rr->sp_dn), &(s->sp_dn), +mu03_BSM);
        }
        else {
          _complex_times_vector(rr->sp_up.s0, rho2, s->sp_up.s0);
          _complex_times_vector(rr->sp_up.s1, rho2, s->sp_up.s1);
          _complex_times_vector(rr->sp_up.s2, rho1, s->sp_up.s2);
          _complex_times_vector(rr->sp_up.s3, rho1, s->sp_up.s3);

          _complex_times_vector(rr->sp_dn.s0, rho1, s->sp_dn.s0);
          _complex_times_vector(rr->sp_dn.s1, rho1, s->sp_dn.s1);
          _complex_times_vector(rr->sp_dn.s2, rho2, s->sp_dn.s2);
          _complex_times_vector(rr->sp_dn.s3, rho2, s->sp_dn.s3);

        }

 
      // tmpr += (3*r_BSM+m0_BSM)*s
      // This part takes care of the constant factor coming from the 
      // second derivative in the wilson term 4r0 delta_xy
      // + the additional wilson bare mass m_0
      // Here we already took a factor of 1. in the clover 
      // term account, thus we only have to add a factor of 3
      _vector_add_mul(rr->sp_up.s0, (3.5+m0_BSM), s->sp_up.s0);
      _vector_add_mul(rr->sp_up.s1, (3.5+m0_BSM), s->sp_up.s1);
      _vector_add_mul(rr->sp_up.s2, (3.5+m0_BSM), s->sp_up.s2);
      _vector_add_mul(rr->sp_up.s3, (3.5+m0_BSM), s->sp_up.s3);

      _vector_add_mul(rr->sp_dn.s0, (3.5+m0_BSM), s->sp_dn.s0);
      _vector_add_mul(rr->sp_dn.s1, (3.5+m0_BSM), s->sp_dn.s1);
      _vector_add_mul(rr->sp_dn.s2, (3.5+m0_BSM), s->sp_dn.s2);
      _vector_add_mul(rr->sp_dn.s3, (3.5+m0_BSM), s->sp_dn.s3);
      
      // tmpr += (\eta_BSM+2*\rho_BSM) * Fbar(x)*Q(x)
      Fadd(rr, s, phi, eta_BSM+2.0*rho_BSM, -1.);
      
      // tmpr += \sum_\mu (\rho_BSM/4) * F(x+-\mu)*Q
      for(int mu = 0; mu < 4; mu++) {
        Fadd(rr, s, phip[mu], 0.25*rho_BSM, -1.);
        Fadd(rr, s, phim[mu], 0.25*rho_BSM, -1.);
      }

      // tmpr += c5phi_BSM \Phi^\dagger\Phi Q
      Fabsadd(rr,s,phi,c5phi_BSM);

      // tmpr+=i\gamma_5\tau_1 mu0 *Q 
      if( fabs(mu01_BSM) > 1.e-10 )
        tm1_add(rr, s, -1);

      // tmpr+=i\gamma_5\tau_3 mu0 *Q 
      /*if( fabs(mu03_BSM) > 1.e-10 )
        tm3_add(rr, s, -1);*/

      // the hopping part:
      // tmpr += +1/2 \sum_\mu (1+\gamma_\mu - \rho_BSM/2*Fbar(x) - \rho_BSM/2*Fbar(x+-\mu)*U_{+-\mu}(x)*Q(x+-\mu)
      /******************************* direction +0 *********************************/
      iy=g_iup[ix][0];
      sp = (bispinor *) Q +iy;
      up=&g_smeared_gauge_field[ix][0];
      p0add_wilsonclover(rr, sp, up, 0, -0.5*phase_0, -1);
      up=&g_gauge_field[ix][0];
      padd_chitildebreak(rr, sp, up, 0, -0.5*phase_0, 0.5*rho_BSM, phi, phip[0], -1.);
 
      /******************************* direction -0 *********************************/
      iy=g_idn[ix][0];
      sm = (bispinor *) Q +iy;
      um=&g_smeared_gauge_field[iy][0];
      p0add_wilsonclover(rr, sm, um, 1, -0.5*phase_0, -1);
      um=&g_gauge_field[iy][0];
      padd_chitildebreak(rr, sm, um, 1, -0.5*phase_0, 0.5*rho_BSM, phi, phim[0], -1.);
      /******************************* direction +1 *********************************/
      iy=g_iup[ix][1];
      sp = (bispinor *) Q +iy;
      up=&g_smeared_gauge_field[ix][1];
      p1add_wilsonclover(rr, sp, up, 0, -0.5*phase_1, -1);
      up=&g_gauge_field[ix][1];
      padd_chitildebreak(rr, sp, up, 0, -0.5*phase_1, 0.5*rho_BSM, phi, phip[1], -1.);
 
      /******************************* direction -1 *********************************/
      iy=g_idn[ix][1];
      sm = (bispinor *) Q +iy;
      um=&g_smeared_gauge_field[iy][1];
      p1add_wilsonclover(rr, sm, um, 1, -0.5*phase_1, -1);
      um=&g_gauge_field[iy][1];
      padd_chitildebreak(rr, sm, um, 1, -0.5*phase_1, 0.5*rho_BSM, phi, phim[1], -1.);
 
      /******************************* direction +2 *********************************/
      iy=g_iup[ix][2];
      sp = (bispinor *) Q +iy;
      up=&g_smeared_gauge_field[ix][2];
      p2add_wilsonclover(rr, sp, up, 0, -0.5*phase_2, -1);
      up=&g_gauge_field[ix][2];
      padd_chitildebreak(rr, sp, up, 0, -0.5*phase_2, 0.5*rho_BSM, phi, phip[2], -1.);

      /******************************* direction -2 *********************************/
      iy=g_idn[ix][2];
      sm = (bispinor *) Q +iy;
      um=&g_smeared_gauge_field[iy][2]; 
      p2add_wilsonclover(rr, sm, um, 1, -0.5*phase_2, -1);
      um=&g_gauge_field[iy][2]; 
      padd_chitildebreak(rr, sm, um, 1, -0.5*phase_2, 0.5*rho_BSM, phi, phim[2], -1.);
 
      /******************************* direction +3 *********************************/
      iy=g_iup[ix][3];
      sp = (bispinor *) Q +iy;
      up=&g_smeared_gauge_field[ix][3];
      p3add_wilsonclover(rr, sp, up, 0, -0.5*phase_3, -1);
      up=&g_gauge_field[ix][3];
      padd_chitildebreak(rr, sp, up, 0, -0.5*phase_3, 0.5*rho_BSM, phi, phip[3], -1.);
      
      /******************************* direction -3 *********************************/
      iy=g_idn[ix][3];
      sm = (bispinor *) Q +iy;
      um=&g_smeared_gauge_field[iy][3];
      p3add_wilsonclover(rr, sm, um, 1, -0.5*phase_3, -1);
      um=&g_gauge_field[iy][3];
      padd_chitildebreak(rr, sm, um, 1, -0.5*phase_3, 0.5*rho_BSM, phi, phim[3], -1.);
    }
#ifdef TM_USE_OMP
  } /* OpenMP closing brace */
#endif
}

/* Q2_psi_BSM acts on bispinor fields */
void Q2_psi_BSM3(bispinor * const P, bispinor * const Q){

  D_psi_dagger_BSM3(tempor , Q);
  D_psi_BSM3(P, tempor);
  /* Q and P are spinor, not bispinor ==> made a cast */
  /* the use of [3] has to be changed to avoid future conflicts */

}
#endif
