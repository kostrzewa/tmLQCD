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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more deta_BSMils.
 * 
 * You should have received a copy of the GNU General Public License
 * along with tmLQCD. If not, see <http://www.gnu.org/licenses/>.
 *
 *******************************************************************************/

/*******************************************************************************
 
 * Implementation of symmetric derivative version of Frezzotti-Rossi Dirac operator
 * with a scalar field coupling.
 *
 *******************************************************************************/

#ifdef HAVE_CONFIG_H
# include<config.h>
#endif

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "global.h"
#include "su3.h"
#include "su3spinor.h"
#include "sse.h"
#include "boundary.h"
#ifdef MPI
# include "xchange/xchange.h"
#endif
#include "update_backward_gauge.h"
#include "operator/D_psi_BSM2b.h"
#include "operator/D_psi_BSM2f.h"
#include "solver/dirac_operator_eigenvectors.h"
#include "buffers/utils.h"
#include "buffers/utils_nonblocking.h"
#include "linalg_eo.h"
#include "fatal_error.h"


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

static bispinor *vm1;
static bispinor *vm2;
static bispinor *vm3;
static bispinor *vm4;

static bispinor *vp1;
static bispinor *vp2;
static bispinor *vp3;
static bispinor *vp4;

static bispinor *v2m1;
static bispinor *v2m2;
static bispinor *v2m3;
static bispinor *v2m4;
void init_D_psi_BSM2f(){

     vm1 =(bispinor *)calloc(VOLUMEPLUSRAND,sizeof(bispinor));
     vm2 =(bispinor *)calloc(VOLUMEPLUSRAND,sizeof(bispinor));
     vm3 =(bispinor *)calloc(VOLUMEPLUSRAND,sizeof(bispinor));
     vm4 =(bispinor *)calloc(VOLUMEPLUSRAND,sizeof(bispinor));
     vp1 =(bispinor *)calloc(VOLUMEPLUSRAND,sizeof(bispinor));
     vp2 =(bispinor *)calloc(VOLUMEPLUSRAND,sizeof(bispinor));
     vp3 =(bispinor *)calloc(VOLUMEPLUSRAND,sizeof(bispinor));
     vp4 =(bispinor *)calloc(VOLUMEPLUSRAND,sizeof(bispinor));
     v2m1=(bispinor *)calloc(VOLUMEPLUSRAND,sizeof(bispinor));
     v2m2=(bispinor *)calloc(VOLUMEPLUSRAND,sizeof(bispinor));
     v2m3=(bispinor *)calloc(VOLUMEPLUSRAND,sizeof(bispinor));
     v2m4=(bispinor *)calloc(VOLUMEPLUSRAND,sizeof(bispinor));

}
void free_D_psi_BSM2f(){
     free(vm1);
     free(vm2);
     free(vm3);
     free(vm4);
     free(vp1);
     free(vp2);
     free(vp3);
     free(vp4);
     free(v2m1);
     free(v2m2);
     free(v2m3);
     free(v2m4);
}
static inline void Fadd(bispinor * const out, const bispinor * const in, const scalar * const phi, const double c, const double sign) {
  static spinor tmp;
  double s = +1.;
  if(sign < 0) s = -1.;

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

static inline void bispinor_times_phase_times_u(bispinor * const us, const _Complex double phase,
            su3 const * restrict const u, bispinor const * const s)
{
  static su3_vector chi;
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

static inline void bispinor_times_phase_times_inverse_u(bispinor * const us, const _Complex double phase,
              su3 const * restrict const u, bispinor const * const s)
{
  static su3_vector chi;
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

static inline void padd(bispinor * restrict const dest, bispinor * const restrict const source,
                         su3 const * restrict const u, const hopdirection dir, const _Complex double phase) 
{
  // us = phase*u*s
  if( dir == HOP_DN )
    bispinor_times_phase_times_inverse_u(dest, phase, u, source);
  else
    bispinor_times_phase_times_u        (dest, phase, u, source);
}
static inline void _bispinor_add_mult_gamma0( bispinor * restrict const dest, bispinor const * restrict const source )
{  
  // tmpr += \gamma_0*us
  _vector_add_assign(dest->sp_up.s0, source->sp_up.s2);
  _vector_add_assign(dest->sp_up.s1, source->sp_up.s3);
  _vector_add_assign(dest->sp_up.s2, source->sp_up.s0);
  _vector_add_assign(dest->sp_up.s3, source->sp_up.s1);

  _vector_add_assign(dest->sp_dn.s0, source->sp_dn.s2);
  _vector_add_assign(dest->sp_dn.s1, source->sp_dn.s3);
  _vector_add_assign(dest->sp_dn.s2, source->sp_dn.s0);
  _vector_add_assign(dest->sp_dn.s3, source->sp_dn.s1);

  return;
}

static inline void _bispinor_add_mult_gamma1( bispinor * restrict const dest, bispinor const * restrict const source )
{
  // tmpr += \gamma_1*us
  _vector_i_add_assign(dest->sp_up.s0, source->sp_up.s3);
  _vector_i_add_assign(dest->sp_up.s1, source->sp_up.s2);
  _vector_i_sub_assign(dest->sp_up.s2, source->sp_up.s1);
  _vector_i_sub_assign(dest->sp_up.s3, source->sp_up.s0);

  _vector_i_add_assign(dest->sp_dn.s0, source->sp_dn.s3);
  _vector_i_add_assign(dest->sp_dn.s1, source->sp_dn.s2);
  _vector_i_sub_assign(dest->sp_dn.s2, source->sp_dn.s1);
  _vector_i_sub_assign(dest->sp_dn.s3, source->sp_dn.s0);

  return;
}
static inline void _bispinor_add_mult_gamma2( bispinor * restrict const dest, bispinor const * restrict const source )
{
  // tmpr += \gamma_2*us
  _vector_add_assign(dest->sp_up.s0, source->sp_up.s3);
  _vector_sub_assign(dest->sp_up.s1, source->sp_up.s2);
  _vector_sub_assign(dest->sp_up.s2, source->sp_up.s1);
  _vector_add_assign(dest->sp_up.s3, source->sp_up.s0);

  _vector_add_assign(dest->sp_dn.s0, source->sp_dn.s3);
  _vector_sub_assign(dest->sp_dn.s1, source->sp_dn.s2);
  _vector_sub_assign(dest->sp_dn.s2, source->sp_dn.s1);
  _vector_add_assign(dest->sp_dn.s3, source->sp_dn.s0);

  return;
}
static inline void _bispinor_add_mult_gamma3( bispinor * restrict const dest, bispinor const * restrict const source )
{
  // tmpr += \gamma_3*us
  _vector_i_add_assign(dest->sp_up.s0, source->sp_up.s2);
  _vector_i_sub_assign(dest->sp_up.s1, source->sp_up.s3);
  _vector_i_sub_assign(dest->sp_up.s2, source->sp_up.s0);
  _vector_i_add_assign(dest->sp_up.s3, source->sp_up.s1);

  _vector_i_add_assign(dest->sp_dn.s0, source->sp_dn.s2);
  _vector_i_sub_assign(dest->sp_dn.s1, source->sp_dn.s3);
  _vector_i_sub_assign(dest->sp_dn.s2, source->sp_dn.s0);
  _vector_i_add_assign(dest->sp_dn.s3, source->sp_dn.s1);

  return;

}


/* D_psi_BSM2f acts on bispinor fields */
void D_psi_BSM2f(bispinor * const P, bispinor * const Q){
  if(P==Q){
    printf("Error in D_psi_BSM2f (D_psi_BSM2f.c):\n");
    printf("Arguments must be different bispinor fields\n");
    printf("Program aborted\n");
    exit(1);
  }
#ifdef _GAUGE_COPY
  if(g_update_gauge_copy) {
    update_backward_gauge(g_gauge_field);
  }
#endif
  int ix;
  su3 * restrict upm;
  bispinor * restrict rr;
  bispinor * restrict rr0;
  bispinor * restrict rr1;
  bispinor * restrict rr2;
  bispinor * restrict rr3;
  bispinor * restrict rr4;
  bispinor * restrict rr5;
  bispinor * restrict rr6;
  bispinor * restrict rr7;
  bispinor * restrict rrs0;
  bispinor * restrict rrs1;
  bispinor * restrict rrs2;
  bispinor * restrict rrs3;

  bispinor const * restrict s;         // Q(x)
  bispinor const * restrict spm;
  scalar phi[4];                       // phi_i(x)
  scalar phip[4][4];                   // phi_i(x+mu) = phip[mu][i]
  scalar phim[4][4];                   // phi_i(x-mu) = phim[mu][i]
  bispinor ALIGN stmp2;


  MPI_Status  statuses[8];
  MPI_Request *request;
  request=( MPI_Request *) malloc(sizeof(MPI_Request)*8);

//start gathering forward
  int count=0;
  generic_exchange_direction_nonblocking(Q, sizeof(bispinor), TUP, request, &count);
  generic_exchange_direction_nonblocking(Q, sizeof(bispinor), XUP, request, &count);
  generic_exchange_direction_nonblocking(Q, sizeof(bispinor), YUP, request, &count);
  generic_exchange_direction_nonblocking(Q, sizeof(bispinor), ZUP, request, &count);

//  computing backward
  for (ix=0;ix<VOLUME;ix++)
  {
// intermediate buffers for storing backward connections : U_mu( x-mu)^dagg psi(x-mu)
    rr4 = vm1 + ix;
    rr5 = vm2 + ix;
    rr6 = vm3 + ix;
    rr7 = vm4 + ix;

    _bispinor_null(*rr4);
    _bispinor_null(*rr5);
    _bispinor_null(*rr6);
    _bispinor_null(*rr7);

//source buffer
    s  = (bispinor *) Q + ix;

//  Direction 0 -
    upm = &g_gauge_field[ix][TUP];
    padd(rr7, s, upm, HOP_DN, 0.5*phase_0);

//  Direction 1 -
    upm = &g_gauge_field[ix][XUP];
    padd(rr6, s, upm, HOP_DN, 0.5*phase_1);

//  Direction 2 -
    upm = &g_gauge_field[ix][YUP];
    padd(rr5, s, upm, HOP_DN, 0.5*phase_2);

//  Direction 3 -
    upm = &g_gauge_field[ix][ZUP];
    padd(rr4, s, upm, HOP_DN, 0.5*phase_3);
  }
  
  MPI_Waitall( count, request, statuses);

//gathering backward

  count=0;
  generic_exchange_direction_nonblocking(vm4, sizeof(bispinor), TDOWN, request, &count);
  generic_exchange_direction_nonblocking(vm3, sizeof(bispinor), XDOWN, request, &count);
  generic_exchange_direction_nonblocking(vm2, sizeof(bispinor), YDOWN, request, &count);
  generic_exchange_direction_nonblocking(vm1, sizeof(bispinor), ZDOWN, request, &count);

//computing forward
  for (ix=0;ix<VOLUME;ix++)
  {
// intermediate buffers for storing forward connections : U_mu(x) psi(x+mu)
    rr0 = vp1 + ix;
    rr1 = vp2 + ix;
    rr2 = vp3 + ix;
    rr3 = vp4 + ix;

// intermedieate buffer for multiplication with gamma_mu 
    _bispinor_null(*rr0);
    _bispinor_null(*rr1);
    _bispinor_null(*rr2);
    _bispinor_null(*rr3);

//  Direction 0 +
    upm = &g_gauge_field[ix][TUP];
    spm = (bispinor *) Q + g_iup[ix][TUP];
    padd(rr0, spm,  upm, HOP_UP, 0.5*phase_0);

//  Direction 1 +
    upm = &g_gauge_field[ix][XUP];
    spm = (bispinor *) Q + g_iup[ix][XUP];
    padd(rr1, spm,  upm, HOP_UP, 0.5*phase_1);

//  Direction 2 +
    upm = &g_gauge_field[ix][YUP];
    spm = (bispinor *) Q + g_iup[ix][YUP];
    padd(rr2, spm,  upm, HOP_UP, 0.5*phase_2);

//  Direction 3 +
    upm = &g_gauge_field[ix][ZUP];
    spm = (bispinor *) Q + g_iup[ix][ZUP];
    padd(rr3, spm,  upm, HOP_UP, 0.5*phase_3);

  }

  MPI_Waitall( count, request, statuses);

// join
  for (ix=0; ix<VOLUME; ++ix){

// destination buffer
    rr = (bispinor *) P + ix;
    _bispinor_null(*rr);

// intermediate buffers for storing backward connections : U_mu( x )^dagg psi(x+mu)
    rr0 = vp1 + ix;
    rr1 = vp2 + ix;
    rr2 = vp3 + ix;
    rr3 = vp4 + ix;

// intermediate buffers for storing backward connections : U_mu( x-mu)^dagg psi(x-mu)
    rr4 = vm1 + g_idn[ix][ZUP];
    rr5 = vm2 + g_idn[ix][YUP];
    rr6 = vm3 + g_idn[ix][XUP];
    rr7 = vm4 + g_idn[ix][TUP];

    _bispinor_null( stmp2 );
    _bispinor_add_mul( stmp2, +1.0, *rr0 );
    _bispinor_add_mul( stmp2, -1.0, *rr7 );
    _bispinor_add_mult_gamma0(rr, &stmp2 );

    _bispinor_null( stmp2 );
    _bispinor_add_mul( stmp2, +1.0, *rr1 );
    _bispinor_add_mul( stmp2, -1.0, *rr6 );
    _bispinor_add_mult_gamma1(rr, &stmp2 );

    _bispinor_null( stmp2 );
    _bispinor_add_mul( stmp2, +1.0, *rr2 );
    _bispinor_add_mul( stmp2, -1.0, *rr5 );
    _bispinor_add_mult_gamma2(rr, &stmp2 );

    _bispinor_null( stmp2 );
    _bispinor_add_mul( stmp2, +1.0, *rr3 );
    _bispinor_add_mul( stmp2, -1.0, *rr4 );
    _bispinor_add_mult_gamma3(rr, &stmp2 );
  
  } //end of gamma_mu D_mu


//start gathering forward
  count=0;
  generic_exchange_direction_nonblocking(vp1, sizeof(bispinor), TUP, request, &count);
  generic_exchange_direction_nonblocking(vp2, sizeof(bispinor), XUP, request, &count);
  generic_exchange_direction_nonblocking(vp3, sizeof(bispinor), YUP, request, &count);
  generic_exchange_direction_nonblocking(vp4, sizeof(bispinor), ZUP, request, &count);


//start computing backward
  for (ix=0;ix<VOLUME;ix++)
  {
// intermediate buffers for storing backward connections : U_mu( x-mu)^dagg psi(x-mu)
    rr4 = vm1 + g_idn[ix][ZUP];
    rr5 = vm2 + g_idn[ix][YUP];
    rr6 = vm3 + g_idn[ix][XUP];
    rr7 = vm4 + g_idn[ix][TUP];
 
    rrs0 =  v2m1 + ix;
    rrs1 =  v2m2 + ix;
    rrs2 =  v2m3 + ix;
    rrs3 =  v2m4 + ix;

    _bispinor_null( *rrs0 );
    _bispinor_null( *rrs1 );
    _bispinor_null( *rrs2 );
    _bispinor_null( *rrs3 );

//  prefatch scalar fields
    phi[0] = g_scalar_field[0][ix];
    phi[1] = g_scalar_field[1][ix];
    phi[2] = g_scalar_field[2][ix];
    phi[3] = g_scalar_field[3][ix];

//  Direction 0 -
    upm = &g_gauge_field[ix][TUP];
    _bispinor_null( stmp2 );
    padd(&stmp2, rr7,  upm, HOP_DN, 2.0*phase_0);
    Fadd( rrs0, &stmp2, phi, -0.125*rho_BSM, +1. );


//  Direction 1 -
    upm = &g_gauge_field[ix][XUP];
    _bispinor_null( stmp2 );
    padd(&stmp2, rr6, upm, HOP_DN, 2.0*phase_1);
    Fadd( rrs1, &stmp2, phi, -0.125*rho_BSM, +1. );


//  Direction 2 -
    upm = &g_gauge_field[ix][YUP];
    _bispinor_null( stmp2 );
    padd(&stmp2, rr5, upm, HOP_DN, 2.0*phase_2);
    Fadd( rrs2, &stmp2, phi, -0.125*rho_BSM, +1. );


//  Direction 3 -
    upm = &g_gauge_field[ix][ZUP];
    _bispinor_null( stmp2 );
    padd(&stmp2, rr4, upm, HOP_DN, 2.0*phase_3);
    Fadd( rrs3, &stmp2, phi, -0.125*rho_BSM, +1. );

  }
  MPI_Waitall( count, request, statuses);

//gathering backward
  count=0;
  generic_exchange_direction_nonblocking(v2m1, sizeof(bispinor), TDOWN, request, &count);
  generic_exchange_direction_nonblocking(v2m2, sizeof(bispinor), XDOWN, request, &count);
  generic_exchange_direction_nonblocking(v2m3, sizeof(bispinor), YDOWN, request, &count);
  generic_exchange_direction_nonblocking(v2m4, sizeof(bispinor), ZDOWN, request, &count);

//computing forward
  for (ix=0;ix<VOLUME;ix++)
  {
// intermediate buffers for storing forward connections : U_mu(x) psi(x+mu)
    rr0 = vp1 + g_iup[ix][TUP];
    rr1 = vp2 + g_iup[ix][XUP];
    rr2 = vp3 + g_iup[ix][YUP];
    rr3 = vp4 + g_iup[ix][ZUP];

    for ( int mu=0; mu<4; mu++ )
    {
      phip[mu][0] = g_scalar_field[0][g_iup[ix][mu]];
      phip[mu][1] = g_scalar_field[1][g_iup[ix][mu]];
      phip[mu][2] = g_scalar_field[2][g_iup[ix][mu]];
      phip[mu][3] = g_scalar_field[3][g_iup[ix][mu]];
    }
//dest buffer
    rr = (bispinor *) P + ix;

//  Direction 0 +
    upm = &g_gauge_field[ix][TUP];
    _bispinor_null( stmp2 );
    padd( &stmp2, rr0, upm, HOP_UP, 2.0*phase_0);
    Fadd( rr, &stmp2, phip[TUP], -0.125*rho_BSM, +1. );

//  Direction 1 +
    upm = &g_gauge_field[ix][XUP];
     _bispinor_null( stmp2 );
    padd( &stmp2, rr1, upm, HOP_UP, 2.0*phase_1);
    Fadd( rr, &stmp2, phip[XUP], -0.125*rho_BSM, +1. );

//  Direction 2 +
    upm = &g_gauge_field[ix][YUP];
    _bispinor_null( stmp2 );
    padd( &stmp2, rr2, upm, HOP_UP, 2.0*phase_2);
    Fadd( rr, &stmp2, phip[YUP], -0.125*rho_BSM, +1. );

//  Direction 3 +
    upm = &g_gauge_field[ix][ZUP];
    _bispinor_null( stmp2 );
    padd( &stmp2, rr3, upm, HOP_UP, 2.0*phase_3);
    Fadd( rr, &stmp2, phip[ZUP], -0.125*rho_BSM, +1. );
  }

  MPI_Waitall( count, request, statuses);

//join

 for (ix=0; ix<VOLUME; ++ix){
    rr = (bispinor *) P + ix;
    s  = (bispinor *) Q + ix;

//  prefatch scalar fields
    phi[0] = g_scalar_field[0][ix];
    phi[1] = g_scalar_field[1][ix];
    phi[2] = g_scalar_field[2][ix];
    phi[3] = g_scalar_field[3][ix];

    for ( int mu=0; mu<4; mu++ )
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
    
// intermediate buffers for storing backward connections : U_mu( x-mu)^dagg psi(x-mu)
    rr4 = v2m1 + g_idn[ix][TUP];
    rr5 = v2m2 + g_idn[ix][XUP];
    rr6 = v2m3 + g_idn[ix][YUP];
    rr7 = v2m4 + g_idn[ix][ZUP];

    _bispinor_add_mul( *rr, +1.0, *rr4 );
    _bispinor_add_mul( *rr, +1.0, *rr5 );
    _bispinor_add_mul( *rr, +1.0, *rr6 );
    _bispinor_add_mul( *rr, +1.0, *rr7 );
  
    Fadd(rr, s, phi, eta_BSM, +1. );

    // tmpr += \sum_\mu (\rho_BSM/8) * F(x+-\mu)*Q
   for( int mu=0; mu<4; mu++ ) {
      Fadd(rr, s, phip[mu], 0.125*rho_BSM, +1. );
      Fadd(rr, s, phim[mu], 0.125*rho_BSM, +1. );
   }


  } // end volume loop
  free(request);

}

/* D_psi_BSM2i acts on bispinor fields */
void D_psi_dagger_BSM2f(bispinor * const P, bispinor * const Q){
  if(P==Q){
    printf("Error in D_psi_BSM2f (D_psi_BSM2f.c):\n");
    printf("Arguments must be different bispinor fields\n");
    printf("Program aborted\n");
    exit(1);
  }
#ifdef _GAUGE_COPY
  if(g_update_gauge_copy) {
    update_backward_gauge(g_gauge_field);
  }
#endif
  int ix;
  su3 * restrict upm;
  bispinor * restrict rr;
  bispinor * restrict rr0;
  bispinor * restrict rr1;
  bispinor * restrict rr2;
  bispinor * restrict rr3;
  bispinor * restrict rr4;
  bispinor * restrict rr5;
  bispinor * restrict rr6;
  bispinor * restrict rr7;
  bispinor * restrict rrs0;
  bispinor * restrict rrs1;
  bispinor * restrict rrs2;
  bispinor * restrict rrs3;

  bispinor const * restrict s;         // Q(x)
  bispinor const * restrict spm;
  scalar phi[4];                       // phi_i(x)
  scalar phip[4][4];                   // phi_i(x+mu) = phip[mu][i]
  scalar phim[4][4];                   // phi_i(x-mu) = phim[mu][i]
  bispinor ALIGN stmp2;

  MPI_Status  statuses[8];
  MPI_Request *request;
  request=( MPI_Request *) malloc(sizeof(MPI_Request)*8);

//start gathering forward
  int count=0;
  generic_exchange_direction_nonblocking(Q, sizeof(bispinor), TUP, request, &count);
  generic_exchange_direction_nonblocking(Q, sizeof(bispinor), XUP, request, &count);
  generic_exchange_direction_nonblocking(Q, sizeof(bispinor), YUP, request, &count);
  generic_exchange_direction_nonblocking(Q, sizeof(bispinor), ZUP, request, &count);

//  computing backward
  for (ix=0;ix<VOLUME;ix++)
  {
// intermediate buffers for storing backward connections : U_mu( x-mu)^dagg psi(x-mu)
    rr4 = vm1 + ix;
    rr5 = vm2 + ix;
    rr6 = vm3 + ix;
    rr7 = vm4 + ix;

    _bispinor_null(*rr4);
    _bispinor_null(*rr5);
    _bispinor_null(*rr6);
    _bispinor_null(*rr7);

//source buffer
    s  = (bispinor *) Q + ix;

//  Direction 0 -
    upm = &g_gauge_field[ix][TUP];
    padd(rr7, s, upm, HOP_DN, 0.5*phase_0);

//  Direction 1 -
    upm = &g_gauge_field[ix][XUP];
    padd(rr6, s, upm, HOP_DN, 0.5*phase_1);

//  Direction 2 -
    upm = &g_gauge_field[ix][YUP];
    padd(rr5, s, upm, HOP_DN, 0.5*phase_2);

//  Direction 3 -
    upm = &g_gauge_field[ix][ZUP];
    padd(rr4, s, upm, HOP_DN, 0.5*phase_3);
  }

  MPI_Waitall( count, request, statuses);

//gathering backward

  count=0;
  generic_exchange_direction_nonblocking(vm4, sizeof(bispinor), TDOWN, request, &count);
  generic_exchange_direction_nonblocking(vm3, sizeof(bispinor), XDOWN, request, &count);
  generic_exchange_direction_nonblocking(vm2, sizeof(bispinor), YDOWN, request, &count);
  generic_exchange_direction_nonblocking(vm1, sizeof(bispinor), ZDOWN, request, &count);

//computing forward
  for (ix=0;ix<VOLUME;ix++)
  {
// intermediate buffers for storing forward connections : U_mu(x) psi(x+mu)
    rr0 = vp1 + ix;
    rr1 = vp2 + ix;
    rr2 = vp3 + ix;
    rr3 = vp4 + ix;

// intermedieate buffer for multiplication with gamma_mu 
    _bispinor_null(*rr0);
    _bispinor_null(*rr1);
    _bispinor_null(*rr2);
    _bispinor_null(*rr3);

//  Direction 0 +
    upm = &g_gauge_field[ix][TUP];
    spm = (bispinor *) Q + g_iup[ix][TUP];
    padd(rr0, spm,  upm, HOP_UP, 0.5*phase_0);

//  Direction 1 +
    upm = &g_gauge_field[ix][XUP];
    spm = (bispinor *) Q + g_iup[ix][XUP];
    padd(rr1, spm,  upm, HOP_UP, 0.5*phase_1);

//  Direction 2 +
    upm = &g_gauge_field[ix][YUP];
    spm = (bispinor *) Q + g_iup[ix][YUP];
    padd(rr2, spm,  upm, HOP_UP, 0.5*phase_2);

//  Direction 3 +
    upm = &g_gauge_field[ix][ZUP];
    spm = (bispinor *) Q + g_iup[ix][ZUP];
    padd(rr3, spm,  upm, HOP_UP, 0.5*phase_3);

  }

  MPI_Waitall( count, request, statuses);

// join
  for (ix=0; ix<VOLUME; ++ix){

// destination buffer
    rr = (bispinor *) P + ix;
    _bispinor_null(*rr);

// intermediate buffers for storing backward connections : U_mu( x )^dagg psi(x+mu)
    rr0 = vp1 + ix;
    rr1 = vp2 + ix;
    rr2 = vp3 + ix;
    rr3 = vp4 + ix;

// intermediate buffers for storing backward connections : U_mu( x-mu)^dagg psi(x-mu)
    rr4 = vm1 + g_idn[ix][ZUP];
    rr5 = vm2 + g_idn[ix][YUP];
    rr6 = vm3 + g_idn[ix][XUP];
    rr7 = vm4 + g_idn[ix][TUP];

    _bispinor_null( stmp2 );
    _bispinor_add_mul( stmp2, -1.0, *rr0 );
    _bispinor_add_mul( stmp2, +1.0, *rr7 );
    _bispinor_add_mult_gamma0(rr, &stmp2 );

    _bispinor_null( stmp2 );
    _bispinor_add_mul( stmp2, -1.0, *rr1 );
    _bispinor_add_mul( stmp2, +1.0, *rr6 );
    _bispinor_add_mult_gamma1(rr, &stmp2 );

    _bispinor_null( stmp2 );
    _bispinor_add_mul( stmp2, -1.0, *rr2 );
    _bispinor_add_mul( stmp2, +1.0, *rr5 );
    _bispinor_add_mult_gamma2(rr, &stmp2 );

    _bispinor_null( stmp2 );
    _bispinor_add_mul( stmp2, -1.0, *rr3 );
    _bispinor_add_mul( stmp2, +1.0, *rr4 );
    _bispinor_add_mult_gamma3(rr, &stmp2 );

  } //end of gamma_mu D_mu
//  for (ix=0; ix<VOLUME; ++ix){
//      rr=(bispinor*)P+ix;
//      printf("endgammamuDmu source= %e %e  \n", creal(rr->sp_up.s0.c0),cimag(rr->sp_up.s0.c0) );
//  }


//start gathering forward
  count=0;
  generic_exchange_direction_nonblocking(vp1, sizeof(bispinor), TUP, request, &count);
  generic_exchange_direction_nonblocking(vp2, sizeof(bispinor), XUP, request, &count);
  generic_exchange_direction_nonblocking(vp3, sizeof(bispinor), YUP, request, &count);
  generic_exchange_direction_nonblocking(vp4, sizeof(bispinor), ZUP, request, &count);

//start computing backward
  for (ix=0;ix<VOLUME;ix++)
  {
// intermediate buffers for storing backward connections : U_mu( x-mu)^dagg psi(x-mu)
    rr4 = vm1 + g_idn[ix][ZUP];
    rr5 = vm2 + g_idn[ix][YUP];
    rr6 = vm3 + g_idn[ix][XUP];
    rr7 = vm4 + g_idn[ix][TUP];

    rrs0 = v2m1 + ix;
    rrs1 = v2m2 + ix;
    rrs2 = v2m3 + ix;
    rrs3 = v2m4 + ix;

    _bispinor_null( *rrs0 );
    _bispinor_null( *rrs1 );
    _bispinor_null( *rrs2 );
    _bispinor_null( *rrs3 );

//  prefatch scalar fields
    phi[0] = g_scalar_field[0][ix];
    phi[1] = g_scalar_field[1][ix];
    phi[2] = g_scalar_field[2][ix];
    phi[3] = g_scalar_field[3][ix];

//  Direction 0 -
    upm = &g_gauge_field[ix][TUP];
    _bispinor_null( stmp2 );
    padd(&stmp2, rr7,  upm, HOP_DN, 2.0*phase_0);
    Fadd( rrs0, &stmp2, phi, -0.125*rho_BSM, -1. );


//  Direction 1 -
    upm = &g_gauge_field[ix][XUP];
    _bispinor_null( stmp2 );
    padd(&stmp2, rr6, upm, HOP_DN, 2.0*phase_1);
    Fadd( rrs1, &stmp2, phi, -0.125*rho_BSM, -1. );


//  Direction 2 -
    upm = &g_gauge_field[ix][YUP];
    _bispinor_null( stmp2 );
    padd(&stmp2, rr5, upm, HOP_DN, 2.0*phase_2);
    Fadd( rrs2, &stmp2, phi, -0.125*rho_BSM, -1. );


//  Direction 3 -
    upm = &g_gauge_field[ix][ZUP];
    _bispinor_null( stmp2 );
    padd(&stmp2, rr4, upm, HOP_DN, 2.0*phase_3);
    Fadd( rrs3, &stmp2, phi, -0.125*rho_BSM, -1. );

  }
  MPI_Waitall( count, request, statuses);

//gathering backward
  count=0;

  generic_exchange_direction_nonblocking(v2m1, sizeof(bispinor), TDOWN, request, &count);
  generic_exchange_direction_nonblocking(v2m2, sizeof(bispinor), XDOWN, request, &count);
  generic_exchange_direction_nonblocking(v2m3, sizeof(bispinor), YDOWN, request, &count);
  generic_exchange_direction_nonblocking(v2m4, sizeof(bispinor), ZDOWN, request, &count);

//computing forward
  for (ix=0;ix<VOLUME;ix++)
  {
// intermediate buffers for storing forward connections : U_mu(x) psi(x+mu)
    rr0 = vp1 + g_iup[ix][TUP];
    rr1 = vp2 + g_iup[ix][XUP];
    rr2 = vp3 + g_iup[ix][YUP];
    rr3 = vp4 + g_iup[ix][ZUP];

    for ( int mu=0; mu<4; mu++ )
    {
      phip[mu][0] = g_scalar_field[0][g_iup[ix][mu]];
      phip[mu][1] = g_scalar_field[1][g_iup[ix][mu]];
      phip[mu][2] = g_scalar_field[2][g_iup[ix][mu]];
      phip[mu][3] = g_scalar_field[3][g_iup[ix][mu]];
    }
//dest buffer
    rr = (bispinor *) P + ix;

//  Direction 0 +
    upm = &g_gauge_field[ix][TUP];
    _bispinor_null( stmp2 );
    padd( &stmp2, rr0, upm, HOP_UP, 2.0*phase_0);
    Fadd( rr, &stmp2, phip[TUP], -0.125*rho_BSM, -1. );

//  Direction 1 +
    upm = &g_gauge_field[ix][XUP];
     _bispinor_null( stmp2 );
    padd( &stmp2, rr1, upm, HOP_UP, 2.0*phase_1);
    Fadd( rr, &stmp2, phip[XUP], -0.125*rho_BSM, -1. );

//  Direction 2 +
    upm = &g_gauge_field[ix][YUP];
    _bispinor_null( stmp2 );
    padd( &stmp2, rr2, upm, HOP_UP, 2.0*phase_2);
    Fadd( rr, &stmp2, phip[YUP], -0.125*rho_BSM, -1. );

//  Direction 3 +
    upm = &g_gauge_field[ix][ZUP];
    _bispinor_null( stmp2 );
    padd( &stmp2, rr3, upm, HOP_UP, 2.0*phase_3);
    Fadd( rr, &stmp2, phip[ZUP], -0.125*rho_BSM, -1. );
  }

  MPI_Waitall( count, request, statuses);


//join

  for (ix=0; ix<VOLUME; ++ix){
    rr = (bispinor *) P + ix;
    s  = (bispinor *) Q + ix;

//  prefatch scalar fields
    phi[0] = g_scalar_field[0][ix];
    phi[1] = g_scalar_field[1][ix];
    phi[2] = g_scalar_field[2][ix];
    phi[3] = g_scalar_field[3][ix];

    for ( int mu=0; mu<4; mu++ )
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

// intermediate buffers for storing backward connections : U_mu( x-mu)^dagg psi(x-mu)
    rr4 = v2m1 + g_idn[ix][TUP];
    rr5 = v2m2 + g_idn[ix][XUP];
    rr6 = v2m3 + g_idn[ix][YUP];
    rr7 = v2m4 + g_idn[ix][ZUP];

    _bispinor_add_mul( *rr, +1.0, *rr4 );
    _bispinor_add_mul( *rr, +1.0, *rr5 );
    _bispinor_add_mul( *rr, +1.0, *rr6 );
    _bispinor_add_mul( *rr, +1.0, *rr7 );

    Fadd(rr, s, phi, eta_BSM, - 1. );

    // tmpr += \sum_\mu (\rho_BSM/8) * F(x+-\mu)*Q
    for( int mu=0; mu<4; mu++ ) {
       Fadd(rr, s, phip[mu], 0.125*rho_BSM, -1. );
       Fadd(rr, s, phim[mu], 0.125*rho_BSM, -1. );
    }


  } // end volume loop
  free(request);
}
/* Q2_psi_BSM2f acts on bispinor fields */
void Q2_psi_BSM2f(bispinor * const P, bispinor * const Q){

  /* TODO: the use of [3] has to be changed to avoid future conflicts */
  D_psi_dagger_BSM2f(g_bispinor_field[3] , Q);
  D_psi_BSM2f(P, g_bispinor_field[3]);
  // only use these cycles if the m0_BSM parameter is really nonzero...
  if( fabs(m0_BSM) > 1.e-10 ){
    /* Q and P are spinor, not bispinor ==> made a cast */
    assign_add_mul_r((spinor*)P, (spinor*)Q, m0_BSM, 2*VOLUME);
  }

}
