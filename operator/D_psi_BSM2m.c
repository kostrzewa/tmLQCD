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
# include<config.h>
#endif

#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "global.h"
#include "su3.h"
#include "sse.h"
#include "boundary.h"
#ifdef MPI
# include "xchange/xchange.h"
#endif
#include "update_backward_gauge.h"
#include "block.h"
#include "operator/D_psi.h"
#include "solver/dirac_operator_eigenvectors.h"
#include "buffers/utils.h"
#include "linalg_eo.h"

void Fadd(bispinor * const out, const bispinor * const in, const scalar * const phi, const double c, const double sign) {
#ifdef OMP
#define static
#endif
  static spinor tmp;
#ifdef OMP
#undef static
#endif
  
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

static inline void bispinor_times_phase_times_u(bispinor * restrict const us, const _Complex double phase,
						su3 const * restrict const u, bispinor const * restrict const s)
{
#ifdef OMP
#define static
#endif
  static su3_vector chi;
#ifdef OMP
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
#ifdef OMP
#define static
#endif
  static su3_vector chi;
#ifdef OMP
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

/********************************************************************************************************
 *  D_psi_BSM2
 * 
 * 
 * *************************************************************/

static inline void p0add_gamma(bispinor * restrict const tmpr , bispinor const * restrict const s,
                         su3 const * restrict const u, const int inv, const _Complex double phase) {

#ifdef OMP
#define static
#endif
  static bispinor us;
#ifdef OMP
#undef static
#endif

  // us = phase*u*s
  if( inv )
    bispinor_times_phase_times_inverse_u(&us, phase, u, s); 
  else
    bispinor_times_phase_times_u(&us, phase, u, s);

  // tmpr += \gamma_0*us
  _vector_add_assign(tmpr->sp_up.s0, us.sp_up.s2);
  _vector_add_assign(tmpr->sp_up.s1, us.sp_up.s3);
  _vector_add_assign(tmpr->sp_up.s2, us.sp_up.s0);
  _vector_add_assign(tmpr->sp_up.s3, us.sp_up.s1);

  _vector_add_assign(tmpr->sp_dn.s0, us.sp_dn.s2);
  _vector_add_assign(tmpr->sp_dn.s1, us.sp_dn.s3);
  _vector_add_assign(tmpr->sp_dn.s2, us.sp_dn.s0);
  _vector_add_assign(tmpr->sp_dn.s3, us.sp_dn.s1);

 
  return;
}



static inline void p1add_gamma(bispinor * restrict const tmpr, bispinor const * restrict const s,
                         su3 const * restrict const u, const int inv, const _Complex double phase) {
#ifdef OMP
#define static
#endif
  static bispinor us;
#ifdef OMP
#undef static
#endif

  // us = phase*u*s
  if( inv )
    bispinor_times_phase_times_inverse_u(&us, phase, u, s);
  else
    bispinor_times_phase_times_u(&us, phase, u, s);

  // tmpr += \gamma_0*us
  _vector_i_add_assign(tmpr->sp_up.s0, us.sp_up.s3);
  _vector_i_add_assign(tmpr->sp_up.s1, us.sp_up.s2);
  _vector_i_sub_assign(tmpr->sp_up.s2, us.sp_up.s1);
  _vector_i_sub_assign(tmpr->sp_up.s3, us.sp_up.s0);

  _vector_i_add_assign(tmpr->sp_dn.s0, us.sp_dn.s3);
  _vector_i_add_assign(tmpr->sp_dn.s1, us.sp_dn.s2);
  _vector_i_sub_assign(tmpr->sp_dn.s2, us.sp_dn.s1);
  _vector_i_sub_assign(tmpr->sp_dn.s3, us.sp_dn.s0);


  return;
}

static inline void p2add_gamma(bispinor * restrict const tmpr, bispinor const * restrict const s,
                         su3 const * restrict const u, const int inv, const _Complex double phase) {
#ifdef OMP
#define static
#endif
  static bispinor us;
#ifdef OMP
#undef static
#endif

  // us = phase*u*s
  if( inv )
    bispinor_times_phase_times_inverse_u(&us, phase, u, s);
  else
    bispinor_times_phase_times_u(&us, phase, u, s);

  // tmpr += \gamma_0*us
  _vector_add_assign(tmpr->sp_up.s0, us.sp_up.s3);
  _vector_sub_assign(tmpr->sp_up.s1, us.sp_up.s2);
  _vector_sub_assign(tmpr->sp_up.s2, us.sp_up.s1);
  _vector_add_assign(tmpr->sp_up.s3, us.sp_up.s0);

  _vector_add_assign(tmpr->sp_dn.s0, us.sp_dn.s3);
  _vector_sub_assign(tmpr->sp_dn.s1, us.sp_dn.s2);
  _vector_sub_assign(tmpr->sp_dn.s2, us.sp_dn.s1);
  _vector_add_assign(tmpr->sp_dn.s3, us.sp_dn.s0);

  return;
}

static inline void p3add_gamma(bispinor * restrict const tmpr, bispinor const * restrict const s,
                         su3 const * restrict const u, const int inv, const _Complex double phase) {
#ifdef OMP
#define static
#endif
  static bispinor us;
#ifdef OMP
#undef static
#endif

  // us = phase*u*s
  if( inv )
    bispinor_times_phase_times_inverse_u(&us, phase, u, s);
  else
    bispinor_times_phase_times_u(&us, phase, u, s);

  // tmpr += \gamma_0*us
  _vector_i_add_assign(tmpr->sp_up.s0, us.sp_up.s2);
  _vector_i_sub_assign(tmpr->sp_up.s1, us.sp_up.s3);
  _vector_i_sub_assign(tmpr->sp_up.s2, us.sp_up.s0);
  _vector_i_add_assign(tmpr->sp_up.s3, us.sp_up.s1);

  _vector_i_add_assign(tmpr->sp_dn.s0, us.sp_dn.s2);
  _vector_i_sub_assign(tmpr->sp_dn.s1, us.sp_dn.s3);
  _vector_i_sub_assign(tmpr->sp_dn.s2, us.sp_dn.s0);
  _vector_i_add_assign(tmpr->sp_dn.s3, us.sp_dn.s1);


  return;
}

static inline void padd_F(bispinor * restrict const tmpr , bispinor const * restrict const s,
                         su3 const * restrict const u, const int inv, const _Complex double phase,
                         const double phaseF, const scalar * const phi,
                         const double sign) {
#ifdef OMP
#define static
#endif
  static bispinor us;
#ifdef OMP
#undef static
#endif

  // us = phase*u*s
  if( inv )
    bispinor_times_phase_times_inverse_u(&us, phase, u, s);
  else
    bispinor_times_phase_times_u(&us, phase, u, s);

 
  // tmpr += F*us
  Fadd(tmpr, &us, phi,  phaseF, sign);
 /*  Fadd(tmpr, &us, phip, phaseF, sign);*/

  return;
}


/* D_psi_BSM2m acts on bispinor fields */
void D_psi_BSM2m(bispinor * const P, bispinor * const Q){
  if(P==Q){
    printf("Error in D_psi_BSM2m (D_psi_BSM2m.c):\n");
    printf("Arguments must be different bispinor fields\n");
    printf("Program aborted\n");
    exit(1);
  }
#ifdef _GAUGE_COPY
  if(g_update_gauge_copy) {
    update_backward_gauge(g_gauge_field);
  }
#endif
#ifdef MPI
  generic_exchange(Q, sizeof(bispinor));
#endif
        
#ifdef OMP
#pragma omp parallel
  {
#endif

    int ix,iy;                       // x, x+-\mu
    su3 * restrict up,* restrict um, tempuu, * restrict uu ; // U_\mu(x), U_\mu(x-\mu)
    bispinor * restrict rr;          // P(x)
    bispinor const * restrict s;     // Q(x)
    bispinor const * restrict sp;    // Q(x+\mu)
    bispinor const * restrict sm;    // Q(x-\mu)
    scalar phi[4];                   // phi_i(x)
    scalar phip[4][4];               // phi_i(x+mu) = phip[mu][i]
    scalar phim[4][4];               // phi_i(x-mu) = phim[mu][i]
    _Complex double c_phase_00,c_phase_11,c_phase_22,c_phase_33;
    c_phase_00=conj(phase_00);
    c_phase_11=conj(phase_11);
    c_phase_22=conj(phase_22);
    c_phase_33=conj(phase_33);

/************************ loop over all lattice sites *************************/
#ifdef OMP
#pragma omp for
#endif
    for (ix=0;ix<VOLUME;ix++)
      {
	rr = (bispinor *) P + ix;
	s  = (bispinor *) Q + ix;
//printf("%d\n",ix);
	// prefatch scalar fields
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

	// the local part (not local in phi)

	// tmpr = m0_BSM*Q(x)
	//_vector_mul(rr->sp_up.s0, m0_BSM, s->sp_up.s0);
	//_vector_mul(rr->sp_up.s1, m0_BSM, s->sp_up.s1);
	//_vector_mul(rr->sp_up.s2, m0_BSM, s->sp_up.s2);
	//_vector_mul(rr->sp_up.s3, m0_BSM, s->sp_up.s3);
	//_vector_mul(rr->sp_dn.s0, m0_BSM, s->sp_dn.s0);
	//_vector_mul(rr->sp_dn.s1, m0_BSM, s->sp_dn.s1);
	//_vector_mul(rr->sp_dn.s2, m0_BSM, s->sp_dn.s2);
	//_vector_mul(rr->sp_dn.s3, m0_BSM, s->sp_dn.s3);
	// no longer needed now...
	_spinor_null(rr->sp_up);
	_spinor_null(rr->sp_dn);


	// tmpr += (\eta_BSM) * F(x)*Q(x)
	Fadd(rr, s, phi, eta_BSM/*+2.0*rho_BSM*/, +1.);

	// tmpr += \sum_\mu (\rho_BSM/8) * F(x+-\mu)*Q
	for( int mu=0; mu<4; mu++ ) {
	  Fadd(rr, s, phip[mu], 0.125*rho_BSM, +1.);
	  Fadd(rr, s, phim[mu], 0.125*rho_BSM, +1.);
	}


	// the hopping part:
	// tmpr += +-1/2 \sum_\mu (\gamma_\mu -+ \rho_BSM/2*F(x) -+ \rho_BSM/2*F(x+-\mu)*U_{+-\mu}(x)*Q(x+-\mu)
	/******************************* direction +0 *********************************/
	iy=g_iup[ix][0];
	sp = (bispinor *) Q +iy;
	up=&g_gauge_field[ix][0];
	p0add_gamma(rr, sp, up, 0, 0.5*phase_0);

        um=&g_gauge_field[iy][0];
        iy=g_iup[iy][0];
        sp = (bispinor *) Q +iy;
        _su3_times_su3(tempuu,*up,*um);uu=&tempuu;
        padd_F(rr, sp, uu, 0, phase_00, -0.125*rho_BSM, phip[0], +1.);
        
	/******************************* direction -0 *********************************/
	iy=g_idn[ix][0];
	sm      = (bispinor *) Q +iy;
	um=&g_gauge_field[iy][0];
	p0add_gamma(rr, sm, um, 1, -0.5*phase_0);
        
        iy=g_idn[iy][0];
        up=&g_gauge_field[iy][0];
        sm = (bispinor *) Q +iy;
        _su3d_times_su3d(tempuu,*um,*up);uu=&tempuu;//it gives the dagger operator, so applay it without the inversion
        padd_F(rr, sm, uu, 0, c_phase_00, -0.125*rho_BSM, phim[0], +1.);


	/******************************* direction +1 *********************************/
	iy=g_iup[ix][1];
	sp = (bispinor *) Q +iy;
	up=&g_gauge_field[ix][1];
	p1add_gamma(rr, sp, up, 0, 0.5*phase_1);

        um=&g_gauge_field[iy][1];
        iy=g_iup[iy][1];
        sp = (bispinor *) Q +iy;
        _su3_times_su3(tempuu,*up,*um);uu=&tempuu;
        padd_F(rr, sp, uu, 0, phase_11, -0.125*rho_BSM, phip[1], +1.);
        
	/******************************* direction -1 *********************************/
	iy=g_idn[ix][1];
	sm      = (bispinor *) Q +iy;
	um=&g_gauge_field[iy][1];
	p1add_gamma(rr, sm, um, 1, -0.5*phase_1);
        
        iy=g_idn[iy][1];
        up=&g_gauge_field[iy][1];
        sm = (bispinor *) Q +iy;
        _su3d_times_su3d(tempuu,*um,*up);uu=&tempuu;
        padd_F(rr, sm, uu, 0, c_phase_11, -0.125*rho_BSM, phim[1], +1.);

	/******************************* direction +2 *********************************/
	iy=g_iup[ix][2];
	sp = (bispinor *) Q +iy;
	up=&g_gauge_field[ix][2];
	p2add_gamma(rr, sp, up, 0, 0.5*phase_2);

        um=&g_gauge_field[iy][2];
        iy=g_iup[iy][2];
        sp = (bispinor *) Q +iy;
        _su3_times_su3(tempuu,*up,*um);uu=&tempuu;
        padd_F(rr, sp, uu, 0, phase_22, -0.125*rho_BSM, phip[2], +1.);
        
	/******************************* direction -2 *********************************/
	iy=g_idn[ix][2];
	sm      = (bispinor *) Q +iy;
	um=&g_gauge_field[iy][2];
	p2add_gamma(rr, sm, um, 1, -0.5*phase_2);
        
        iy=g_idn[iy][2];
        up=&g_gauge_field[iy][2];
        sm = (bispinor *) Q +iy;
        _su3d_times_su3d(tempuu,*um,*up);uu=&tempuu;
        padd_F(rr, sm, uu, 0, c_phase_22, -0.125*rho_BSM, phim[2], +1.);

	/******************************* direction +3 *********************************/
	iy=g_iup[ix][3];
	sp = (bispinor *) Q +iy;
	up=&g_gauge_field[ix][3];
	p3add_gamma(rr, sp, up, 0, 0.5*phase_3);

        um=&g_gauge_field[iy][3];
        iy=g_iup[iy][3];
        sp = (bispinor *) Q +iy;
        _su3_times_su3(tempuu,*up,*um);uu=&tempuu;
        padd_F(rr, sp, uu, 0, phase_33, -0.125*rho_BSM, phip[3], +1.);
   
	/******************************* direction -3 *********************************/
	iy=g_idn[ix][3];
	sm      = (bispinor *) Q +iy;
	um=&g_gauge_field[iy][3];
	p3add_gamma(rr, sm, um, 1, -0.5*phase_3);
        
        iy=g_idn[iy][3];
        up=&g_gauge_field[iy][3];
        sm = (bispinor *) Q +iy;
         _su3d_times_su3d(tempuu,*um,*up);uu=&tempuu;
        padd_F(rr, sm, uu, 0, c_phase_33, -0.125*rho_BSM, phim[3], +1.);
 
        
      }
#ifdef OMP
  } /* OpenMP closing brace */
#endif
}

/* D_psi_dagger_BSM2m acts on bispinor fields */
void D_psi_dagger_BSM2m(bispinor * const P, bispinor * const Q){
  if(P==Q){
    printf("Error in D_psi_BSM2m (D_psi_BSM2m.c):\n");
    printf("Arguments must be different bispinor fields\n");
    printf("Program aborted\n");
    exit(1);
  }
#ifdef _GAUGE_COPY
  if(g_update_gauge_copy) {
    update_backward_gauge(g_gauge_field);
  }
#endif
#ifdef MPI
  generic_exchange(Q, sizeof(bispinor));
#endif
        
#ifdef OMP
#pragma omp parallel
  {
#endif

    int ix,iy;                       // x, x+-\mu
    su3 * restrict up,* restrict um,  tempuu, *restrict uu; // U_\mu(x), U_\mu(x-\mu)
    bispinor * restrict rr;          // P(x)
    bispinor const * restrict s;     // Q(x)
    bispinor const * restrict sp;    // Q(x+\mu)
    bispinor const * restrict sm;    // Q(x-\mu)
    scalar phi[4];                   // phi_i(x)
    scalar phip[4][4];               // phi_i(x+mu) = phip[mu][i]
    scalar phim[4][4];               // phi_i(x-mu) = phim[mu][i]
    _Complex double c_phase_00,c_phase_11,c_phase_22,c_phase_33;
    c_phase_00=conj(phase_00);
    c_phase_11=conj(phase_11);
    c_phase_22=conj(phase_22);
    c_phase_33=conj(phase_33);


    /************************ loop over all lattice sites *************************/

#ifdef OMP
#pragma omp for
#endif
    for (ix=0;ix<VOLUME;ix++)
      {
	rr = (bispinor *) P + ix;
	s  = (bispinor *) Q + ix;

	// prefatch scalar fields
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

	// the local part (not local in phi)

	// tmpr = m0_BSM*Q(x)
	//_vector_mul(rr->sp_up.s0, m0_BSM, s->sp_up.s0);
	//_vector_mul(rr->sp_up.s1, m0_BSM, s->sp_up.s1);
	//_vector_mul(rr->sp_up.s2, m0_BSM, s->sp_up.s2);
	//_vector_mul(rr->sp_up.s3, m0_BSM, s->sp_up.s3);
	//_vector_mul(rr->sp_dn.s0, m0_BSM, s->sp_dn.s0);
	//_vector_mul(rr->sp_dn.s1, m0_BSM, s->sp_dn.s1);
	//_vector_mul(rr->sp_dn.s2, m0_BSM, s->sp_dn.s2);
	//_vector_mul(rr->sp_dn.s3, m0_BSM, s->sp_dn.s3);
	// no longer needed now...
	_spinor_null(rr->sp_up);
	_spinor_null(rr->sp_dn);


	// tmpr += (\eta_BSM) * F(x)*Q(x)
	Fadd(rr, s, phi, eta_BSM/*+2.0*rho_BSM*/, -1.);

	// tmpr += \sum_\mu (\rho_BSM/8) * F(x+-\mu)*Q
	for( int mu=0; mu<4; mu++ ) {
	  Fadd(rr, s, phip[mu], 0.125*rho_BSM, -1.);
	  Fadd(rr, s, phim[mu], 0.125*rho_BSM, -1.);
	}


	// the hopping part:
	// tmpr += +-1/2 \sum_\mu (\gamma_\mu -+ \rho_BSM/2*F(x) -+ \rho_BSM/2*F(x+-\mu)*U_{+-\mu}(x)*Q(x+-\mu)
	/******************************* direction +0 *********************************/
	iy=g_iup[ix][0];
	sp = (bispinor *) Q +iy;
	up=&g_gauge_field[ix][0];
	p0add_gamma(rr, sp, up, 0, -0.5*phase_0);

        um=&g_gauge_field[iy][0];
        iy=g_iup[iy][0];
        sp = (bispinor *) Q +iy;
        _su3_times_su3(tempuu,*up,*um);uu=&tempuu;
        padd_F(rr, sp, uu, 0, phase_00, -0.125*rho_BSM, phip[0], -1.);
        
	/******************************* direction -0 *********************************/
	iy=g_idn[ix][0];
	sm      = (bispinor *) Q +iy;
	um=&g_gauge_field[iy][0];
	p0add_gamma(rr, sm, um, 1, +0.5*phase_0);
        
        iy=g_idn[iy][0];
        up=&g_gauge_field[iy][0];
        sm = (bispinor *) Q +iy;
         _su3d_times_su3d(tempuu,*um,*up);uu=&tempuu;
        padd_F(rr, sm, uu, 0, c_phase_00, -0.125*rho_BSM, phim[0], -1.);


	/******************************* direction +1 *********************************/
	iy=g_iup[ix][1];
	sp = (bispinor *) Q +iy;
	up=&g_gauge_field[ix][1];
	p1add_gamma(rr, sp, up, 0, -0.5*phase_1);

        um=&g_gauge_field[iy][1];
        iy=g_iup[iy][1];
        sp = (bispinor *) Q +iy;
        _su3_times_su3(tempuu,*up,*um);uu=&tempuu;
        padd_F(rr, sp, uu, 0, phase_11, -0.125*rho_BSM, phip[1], -1.);
        
	/******************************* direction -1 *********************************/
	iy=g_idn[ix][1];
	sm      = (bispinor *) Q +iy;
	um=&g_gauge_field[iy][1];
	p1add_gamma(rr, sm, um, 1, +0.5*phase_1);
        
        iy=g_idn[iy][1];
        up=&g_gauge_field[iy][1];
        sm = (bispinor *) Q +iy;
        _su3d_times_su3d(tempuu,*um,*up);uu=&tempuu;
        padd_F(rr, sm, uu, 0, c_phase_11, -0.125*rho_BSM, phim[1], -1.);

	/******************************* direction +2 *********************************/
	iy=g_iup[ix][2];
	sp = (bispinor *) Q +iy;
	up=&g_gauge_field[ix][2];
	p2add_gamma(rr, sp, up, 0, -0.5*phase_2);

        um=&g_gauge_field[iy][2];
        iy=g_iup[iy][2];
        sp = (bispinor *) Q +iy;
        _su3_times_su3(tempuu,*up,*um);uu=&tempuu;
        padd_F(rr, sp, uu, 0, phase_22, -0.125*rho_BSM, phip[2], -1.);
        
	/******************************* direction -2 *********************************/
	iy=g_idn[ix][2];
	sm      = (bispinor *) Q +iy;
	um=&g_gauge_field[iy][2];
	p2add_gamma(rr, sm, um, 1, +0.5*phase_2);
        
        iy=g_idn[iy][2];
        up=&g_gauge_field[iy][2];
        sm = (bispinor *) Q +iy;
        _su3d_times_su3d(tempuu,*um,*up);uu=&tempuu;
        padd_F(rr, sm, uu, 0, c_phase_22, -0.125*rho_BSM, phim[2], -1.);

	/******************************* direction +3 *********************************/
	iy=g_iup[ix][3];
	sp = (bispinor *) Q +iy;
	up=&g_gauge_field[ix][3];
	p3add_gamma(rr, sp, up, 0, -0.5*phase_3);

        um=&g_gauge_field[iy][3];
        iy=g_iup[iy][3];
        sp = (bispinor *) Q +iy;
        _su3_times_su3(tempuu,*up,*um);uu=&tempuu;
        padd_F(rr, sp, uu, 0, phase_33, -0.125*rho_BSM, phip[3], -1.);
        
	/******************************* direction -3 *********************************/
	iy=g_idn[ix][3];
	sm      = (bispinor *) Q +iy;
	um=&g_gauge_field[iy][3];
	p3add_gamma(rr, sm, um, 1, +0.5*phase_3);
        
        iy=g_idn[iy][3];
        up=&g_gauge_field[iy][3];
        sm = (bispinor *) Q +iy;
        _su3d_times_su3d(tempuu,*um,*up);uu=&tempuu;
        padd_F(rr, sm, uu, 0, c_phase_33, -0.125*rho_BSM, phim[3], -1.);

    }
#ifdef OMP
  } /* OpenMP closing brace */
#endif
}

void Q2_psi_BSM2m(bispinor * const P, bispinor * const Q){

  /* the use of [3] has to be changed to avoid future conflicts */
  D_psi_dagger_BSM2m(g_bispinor_field[3] , Q);
  D_psi_BSM2m(P, g_bispinor_field[3]);
  // only use these cycles if the m0_BSM parameter is really nonzero...
  if( fabs(m0_BSM) > 1.e-10 ){
    /* Q and P are spinor, not bispinor ==> made a cast */
    assign_add_mul_r((spinor*)P, (spinor*)Q, m0_BSM, 2*VOLUME);
  }

}

