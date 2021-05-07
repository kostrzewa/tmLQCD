/***********************************************************************
 *
 * Copyright (C) 2015 Ferenc Pittler
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
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with tmLQCD.  If not, see <http://www.gnu.org/licenses/>.
 ***********************************************************************/

#ifndef _CONTRACTIONS_HELPER_H
#define _CONTRACTIONS_HELPER_H

_Complex double bispinor_scalar_product ( bispinor *s1, bispinor *s2 );
void multiply_backward_propagator( bispinor *dest, bispinor **propagator, bispinor *source, int idx, int dir);
void bispinor_mult_su3matrix( bispinor *dest, bispinor *source, su3 *a, int dagger);
void bispinor_spindown_mult_su3matrix( bispinor *dest, bispinor *source, su3 *a, int dagger);
void bispinor_spinup_mult_su3matrix( bispinor *dest, bispinor *source, su3 *a, int dagger);
void bispinor_timesgamma0( bispinor *dest);
void bispinor_timesgamma5( bispinor *dest);
void bispinor_taui( bispinor *dest, int tauindex);
void taui_scalarfield_flavoronly( _Complex double *dest, int tauindex, int dagger, int dir );
void taui_scalarfield_flavoronly_s0s0( _Complex double *dest, int dagger );
void mult_phi_flavoronly( _Complex double *dest, int dagg);
void mult_taui_flavoronly( _Complex double *dest, int tauindex);
void mult_phi( bispinor *dest, bispinor *source, int ix, int dagg);
void taui_spinor( bispinor *dest, bispinor *source, int tauindex );
void phi0_taui_commutator( _Complex double *dest,int tauindex );
void phix_taui_commutator_bispinor( bispinor *dest,int tauindex, int gamma5, int ix );
void phi0_taui_anticommutator( _Complex double *dest, int tauindex, int dagger );
void taui_scalarfield_spinor_s0s0( bispinor *dest, bispinor *source, int gamma5, int idx, int direction, int dagger);
void taui_scalarfield_spinor( bispinor *dest, bispinor *source, int gamma5, int tauindex, int idx, int direction, int dagger);
void trace_in_spinor( _Complex double *dest, _Complex double *src, int spinorindex);
void trace_in_color(_Complex double *dest, bispinor *src, int colorindex);
void trace_in_space(_Complex double *dest, _Complex double *source, int idx);
void trace_in_flavor(_Complex double *dest, _Complex double *source, int f1);
void phix_taui_anti_commutator_bispinor( bispinor *dest,int tauindex, int gamma5, int dagger,int ix );
#endif
