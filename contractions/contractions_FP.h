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

#ifndef _CONTRACTIONS_FP_H
#define _CONTRACTIONS_FP_H
void density_density_1234(bispinor **propagators, int type, _Complex double **res);
void density_density_1234_s0s0( bispinor ** propagators, int type, _Complex double **res );
void density_density_1234_sxsx( bispinor ** propagators, int type, _Complex double **res );
void vector_2_psueodoscalar_1_12( bispinor ** propfields, int type_12,_Complex double **results );
void naivedirac_current_density_12ab( bispinor ** propagators, int type_12, int type_ab,  _Complex double **tes );
void naivedirac_current_density_12ab_lr( bispinor ** propagators, int type_12, int type_ab,  _Complex double **tes );
void wilsonterm_current_density_312ab( bispinor ** propagators, int type_12, int type_ab, _Complex double **res );
void wilsonterm_current_density_412ab( bispinor ** propagators, int type_12, int type_ab, _Complex double **res );
void wilsonterm_current_density_512ab( bispinor ** propagators, int type_12, int type_ab, _Complex double **res );
void wilsonterm_current_density_612ab( bispinor ** propagators, int type_12, int type_ab, _Complex double **res );
void vector_axial_current_density_1234( bispinor ** propfields, int type_1234,int taudensity, int taucurrent, int vectororaxial, int scalarorpseudoscalar,_Complex double **results );
void vector_density_density_1234( bispinor ** propfields, int type_1234,int taudensity, _Complex double **results );
void vector_axial_current_current_1234( bispinor ** propfields_source_zero, bispinor ** propfields_source_ntmone, int type_1234, int taucurrent, int vectororaxial, _Complex double **results );
void giancarlodensity( bispinor ** propfields, int tau3, _Complex double  **results );
void density_ptau_density_vector( bispinor **propfields, int type_12, _Complex double **results);
#endif
