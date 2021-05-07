/***********************************************************************
 *
 * Copyright (C) 2017 Ferenc Pittler
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
#ifdef HAVE_CONFIG_H
# include<tmlqcd_config.h>
#endif

#ifdef TM_USE_BSM
#include"lime.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <time.h>
#ifdef TM_USE_MPI
#include <mpi.h>
#endif
#include "global.h"
#include "getopt.h"
#include "default_input_values.h"
#include "read_input.h"
#include "su3.h"
#include "operator/tm_operators.h"
#include "linalg_eo.h"
#include "geometry_eo.h"
#include "linalg/assign.h"
#include "operator/D_psi.h"
#include "operator/D_psi_BSM.h"
#include "operator/D_psi_BSM2b.h"
#include "operator/D_psi_BSM2f.h"
#include "operator/D_psi_BSM2m.h"
#include "operator/Dov_psi.h"
#include "operator/tm_operators_nd.h"
#include "operator/Hopping_Matrix.h"
#include "invert_eo.h"
#include "invert_doublet_eo.h"
#include "invert_overlap.h"
#include "invert_clover_eo.h"
#include "init/init_scalar_field.h"
#include "init/init_bsm_2hop_lookup.h"
#include "boundary.h"
#include "start.h"
#include "solver/solver.h"
#include "xchange/xchange_gauge.h"
#include "prepare_source.h"
#include <io/params.h>
#include <io/gauge.h>
#include <io/spinor.h>
#include <io/utils.h>
#include "io/scalar.h"
#include "buffers/utils_nonblocking.h"
#include "buffers/utils_nogauge.h"
#include "test/overlaptests.h"
#include "solver/index_jd.h"
#include "operator/clovertm_operators.h"
#include "operator/clover_leaf.h"
#include "operator.h"
#include "gettime.h"
#include "measure_gauge_action.h"
#include "mpi_init.h"
#include "init/init_geometry_indices.h"
#include "init/init_openmp.h"
#include "init/init_gauge_field.h"
#include "init/init_spinor_field.h"
#include "init/init_bispinor_field.h"
#include "solver/solver_field.h"
#include "ranlxd.h"

/* indexing of propfields;
   
   propagator for  (dagger or nondagger source)
              for  flavor component f
              for  color  component c    
              for  spinor component s
   is the following bispinor array of size VOLUME(PLUSRAND)

   propfields[12*s + 4*c + 2*f + dagg ? 1: 0]  
     
 */
/**************************
Multiplication with the backward propagator

S == matrix element of D^-1 between the following states

S( ytilde , x+-dir )       psi   x
   flavor2, flavor1    x         flavor1
   spinor2, spinor1              spinor1
   color 2, color 1              color1

=
Stilde* (x+-dir , ytilde)      psi   x
         flavor1, flavor2  x         flavor1
         spinor1, spinor2            spinor1  
         color 1, color 2            color1
where Stilde is the matrix element of D^dagger^-1 between 
the correspondig states

**************************/

extern int DAGGER;
extern int NO_DAGG;

extern int GAMMA_UP;
extern int GAMMA_DN;
extern int NO_GAMMA;

extern int WITH_SCALAR;
extern int NO_SCALAR;

extern int TYPE_A;
extern int TYPE_B;

extern int TYPE_1;
extern int TYPE_2;
extern int TYPE_3;
extern int TYPE_4;

extern int TYPE_I;
extern int TYPE_II;

extern int RIGHT;
extern int LEFT;


_Complex double bispinor_scalar_product ( bispinor *s1, bispinor *s2 ){
   _Complex double res=0.0;
   res   =s2->sp_up.s0.c0 * conj(s1->sp_up.s0.c0) + s2->sp_up.s0.c1 * conj(s1->sp_up.s0.c1) + s2->sp_up.s0.c2 * conj(s1->sp_up.s0.c2) +
          s2->sp_up.s1.c0 * conj(s1->sp_up.s1.c0) + s2->sp_up.s1.c1 * conj(s1->sp_up.s1.c1) + s2->sp_up.s1.c2 * conj(s1->sp_up.s1.c2) +
          s2->sp_up.s2.c0 * conj(s1->sp_up.s2.c0) + s2->sp_up.s2.c1 * conj(s1->sp_up.s2.c1) + s2->sp_up.s2.c2 * conj(s1->sp_up.s2.c2) +
          s2->sp_up.s3.c0 * conj(s1->sp_up.s3.c0) + s2->sp_up.s3.c1 * conj(s1->sp_up.s3.c1) + s2->sp_up.s3.c2 * conj(s1->sp_up.s3.c2) +
          s2->sp_dn.s0.c0 * conj(s1->sp_dn.s0.c0) + s2->sp_dn.s0.c1 * conj(s1->sp_dn.s0.c1) + s2->sp_dn.s0.c2 * conj(s1->sp_dn.s0.c2) +
          s2->sp_dn.s1.c0 * conj(s1->sp_dn.s1.c0) + s2->sp_dn.s1.c1 * conj(s1->sp_dn.s1.c1) + s2->sp_dn.s1.c2 * conj(s1->sp_dn.s1.c2) +
          s2->sp_dn.s2.c0 * conj(s1->sp_dn.s2.c0) + s2->sp_dn.s2.c1 * conj(s1->sp_dn.s2.c1) + s2->sp_dn.s2.c2 * conj(s1->sp_dn.s2.c2) +
          s2->sp_dn.s3.c0 * conj(s1->sp_dn.s3.c0) + s2->sp_dn.s3.c1 * conj(s1->sp_dn.s3.c1) + s2->sp_dn.s3.c2 * conj(s1->sp_dn.s3.c2);
   return res;
}
void multiply_backward_propagator( bispinor *dest, bispinor **propagator, bispinor *source, int idx, int dir){
   int propcoord;
   bispinor source_copy;
   if (dir == NODIR){
      propcoord=idx;
   } 
   else if (dir == TUP){
      propcoord=g_iup[idx][TUP];
   }
   else if (dir == TDOWN){
      propcoord=g_idn[idx][TUP];
   }
   else{
      propcoord=0;
      if (g_cart_id == 0){ fprintf(stderr,"Wrong direction in multiply backward prop\n"); 
                           exit(1); }
   }

   _spinor_assign( source_copy.sp_dn, source->sp_dn);
   _spinor_assign( source_copy.sp_up, source->sp_up);

   dest->sp_up.s0.c0= bispinor_scalar_product ( &propagator[ 1][propcoord], &source_copy );
   dest->sp_up.s0.c1= bispinor_scalar_product ( &propagator[ 5][propcoord], &source_copy );
   dest->sp_up.s0.c2= bispinor_scalar_product ( &propagator[ 9][propcoord], &source_copy );

   dest->sp_up.s1.c0= bispinor_scalar_product ( &propagator[13][propcoord], &source_copy );
   dest->sp_up.s1.c1= bispinor_scalar_product ( &propagator[17][propcoord], &source_copy );
   dest->sp_up.s1.c2= bispinor_scalar_product ( &propagator[21][propcoord], &source_copy );

   dest->sp_up.s2.c0= bispinor_scalar_product ( &propagator[25][propcoord], &source_copy );
   dest->sp_up.s2.c1= bispinor_scalar_product ( &propagator[29][propcoord], &source_copy );
   dest->sp_up.s2.c2= bispinor_scalar_product ( &propagator[33][propcoord], &source_copy );

   dest->sp_up.s3.c0= bispinor_scalar_product ( &propagator[37][propcoord], &source_copy );
   dest->sp_up.s3.c1= bispinor_scalar_product ( &propagator[41][propcoord], &source_copy );
   dest->sp_up.s3.c2= bispinor_scalar_product ( &propagator[45][propcoord], &source_copy );

   dest->sp_dn.s0.c0= bispinor_scalar_product ( &propagator[ 3][propcoord], &source_copy );
   dest->sp_dn.s0.c1= bispinor_scalar_product ( &propagator[ 7][propcoord], &source_copy );
   dest->sp_dn.s0.c2= bispinor_scalar_product ( &propagator[11][propcoord], &source_copy );

   dest->sp_dn.s1.c0= bispinor_scalar_product ( &propagator[15][propcoord], &source_copy );
   dest->sp_dn.s1.c1= bispinor_scalar_product ( &propagator[19][propcoord], &source_copy );
   dest->sp_dn.s1.c2= bispinor_scalar_product ( &propagator[23][propcoord], &source_copy );

   dest->sp_dn.s2.c0= bispinor_scalar_product ( &propagator[27][propcoord], &source_copy );
   dest->sp_dn.s2.c1= bispinor_scalar_product ( &propagator[31][propcoord], &source_copy );
   dest->sp_dn.s2.c2= bispinor_scalar_product ( &propagator[35][propcoord], &source_copy );

   dest->sp_dn.s3.c0= bispinor_scalar_product ( &propagator[39][propcoord], &source_copy );
   dest->sp_dn.s3.c1= bispinor_scalar_product ( &propagator[43][propcoord], &source_copy );
   dest->sp_dn.s3.c2= bispinor_scalar_product ( &propagator[47][propcoord], &source_copy );
}
void bispinor_mult_su3matrix( bispinor *dest, bispinor *source, su3 *a, int dagger){
   bispinor source_copy;
   _spinor_assign(source_copy.sp_up, source->sp_up);
   _spinor_assign(source_copy.sp_dn, source->sp_dn);

   if (dagger == DAGGER){
     _su3_inverse_multiply(dest->sp_up.s0, *a, source_copy.sp_up.s0);
     _su3_inverse_multiply(dest->sp_up.s1, *a, source_copy.sp_up.s1);
     _su3_inverse_multiply(dest->sp_up.s2, *a, source_copy.sp_up.s2);
     _su3_inverse_multiply(dest->sp_up.s3, *a, source_copy.sp_up.s3);

     _su3_inverse_multiply(dest->sp_dn.s0, *a, source_copy.sp_dn.s0);
     _su3_inverse_multiply(dest->sp_dn.s1, *a, source_copy.sp_dn.s1);
     _su3_inverse_multiply(dest->sp_dn.s2, *a, source_copy.sp_dn.s2);
     _su3_inverse_multiply(dest->sp_dn.s3, *a, source_copy.sp_dn.s3);

     _complexcjg_times_vector(dest->sp_up.s0, phase_0, dest->sp_up.s0);
     _complexcjg_times_vector(dest->sp_up.s1, phase_0, dest->sp_up.s1);
     _complexcjg_times_vector(dest->sp_up.s2, phase_0, dest->sp_up.s2);
     _complexcjg_times_vector(dest->sp_up.s3, phase_0, dest->sp_up.s3);

     _complexcjg_times_vector(dest->sp_dn.s0, phase_0, dest->sp_dn.s0);
     _complexcjg_times_vector(dest->sp_dn.s1, phase_0, dest->sp_dn.s1);
     _complexcjg_times_vector(dest->sp_dn.s2, phase_0, dest->sp_dn.s2);
     _complexcjg_times_vector(dest->sp_dn.s3, phase_0, dest->sp_dn.s3);


   }
   else{
     _su3_multiply(dest->sp_up.s0, *a, source_copy.sp_up.s0);
     _su3_multiply(dest->sp_up.s1, *a, source_copy.sp_up.s1);
     _su3_multiply(dest->sp_up.s2, *a, source_copy.sp_up.s2);
     _su3_multiply(dest->sp_up.s3, *a, source_copy.sp_up.s3);

     _su3_multiply(dest->sp_dn.s0, *a, source_copy.sp_dn.s0);
     _su3_multiply(dest->sp_dn.s1, *a, source_copy.sp_dn.s1);
     _su3_multiply(dest->sp_dn.s2, *a, source_copy.sp_dn.s2);
     _su3_multiply(dest->sp_dn.s3, *a, source_copy.sp_dn.s3);

     _complex_times_vector(dest->sp_up.s0, phase_0, dest->sp_up.s0);
     _complex_times_vector(dest->sp_up.s1, phase_0, dest->sp_up.s1);
     _complex_times_vector(dest->sp_up.s2, phase_0, dest->sp_up.s2);
     _complex_times_vector(dest->sp_up.s3, phase_0, dest->sp_up.s3);

     _complex_times_vector(dest->sp_dn.s0, phase_0, dest->sp_dn.s0);
     _complex_times_vector(dest->sp_dn.s1, phase_0, dest->sp_dn.s1);
     _complex_times_vector(dest->sp_dn.s2, phase_0, dest->sp_dn.s2);
     _complex_times_vector(dest->sp_dn.s3, phase_0, dest->sp_dn.s3);
   }
}
void bispinor_spinup_mult_su3matrix( bispinor *dest, bispinor *source, su3 *a, int dagger){
   bispinor source_copy;
   _bispinor_null(source_copy);
   _vector_assign(source_copy.sp_up.s0, source->sp_up.s0);
   _vector_assign(source_copy.sp_up.s1, source->sp_up.s1);
   _vector_assign(source_copy.sp_dn.s0, source->sp_dn.s0);
   _vector_assign(source_copy.sp_dn.s1, source->sp_dn.s1);

   if (dagger == DAGGER){
     _su3_inverse_multiply(dest->sp_up.s0, *a, source_copy.sp_up.s0);
     _su3_inverse_multiply(dest->sp_up.s1, *a, source_copy.sp_up.s1);

     _su3_inverse_multiply(dest->sp_dn.s0, *a, source_copy.sp_dn.s0);
     _su3_inverse_multiply(dest->sp_dn.s1, *a, source_copy.sp_dn.s1);

     _complexcjg_times_vector(dest->sp_up.s0, phase_0, dest->sp_up.s0);
     _complexcjg_times_vector(dest->sp_up.s1, phase_0, dest->sp_up.s1);

     _complexcjg_times_vector(dest->sp_dn.s0, phase_0, dest->sp_dn.s0);
     _complexcjg_times_vector(dest->sp_dn.s1, phase_0, dest->sp_dn.s1);

   }
   else{
     _su3_multiply(dest->sp_up.s0, *a, source_copy.sp_up.s0);
     _su3_multiply(dest->sp_up.s1, *a, source_copy.sp_up.s1);

     _su3_multiply(dest->sp_dn.s0, *a, source_copy.sp_dn.s0);
     _su3_multiply(dest->sp_dn.s1, *a, source_copy.sp_dn.s1);

     _complex_times_vector(dest->sp_up.s0, phase_0, dest->sp_up.s0);
     _complex_times_vector(dest->sp_up.s1, phase_0, dest->sp_up.s1);

     _complex_times_vector(dest->sp_dn.s0, phase_0, dest->sp_dn.s0);
     _complex_times_vector(dest->sp_dn.s1, phase_0, dest->sp_dn.s1);
   }
}

void bispinor_spindown_mult_su3matrix( bispinor *dest, bispinor *source, su3 *a, int dagger){
   bispinor source_copy;
   _bispinor_null(source_copy);
   _vector_assign(source_copy.sp_up.s2, source->sp_up.s2);
   _vector_assign(source_copy.sp_up.s3, source->sp_up.s3);
   _vector_assign(source_copy.sp_dn.s2, source->sp_dn.s2);
   _vector_assign(source_copy.sp_dn.s3, source->sp_dn.s3);
   
   if (dagger == DAGGER){
     _su3_inverse_multiply(dest->sp_up.s2, *a, source_copy.sp_up.s2);
     _su3_inverse_multiply(dest->sp_up.s3, *a, source_copy.sp_up.s3);

     _su3_inverse_multiply(dest->sp_dn.s2, *a, source_copy.sp_dn.s2);
     _su3_inverse_multiply(dest->sp_dn.s3, *a, source_copy.sp_dn.s3);

     _complexcjg_times_vector(dest->sp_up.s2, phase_0, dest->sp_up.s2);
     _complexcjg_times_vector(dest->sp_up.s3, phase_0, dest->sp_up.s3);

     _complexcjg_times_vector(dest->sp_dn.s2, phase_0, dest->sp_dn.s2);
     _complexcjg_times_vector(dest->sp_dn.s3, phase_0, dest->sp_dn.s3);

   }
   else{
     _su3_multiply(dest->sp_up.s2, *a, source_copy.sp_up.s2);
     _su3_multiply(dest->sp_up.s3, *a, source_copy.sp_up.s3);

     _su3_multiply(dest->sp_dn.s2, *a, source_copy.sp_dn.s2);
     _su3_multiply(dest->sp_dn.s3, *a, source_copy.sp_dn.s3);

     _complex_times_vector(dest->sp_up.s2, phase_0, dest->sp_up.s2);
     _complex_times_vector(dest->sp_up.s3, phase_0, dest->sp_up.s3);

     _complex_times_vector(dest->sp_dn.s2, phase_0, dest->sp_dn.s2);
     _complex_times_vector(dest->sp_dn.s3, phase_0, dest->sp_dn.s3);
   }
}


void bispinor_timesgamma0( bispinor *dest){
   su3_vector tempvec1, tempvec2;

   _vector_assign(  tempvec1, dest->sp_up.s0);
   _vector_assign(  tempvec2, dest->sp_up.s1);
   _vector_assign(  dest->sp_up.s0, dest->sp_up.s2);
   _vector_assign(  dest->sp_up.s1, dest->sp_up.s3);
   _vector_assign(  dest->sp_up.s2, tempvec1);
   _vector_assign(  dest->sp_up.s3, tempvec2);

   _vector_assign(  tempvec1, dest->sp_dn.s0);
   _vector_assign(  tempvec2, dest->sp_dn.s1);
   _vector_assign(  dest->sp_dn.s0, dest->sp_dn.s2);
   _vector_assign(  dest->sp_dn.s1, dest->sp_dn.s3);
   _vector_assign(  dest->sp_dn.s2, tempvec1);
   _vector_assign(  dest->sp_dn.s3, tempvec2);

}
void bispinor_timesgamma5( bispinor *dest){

   _vector_mul(dest->sp_up.s2, -1, dest->sp_up.s2);
   _vector_mul(dest->sp_up.s3, -1, dest->sp_up.s3);
   _vector_mul(dest->sp_dn.s2, -1, dest->sp_dn.s2);
   _vector_mul(dest->sp_dn.s3, -1, dest->sp_dn.s3);

}
void bispinor_taui( bispinor *dest, int tauindex){
   bispinor source_copy;

   _spinor_assign(source_copy.sp_up,  dest->sp_up);
   _spinor_assign(source_copy.sp_dn,  dest->sp_dn);

   if (tauindex == 2){
     _spinor_assign( dest->sp_up, source_copy.sp_up);
     _vector_mul(dest->sp_dn.s0, -1, source_copy.sp_dn.s0);
     _vector_mul(dest->sp_dn.s1, -1, source_copy.sp_dn.s1);
     _vector_mul(dest->sp_dn.s2, -1, source_copy.sp_dn.s2);
     _vector_mul(dest->sp_dn.s3, -1, source_copy.sp_dn.s3);
   }
   if (tauindex == 1){
     _vector_mul(dest->sp_up.s0, -1.*I, source_copy.sp_dn.s0);
     _vector_mul(dest->sp_up.s1, -1.*I, source_copy.sp_dn.s1);
     _vector_mul(dest->sp_up.s2, -1.*I, source_copy.sp_dn.s2);
     _vector_mul(dest->sp_up.s3, -1.*I, source_copy.sp_dn.s3);

     _vector_mul(dest->sp_dn.s0, +1.*I, source_copy.sp_up.s0);
     _vector_mul(dest->sp_dn.s1, +1.*I, source_copy.sp_up.s1);
     _vector_mul(dest->sp_dn.s2, +1.*I, source_copy.sp_up.s2);
     _vector_mul(dest->sp_dn.s3, +1.*I, source_copy.sp_up.s3);

   }
   if (tauindex == 0){
     _spinor_assign( dest->sp_up, source_copy.sp_dn);
     _spinor_assign( dest->sp_dn, source_copy.sp_up);
   }
}

//dest used as a source, an output it is overwritten
void taui_scalarfield_flavoronly( _Complex double *dest, int tauindex, int dagger, int dir ){
   _Complex double *source_copy;
   _Complex double a11=0.0, a12=0.0, a21=0.0, a22=0.0;
   int i;
  
   source_copy=(_Complex double *)malloc(sizeof(_Complex double)*2*T_global);

   if (source_copy == NULL) {
     if (g_cart_id == 0) {printf("memory allocation failed\n"); exit(1);}
   }
   for (i=0; i<2*T_global; ++i)
     source_copy[i]=dest[i];

   if (dir == LEFT){   
     if (dagger == DAGGER){
       if (tauindex == 0){
         if (smearedcorrelator_BSM == 1){
           a11=  -1.*g_smeared_scalar_field[2][0] - I*g_smeared_scalar_field[1][0];
           a12=  +1.*g_smeared_scalar_field[0][0] - I*g_smeared_scalar_field[3][0];

           a21=  +1.*g_smeared_scalar_field[0][0] + I*g_smeared_scalar_field[3][0];
           a22=  +1.*g_smeared_scalar_field[2][0] - I*g_smeared_scalar_field[1][0];
         }
         else{
           a11=  -1.*g_scalar_field[2][0] - I*g_scalar_field[1][0];
           a12=  +1.*g_scalar_field[0][0] - I*g_scalar_field[3][0];

           a21=  +1.*g_scalar_field[0][0] + I*g_scalar_field[3][0];
           a22=  +1.*g_scalar_field[2][0] - I*g_scalar_field[1][0];
         }
       }
       else  if (tauindex == 1){
         if (smearedcorrelator_BSM == 1) {
           a11=  +1.*g_smeared_scalar_field[1][0] - I*g_smeared_scalar_field[2][0];
           a12=  -1.*g_smeared_scalar_field[3][0] - I*g_smeared_scalar_field[0][0];

           a21=  -1.*g_smeared_scalar_field[3][0] + I*g_smeared_scalar_field[0][0];
           a22=  -1.*g_smeared_scalar_field[1][0] - I*g_smeared_scalar_field[2][0];
         }
         else{
           a11=  +1.*g_scalar_field[1][0] - I*g_scalar_field[2][0];
           a12=  -1.*g_scalar_field[3][0] - I*g_scalar_field[0][0];

           a21=  -1.*g_scalar_field[3][0] + I*g_scalar_field[0][0];
           a22=  -1.*g_scalar_field[1][0] - I*g_scalar_field[2][0];
         }
       }
       else  if (tauindex == 2){
         if (smearedcorrelator_BSM == 1){ 
           a11=  +1.*g_smeared_scalar_field[0][0] - I*g_smeared_scalar_field[3][0];
           a12=  +1.*g_smeared_scalar_field[2][0] + I*g_smeared_scalar_field[1][0];

           a21=  +1.*g_smeared_scalar_field[2][0] - I*g_smeared_scalar_field[1][0];
           a22=  -1.*g_smeared_scalar_field[0][0] - I*g_smeared_scalar_field[3][0];
         }
         else{
           a11=  +1.*g_scalar_field[0][0] - I*g_scalar_field[3][0];
           a12=  +1.*g_scalar_field[2][0] + I*g_scalar_field[1][0];

           a21=  +1.*g_scalar_field[2][0] - I*g_scalar_field[1][0];
           a22=  -1.*g_scalar_field[0][0] - I*g_scalar_field[3][0];
         }
       }
     }
     else if (dagger == NO_DAGG){
       if (tauindex == 0){
         if (smearedcorrelator_BSM == 1){
           a11=  -1.*g_smeared_scalar_field[2][0] + I*g_smeared_scalar_field[1][0];
           a12=  +1.*g_smeared_scalar_field[0][0] - I*g_smeared_scalar_field[3][0];

           a21=  +1.*g_smeared_scalar_field[0][0] + I*g_smeared_scalar_field[3][0];
           a22=  +1.*g_smeared_scalar_field[2][0] + I*g_smeared_scalar_field[1][0];
         }
         else{
           a11=  -1.*g_scalar_field[2][0] + I*g_scalar_field[1][0];
           a12=  +1.*g_scalar_field[0][0] - I*g_scalar_field[3][0];

           a21=  +1.*g_scalar_field[0][0] + I*g_scalar_field[3][0];
           a22=  +1.*g_scalar_field[2][0] + I*g_scalar_field[1][0];
         }
       }
       else if (tauindex == 1){
         if (smearedcorrelator_BSM == 1){
           a11=  +1.*g_smeared_scalar_field[1][0] + I*g_smeared_scalar_field[2][0];
           a12=  -1.*g_smeared_scalar_field[3][0] - I*g_smeared_scalar_field[0][0];

           a21=  -1.*g_smeared_scalar_field[3][0] + I*g_smeared_scalar_field[0][0];
           a22=  -1.*g_smeared_scalar_field[1][0] + I*g_smeared_scalar_field[2][0];
         }
         else{
           a11=  +1.*g_scalar_field[1][0] + I*g_scalar_field[2][0];
           a12=  -1.*g_scalar_field[3][0] - I*g_scalar_field[0][0];

           a21=  -1.*g_scalar_field[3][0] + I*g_scalar_field[0][0];
           a22=  -1.*g_scalar_field[1][0] + I*g_scalar_field[2][0];
         }
       }
       else if (tauindex == 2){
         if (smearedcorrelator_BSM == 1){
           a11=  +1.*g_smeared_scalar_field[0][0] + I*g_smeared_scalar_field[3][0];
           a12=  +1.*g_smeared_scalar_field[2][0] + I*g_smeared_scalar_field[1][0];

           a21=  +1.*g_smeared_scalar_field[2][0] - I*g_smeared_scalar_field[1][0];
           a22=  -1.*g_smeared_scalar_field[0][0] + I*g_smeared_scalar_field[3][0];
         }
         else{
           a11=  +1.*g_scalar_field[0][0] + I*g_scalar_field[3][0];
           a12=  +1.*g_scalar_field[2][0] + I*g_scalar_field[1][0];

           a21=  +1.*g_scalar_field[2][0] - I*g_scalar_field[1][0];
           a22=  -1.*g_scalar_field[0][0] + I*g_scalar_field[3][0];
         }
       }
     }
   }
   else if ( dir == RIGHT ){
     if (dagger == DAGGER){
       if (tauindex == 0){
         if (smearedcorrelator_BSM == 1){
           a11=  +1.*g_smeared_scalar_field[2][0] - I*g_smeared_scalar_field[1][0];
           a12=  +1.*g_smeared_scalar_field[0][0] + I*g_smeared_scalar_field[3][0];

           a21=  +1.*g_smeared_scalar_field[0][0] - I*g_smeared_scalar_field[3][0];
           a22=  -1.*g_smeared_scalar_field[2][0] - I*g_smeared_scalar_field[1][0];
         }
         else{
           a11=  +1.*g_scalar_field[2][0] - I*g_scalar_field[1][0];
           a12=  +1.*g_scalar_field[0][0] + I*g_scalar_field[3][0];

           a21=  +1.*g_scalar_field[0][0] - I*g_scalar_field[3][0];
           a22=  -1.*g_scalar_field[2][0] - I*g_scalar_field[1][0];
         }
       }
       else  if (tauindex == 1){
         if (smearedcorrelator_BSM == 1) {
           a11=  -1.*g_smeared_scalar_field[1][0] - I*g_smeared_scalar_field[2][0];
           a12=  +1.*g_smeared_scalar_field[3][0] - I*g_smeared_scalar_field[0][0];

           a21=  +1.*g_smeared_scalar_field[3][0] + I*g_smeared_scalar_field[0][0];
           a22=  +1.*g_smeared_scalar_field[1][0] - I*g_smeared_scalar_field[2][0];
         }
         else{
           a11=  -1.*g_scalar_field[1][0] - I*g_scalar_field[2][0];
           a12=  +1.*g_scalar_field[3][0] - I*g_scalar_field[0][0];

           a21=  +1.*g_scalar_field[3][0] + I*g_scalar_field[0][0];
           a22=  +1.*g_scalar_field[1][0] - I*g_scalar_field[2][0];
         }
       }
       else  if (tauindex == 2){
         if (smearedcorrelator_BSM == 1){
           a11=  +1.*g_smeared_scalar_field[0][0] - I*g_smeared_scalar_field[3][0];
           a12=  -1.*g_smeared_scalar_field[2][0] - I*g_smeared_scalar_field[1][0];

           a21=  -1.*g_smeared_scalar_field[2][0] + I*g_smeared_scalar_field[1][0];
           a22=  -1.*g_smeared_scalar_field[0][0] - I*g_smeared_scalar_field[3][0];
         }
         else{
           a11=  +1.*g_scalar_field[0][0] - I*g_scalar_field[3][0];
           a12=  -1.*g_scalar_field[2][0] - I*g_scalar_field[1][0];

           a21=  -1.*g_scalar_field[2][0] + I*g_scalar_field[1][0];
           a22=  -1.*g_scalar_field[0][0] - I*g_scalar_field[3][0];
         }
       }
     }
     else if (dagger == NO_DAGG){
       if (tauindex == 0){
         if (smearedcorrelator_BSM == 1){
           a11=  +1.*g_smeared_scalar_field[2][0] + I*g_smeared_scalar_field[1][0];
           a12=  +1.*g_smeared_scalar_field[0][0] + I*g_smeared_scalar_field[3][0];

           a21=  +1.*g_smeared_scalar_field[0][0] - I*g_smeared_scalar_field[3][0];
           a22=  -1.*g_smeared_scalar_field[2][0] + I*g_smeared_scalar_field[1][0];
         }
         else{
           a11=  +1.*g_scalar_field[2][0] + I*g_scalar_field[1][0];
           a12=  +1.*g_scalar_field[0][0] + I*g_scalar_field[3][0];

           a21=  +1.*g_scalar_field[0][0] - I*g_scalar_field[3][0];
           a22=  -1.*g_scalar_field[2][0] + I*g_scalar_field[1][0];
         }
       }
       else if (tauindex == 1){
         if (smearedcorrelator_BSM == 1){
           a11=  -1.*g_smeared_scalar_field[1][0] + I*g_smeared_scalar_field[2][0];
           a12=  +1.*g_smeared_scalar_field[3][0] - I*g_smeared_scalar_field[0][0];

           a21=  +1.*g_smeared_scalar_field[3][0] + I*g_smeared_scalar_field[0][0];
           a22=  +1.*g_smeared_scalar_field[1][0] + I*g_smeared_scalar_field[2][0];
         }
         else{
           a11=  -1.*g_scalar_field[1][0] + I*g_scalar_field[2][0];
           a12=  +1.*g_scalar_field[3][0] - I*g_scalar_field[0][0];

           a21=  +1.*g_scalar_field[3][0] + I*g_scalar_field[0][0];
           a22=  +1.*g_scalar_field[1][0] + I*g_scalar_field[2][0];
         }
       }
       else if (tauindex == 2){
         if (smearedcorrelator_BSM == 1){
           a11=  +1.*g_smeared_scalar_field[0][0] + I*g_smeared_scalar_field[3][0];
           a12=  -1.*g_smeared_scalar_field[2][0] - I*g_smeared_scalar_field[1][0];

           a21=  -1.*g_smeared_scalar_field[2][0] + I*g_smeared_scalar_field[1][0];
           a22=  -1.*g_smeared_scalar_field[0][0] + I*g_smeared_scalar_field[3][0];
         }
         else{
           a11=  +1.*g_scalar_field[0][0] + I*g_scalar_field[3][0];
           a12=  -1.*g_scalar_field[2][0] - I*g_scalar_field[1][0];

           a21=  -1.*g_scalar_field[2][0] + I*g_scalar_field[1][0];
           a22=  -1.*g_scalar_field[0][0] + I*g_scalar_field[3][0];
         }
       }
     }
   }
   for (i=0; i<T_global; ++i){
     dest[2*i +0]= a11* source_copy[2*i + 0] + a12* source_copy[2*i + 1];
     dest[2*i +1]= a21* source_copy[2*i + 0] + a22* source_copy[2*i + 1];
   }
   free(source_copy);  
}
//dest used as a source, an output it is overwritten
void taui_scalarfield_flavoronly_s0s0( _Complex double *dest, int dagger ){
   _Complex double *source_copy;
   _Complex double a11=0.0, a12=0.0, a21=0.0, a22=0.0;
   int i;
 
   source_copy=(_Complex double *)malloc(sizeof(_Complex double)*2*T_global);
   for (i=0; i<2*T_global; ++i)
     source_copy[i]=dest[i];
   if (dagger == DAGGER){
     if (smearedcorrelator_BSM == 1){
       a11=  +1.*g_smeared_scalar_field[0][0] - I*g_smeared_scalar_field[3][0];
       a12=  -1.*g_smeared_scalar_field[2][0] - I*g_smeared_scalar_field[1][0];

       a21=  +1.*g_smeared_scalar_field[2][0] - I*g_smeared_scalar_field[1][0];
       a22=  +1.*g_smeared_scalar_field[0][0] + I*g_smeared_scalar_field[3][0];
     }
     else{
       a11=  +1.*g_scalar_field[0][0] - I*g_scalar_field[3][0];
       a12=  -1.*g_scalar_field[2][0] - I*g_scalar_field[1][0];

       a21=  +1.*g_scalar_field[2][0] - I*g_scalar_field[1][0];
       a22=  +1.*g_scalar_field[0][0] + I*g_scalar_field[3][0];     
     }
   }
   else if (dagger==NO_DAGG){
     if (smearedcorrelator_BSM == 1){
       a11=  +1.*g_smeared_scalar_field[0][0] + I*g_smeared_scalar_field[3][0];
       a12=  +1.*g_smeared_scalar_field[2][0] + I*g_smeared_scalar_field[1][0];

       a21=  -1.*g_smeared_scalar_field[2][0] + I*g_smeared_scalar_field[1][0];
       a22=  +1.*g_smeared_scalar_field[0][0] - I*g_smeared_scalar_field[3][0];
     }
     else{
       a11=  +1.*g_scalar_field[0][0] + I*g_scalar_field[3][0];
       a12=  +1.*g_scalar_field[2][0] + I*g_scalar_field[1][0];

       a21=  -1.*g_scalar_field[2][0] + I*g_scalar_field[1][0];
       a22=  +1.*g_scalar_field[0][0] - I*g_scalar_field[3][0];
     }
   }
   else{
      a11=0.;
      a12=0.;
      a21=0.;
      a22=0.;
      if (g_cart_id == 0){printf("Wrong Dagger index\n"); exit(1);}
   }
   for (i=0; i<T_global; ++i){
     dest[2*i +0]= a11* source_copy[2*i + 0] + a12* source_copy[2*i + 1];
     dest[2*i +1]= a21* source_copy[2*i + 0] + a22* source_copy[2*i + 1];
   }
   free(source_copy);
}
void mult_phi_flavoronly( _Complex double *dest, int dagg){
   _Complex double *source_copy;
   _Complex double a11=0.0, a12=0.0, a21=0.0, a22=0.0;
   int i;
   source_copy=(_Complex double *)malloc(sizeof(_Complex double)*2*T_global);
   if (source_copy == NULL){
     printf("Error in mem allcoation in phi0 tau3 commutator\n");
     exit(1);
   }
   for (i=0; i<2*T_global; ++i)
     source_copy[i]=dest[i];
   if ( dagg == NO_DAGG ){
     if ( smearedcorrelator_BSM == 1 ){
       a11=+1.*g_smeared_scalar_field[0][0]+1.*I*g_smeared_scalar_field[3][0];
       a12=+1.*g_smeared_scalar_field[2][0]+1.*I*g_smeared_scalar_field[1][0];
       a21=-1.*g_smeared_scalar_field[2][0]+1.*I*g_smeared_scalar_field[1][0];
       a22=+1.*g_smeared_scalar_field[0][0]-1.*I*g_smeared_scalar_field[3][0];
     }
     else{
       a11=+1.*g_scalar_field[0][0]+1.*I*g_scalar_field[3][0];
       a12=+1.*g_scalar_field[2][0]+1.*I*g_scalar_field[1][0];
       a21=-1.*g_scalar_field[2][0]+1.*I*g_scalar_field[1][0];
       a22=+1.*g_scalar_field[0][0]-1.*I*g_scalar_field[3][0];
     }
   }
   else if (dagg == DAGGER){
     if ( smearedcorrelator_BSM == 1 ){
       a11=+1.*g_smeared_scalar_field[0][0]-1.*I*g_smeared_scalar_field[3][0];
       a12=-1.*g_smeared_scalar_field[2][0]-1.*I*g_smeared_scalar_field[1][0];
       a21=+1.*g_smeared_scalar_field[2][0]-1.*I*g_smeared_scalar_field[1][0];
       a22=+1.*g_smeared_scalar_field[0][0]+1.*I*g_smeared_scalar_field[3][0];
     }
     else{
       a11=+1.*g_scalar_field[0][0]-1.*I*g_scalar_field[3][0];
       a12=-1.*g_scalar_field[2][0]-1.*I*g_scalar_field[1][0];
       a21=+1.*g_scalar_field[2][0]-1.*I*g_scalar_field[1][0];
       a22=+1.*g_scalar_field[0][0]+1.*I*g_scalar_field[3][0];
     }
   }
   else{
     if (g_cart_id == 0) {printf("Error in giving the index in mult_phi_flavoronly\n");
                          exit(1);
                         }
   }
   for (i=0; i<T_global; ++i){
     dest[2*i +0]= a11* source_copy[2*i + 0] + a12* source_copy[2*i + 1];
     dest[2*i +1]= a21* source_copy[2*i + 0] + a22* source_copy[2*i + 1];
   }
   free(source_copy);
}
void mult_taui_flavoronly( _Complex double *dest, int tauindex){
   _Complex double *source_copy;
   _Complex double a11=0.0, a12=0.0, a21=0.0, a22=0.0;
   int i;
   source_copy=(_Complex double *)malloc(sizeof(_Complex double)*2*T_global);
   if (source_copy == NULL){
     printf("Error in mem allcoation in phi0 tau3 commutator\n");
     exit(1);
   }
   for (i=0; i<2*T_global; ++i)
     source_copy[i]=dest[i];
   if ( tauindex == 2 ){
     a11=+1.;
     a12= 0.;
     a21= 0.;
     a22=-1.;
   }
   else if ( tauindex == 1 ){
     a11=    0.;
     a12= -1.*I;
     a21=     I;
     a22=     0;
   }
   else if ( tauindex == 0 ){
     a11= 0.;
     a12=+1.;
     a21=+1.;
     a22= 0.;
   }
   else{
     if (g_cart_id == 0) {printf("Error in giving the tauindex in mult_taui_flavoronly\n");
                          exit(1);
                         }
   }
   for (i=0; i<T_global; ++i){
     dest[2*i +0]= a11* source_copy[2*i + 0] + a12* source_copy[2*i + 1];
     dest[2*i +1]= a21* source_copy[2*i + 0] + a22* source_copy[2*i + 1];
   }
   free(source_copy);
}
void mult_phi( bispinor *dest, bispinor *source, int ix, int dagg){
   bispinor tmp;
   _spinor_assign(tmp.sp_up, source->sp_up);
   _spinor_assign(tmp.sp_dn, source->sp_dn);
   _Complex double a11=0., a12=0., a21=0., a22=0.;

   if ( dagg == NO_DAGG ){
     if ( smearedcorrelator_BSM == 1 ){
       a11=+1.*g_smeared_scalar_field[0][ix]+1.*I*g_smeared_scalar_field[3][ix];
       a12=+1.*g_smeared_scalar_field[2][ix]+1.*I*g_smeared_scalar_field[1][ix];
       a21=-1.*g_smeared_scalar_field[2][ix]+1.*I*g_smeared_scalar_field[1][ix];
       a22=+1.*g_smeared_scalar_field[0][ix]-1.*I*g_smeared_scalar_field[3][ix];
     }
     else{
       a11=+1.*g_scalar_field[0][ix]+1.*I*g_scalar_field[3][ix];
       a12=+1.*g_scalar_field[2][ix]+1.*I*g_scalar_field[1][ix];
       a21=-1.*g_scalar_field[2][ix]+1.*I*g_scalar_field[1][ix];
       a22=+1.*g_scalar_field[0][ix]-1.*I*g_scalar_field[3][ix];
     }
   }
   else if (dagg == DAGGER){
     if ( smearedcorrelator_BSM == 1 ){
       a11=+1.*g_smeared_scalar_field[0][ix]-1.*I*g_smeared_scalar_field[3][ix];
       a12=-1.*g_smeared_scalar_field[2][ix]-1.*I*g_smeared_scalar_field[1][ix];
       a21=+1.*g_smeared_scalar_field[2][ix]-1.*I*g_smeared_scalar_field[1][ix];
       a22=+1.*g_smeared_scalar_field[0][ix]+1.*I*g_smeared_scalar_field[3][ix];
     }
     else{
       a11=+1.*g_scalar_field[0][ix]-1.*I*g_scalar_field[3][ix];
       a12=-1.*g_scalar_field[2][ix]-1.*I*g_scalar_field[1][ix];
       a21=+1.*g_scalar_field[2][ix]-1.*I*g_scalar_field[1][ix];
       a22=+1.*g_scalar_field[0][ix]+1.*I*g_scalar_field[3][ix];
     }
   }
   else{
     if (g_cart_id == 0) {printf("Error in giving the index in mult_phi_flavoronly\n");
                          exit(1);
                         }
   }
   dest->sp_up.s0.c0 = a11 * tmp.sp_up.s0.c0 + a12 * tmp.sp_dn.s0.c0;
   dest->sp_up.s0.c1 = a11 * tmp.sp_up.s0.c1 + a12 * tmp.sp_dn.s0.c1;
   dest->sp_up.s0.c2 = a11 * tmp.sp_up.s0.c2 + a12 * tmp.sp_dn.s0.c2;

   dest->sp_up.s1.c0 = a11 * tmp.sp_up.s1.c0 + a12 * tmp.sp_dn.s1.c0;
   dest->sp_up.s1.c1 = a11 * tmp.sp_up.s1.c1 + a12 * tmp.sp_dn.s1.c1;
   dest->sp_up.s1.c2 = a11 * tmp.sp_up.s1.c2 + a12 * tmp.sp_dn.s1.c2;

   dest->sp_up.s2.c0 = a11 * tmp.sp_up.s2.c0 + a12 * tmp.sp_dn.s2.c0;
   dest->sp_up.s2.c1 = a11 * tmp.sp_up.s2.c1 + a12 * tmp.sp_dn.s2.c1;
   dest->sp_up.s2.c2 = a11 * tmp.sp_up.s2.c2 + a12 * tmp.sp_dn.s2.c2;

   dest->sp_up.s3.c0 = a11 * tmp.sp_up.s3.c0 + a12 * tmp.sp_dn.s3.c0;
   dest->sp_up.s3.c1 = a11 * tmp.sp_up.s3.c1 + a12 * tmp.sp_dn.s3.c1;
   dest->sp_up.s3.c2 = a11 * tmp.sp_up.s3.c2 + a12 * tmp.sp_dn.s3.c2;

   dest->sp_dn.s0.c0 = a21 * tmp.sp_up.s0.c0 + a22 * tmp.sp_dn.s0.c0;
   dest->sp_dn.s0.c1 = a21 * tmp.sp_up.s0.c1 + a22 * tmp.sp_dn.s0.c1;
   dest->sp_dn.s0.c2 = a21 * tmp.sp_up.s0.c2 + a22 * tmp.sp_dn.s0.c2;

   dest->sp_dn.s1.c0 = a21 * tmp.sp_up.s1.c0 + a22 * tmp.sp_dn.s1.c0;
   dest->sp_dn.s1.c1 = a21 * tmp.sp_up.s1.c1 + a22 * tmp.sp_dn.s1.c1;
   dest->sp_dn.s1.c2 = a21 * tmp.sp_up.s1.c2 + a22 * tmp.sp_dn.s1.c2;

   dest->sp_dn.s2.c0 = a21 * tmp.sp_up.s2.c0 + a22 * tmp.sp_dn.s2.c0;
   dest->sp_dn.s2.c1 = a21 * tmp.sp_up.s2.c1 + a22 * tmp.sp_dn.s2.c1;
   dest->sp_dn.s2.c2 = a21 * tmp.sp_up.s2.c2 + a22 * tmp.sp_dn.s2.c2;

   dest->sp_dn.s3.c0 = a21 * tmp.sp_up.s3.c0 + a22 * tmp.sp_dn.s3.c0;
   dest->sp_dn.s3.c1 = a21 * tmp.sp_up.s3.c1 + a22 * tmp.sp_dn.s3.c1;
   dest->sp_dn.s3.c2 = a21 * tmp.sp_up.s3.c2 + a22 * tmp.sp_dn.s3.c2;

}
void taui_spinor( bispinor *dest, bispinor *source, int tauindex ){

   su3_vector tmp2;
   bispinor tmp;
   _spinor_assign(tmp.sp_up, source->sp_up);
   _spinor_assign(tmp.sp_dn, source->sp_dn);


   if (tauindex == 0 ){
    _vector_assign(tmp2        , tmp.sp_up.s0);
    _vector_assign(tmp.sp_up.s0, tmp.sp_dn.s0);
    _vector_assign(tmp.sp_dn.s0, tmp2);

    _vector_assign(tmp2        , tmp.sp_up.s1);
    _vector_assign(tmp.sp_up.s1, tmp.sp_dn.s1);
    _vector_assign(tmp.sp_dn.s1, tmp2);

    _vector_assign(tmp2        , tmp.sp_up.s2);
    _vector_assign(tmp.sp_up.s2, tmp.sp_dn.s2);
    _vector_assign(tmp.sp_dn.s2, tmp2);

    _vector_assign(tmp2        , tmp.sp_up.s3);
    _vector_assign(tmp.sp_up.s3, tmp.sp_dn.s3);
    _vector_assign(tmp.sp_dn.s3, tmp2);

 
    _spinor_assign(dest->sp_up, tmp.sp_up);
    _spinor_assign(dest->sp_dn, tmp.sp_dn);
   }
   else if (tauindex == 1 ){
    _vector_assign(tmp2             ,tmp.sp_up.s0);
    _vector_i_mul( tmp.sp_up.s0, -1 ,tmp.sp_dn.s0);
    _vector_i_mul( tmp.sp_dn.s0, +1 ,tmp2);

    _vector_assign(tmp2             ,tmp.sp_up.s1);
    _vector_i_mul( tmp.sp_up.s1, -1 ,tmp.sp_dn.s1);
    _vector_i_mul( tmp.sp_dn.s1, +1, tmp2);   

    _vector_assign(tmp2             ,tmp.sp_up.s2);
    _vector_i_mul( tmp.sp_up.s2, -1 ,tmp.sp_dn.s2);
    _vector_i_mul( tmp.sp_dn.s2, +1 ,tmp2);

    _vector_assign(tmp2             ,tmp.sp_up.s3);
    _vector_i_mul( tmp.sp_up.s3, -1 ,tmp.sp_dn.s3);
    _vector_i_mul( tmp.sp_dn.s3, +1, tmp2);
     
    _spinor_assign(dest->sp_up, tmp.sp_up);
    _spinor_assign(dest->sp_dn, tmp.sp_dn);
   }
   else if (tauindex == 2 ){
    _vector_mul(tmp.sp_dn.s0, -1, tmp.sp_dn.s0);
    _vector_mul(tmp.sp_dn.s1, -1, tmp.sp_dn.s1);

    _vector_mul(tmp.sp_dn.s2, -1, tmp.sp_dn.s2);
    _vector_mul(tmp.sp_dn.s3, -1, tmp.sp_dn.s3);

     
    _spinor_assign(dest->sp_up, tmp.sp_up);
    _spinor_assign(dest->sp_dn, tmp.sp_dn);
   }
}

void phi0_taui_commutator( _Complex double *dest,int tauindex ){

   _Complex double *source_copy;
   _Complex double a11=0.0, a12=0.0, a21=0.0, a22=0.0;
   int i;

   source_copy=(_Complex double *)malloc(sizeof(_Complex double)*2*T_global);
   if (source_copy == NULL){ 
     printf("Error in mem allcoation in phi0 tau3 commutator\n");
     exit(1);
   }
   for (i=0; i<2*T_global; ++i)
     source_copy[i]=dest[i];
   if (tauindex == 2){
     if (smearedcorrelator_BSM == 1){
       a11=0.;
       a12=-2.*g_smeared_scalar_field[2][0]-2.*I*g_smeared_scalar_field[1][0];
       a21=-2.*g_smeared_scalar_field[2][0]+2.*I*g_smeared_scalar_field[1][0];
       a22=0.;
     }
     else{
       a11=0.;
       a12=-2.*g_scalar_field[2][0]-2.*I*g_scalar_field[1][0];
       a21=-2.*g_scalar_field[2][0]+2.*I*g_scalar_field[1][0];
       a22=0.;
     }
   }
   if (tauindex == 1){
     if (smearedcorrelator_BSM == 1){
       a11=-2.*g_smeared_scalar_field[1][0];
       a12=+2.*g_smeared_scalar_field[3][0];
       a21=+2.*g_smeared_scalar_field[3][0];
       a22=+2.*g_smeared_scalar_field[1][0];
     }
     else{
       a11=-2.*g_scalar_field[1][0];
       a12=+2.*g_scalar_field[3][0];
       a21=+2.*g_scalar_field[3][0];
       a22=+2.*g_scalar_field[1][0];
     }
   }
   if (tauindex == 0){
     if (smearedcorrelator_BSM == 1){
       a11=+2.*  g_smeared_scalar_field[2][0];
       a12=+2.*I*g_smeared_scalar_field[3][0];
       a21=-2.*I*g_smeared_scalar_field[3][0];
       a22=-2.*  g_smeared_scalar_field[2][0];
     }
     else{
       a11=+2.*  g_scalar_field[2][0];
       a12=+2.*I*g_scalar_field[3][0];
       a21=-2.*I*g_scalar_field[3][0];
       a22=-2.*  g_scalar_field[2][0];
     }
   }


   for (i=0; i<T_global; ++i){
     dest[2*i +0]= a11* source_copy[2*i + 0] + a12* source_copy[2*i + 1];
     dest[2*i +1]= a21* source_copy[2*i + 0] + a22* source_copy[2*i + 1];
   }
   free(source_copy);
}
//This routine computes the commutator between Phi(x)and tau^i
//times a bispinor vector
//Here Phi(x) is represented by a matrix
//
//(+phi_0+i*phi_3   phi_2+iphi_1)
//(-phi_2+i*phi_1   phi_0-iphi_3)
//
void phix_taui_commutator_bispinor( bispinor *dest,int tauindex, int gamma5, int ix ){

   bispinor source_copy;
   bispinor tmpbi2;
   _Complex double a11=0.0, a12=0.0, a21=0.0, a22=0.0;

   _spinor_assign(source_copy.sp_up, dest->sp_up);
   _spinor_assign(source_copy.sp_dn, dest->sp_dn);

   if (tauindex == 2){
     if (smearedcorrelator_BSM == 1){
       a11=0.;
       a12=-2.*g_smeared_scalar_field[2][ix]-2.*I*g_smeared_scalar_field[1][ix];
       a21=-2.*g_smeared_scalar_field[2][ix]+2.*I*g_smeared_scalar_field[1][ix];
       a22=0.;
     }
     else{
       a11=0.;
       a12=-2.*g_scalar_field[2][ix]-2.*I*g_scalar_field[1][ix];
       a21=-2.*g_scalar_field[2][ix]+2.*I*g_scalar_field[1][ix];
       a22=0.;
     }
   }
   else if (tauindex == 1){
     if (smearedcorrelator_BSM == 1){
       a11=-2.*g_smeared_scalar_field[1][ix];
       a12=+2.*g_smeared_scalar_field[3][ix];
       a21=+2.*g_smeared_scalar_field[3][ix];
       a22=+2.*g_smeared_scalar_field[1][ix];
     }
     else{
       a11=-2.*g_scalar_field[1][ix];
       a12=+2.*g_scalar_field[3][ix];
       a21=+2.*g_scalar_field[3][ix];
       a22=+2.*g_scalar_field[1][ix];
     }
   }
   else if (tauindex == 0){
     if (smearedcorrelator_BSM == 1){
       a11=+2.*  g_smeared_scalar_field[2][ix];
       a12=+2.*I*g_smeared_scalar_field[3][ix];
       a21=-2.*I*g_smeared_scalar_field[3][ix];
       a22=-2.*  g_smeared_scalar_field[2][ix];
     }
     else{
       a11=+2.*  g_scalar_field[2][ix];
       a12=+2.*I*g_scalar_field[3][ix];
       a21=-2.*I*g_scalar_field[3][ix];
       a22=-2.*  g_scalar_field[2][ix];
     }
   }
   else {
     if (g_cart_id == 0){ 
       printf("Wrong Pauli matrix index\n");
       exit(1);
     }    
   }
   _spinor_null(tmpbi2.sp_up);
   _spinor_null(tmpbi2.sp_dn);

   if ( gamma5 == GAMMA_UP){
     _vector_mul_complex(    tmpbi2.sp_up.s0, a11, source_copy.sp_up.s0);
     _vector_add_mul_complex(tmpbi2.sp_up.s0, a12, source_copy.sp_dn.s0);

     _vector_mul_complex    (tmpbi2.sp_dn.s0, a21, source_copy.sp_up.s0);
     _vector_add_mul_complex(tmpbi2.sp_dn.s0, a22, source_copy.sp_dn.s0);

     _vector_mul_complex(    tmpbi2.sp_up.s1, a11, source_copy.sp_up.s1);
     _vector_add_mul_complex(tmpbi2.sp_up.s1, a12, source_copy.sp_dn.s1);

     _vector_mul_complex    (tmpbi2.sp_dn.s1, a21, source_copy.sp_up.s1);
     _vector_add_mul_complex(tmpbi2.sp_dn.s1, a22, source_copy.sp_dn.s1);
   }
   else if  ( gamma5 == GAMMA_DN ){
     _vector_mul_complex(    tmpbi2.sp_up.s2, a11, source_copy.sp_up.s2);
     _vector_add_mul_complex(tmpbi2.sp_up.s2, a12, source_copy.sp_dn.s2);

     _vector_mul_complex    (tmpbi2.sp_dn.s2, a21, source_copy.sp_up.s2);
     _vector_add_mul_complex(tmpbi2.sp_dn.s2, a22, source_copy.sp_dn.s2);

     _vector_mul_complex(    tmpbi2.sp_up.s3, a11, source_copy.sp_up.s3);
     _vector_add_mul_complex(tmpbi2.sp_up.s3, a12, source_copy.sp_dn.s3);

     _vector_mul_complex    (tmpbi2.sp_dn.s3, a21, source_copy.sp_up.s3);
     _vector_add_mul_complex(tmpbi2.sp_dn.s3, a22, source_copy.sp_dn.s3);
   }
   else if ( gamma5 == NO_GAMMA ){
     _spinor_mul_complex    (tmpbi2.sp_up,    a11, source_copy.sp_up);
     _spinor_add_mul_complex(tmpbi2.sp_up,    a12, source_copy.sp_dn);

     _spinor_mul_complex    (tmpbi2.sp_dn,    a21, source_copy.sp_up);
     _spinor_add_mul_complex(tmpbi2.sp_dn,    a22, source_copy.sp_dn);
   }

   _spinor_assign(dest->sp_up, tmpbi2.sp_up);
   _spinor_assign(dest->sp_dn, tmpbi2.sp_dn);

}

void phix_taui_anti_commutator_bispinor( bispinor *dest,int tauindex, int gamma5, int dagger,int ix ){

   bispinor source_copy;
   bispinor tmpbi2;
   _Complex double a11=0.0, a12=0.0, a21=0.0, a22=0.0;

   _spinor_assign(source_copy.sp_up, dest->sp_up);
   _spinor_assign(source_copy.sp_dn, dest->sp_dn);

   if (dagger == NO_DAGG){
     if (tauindex == 2){
       if (smearedcorrelator_BSM == 1){
         a11=+2.*(g_smeared_scalar_field[0][ix]+I*g_smeared_scalar_field[3][ix]);
         a12=0.;
         a21=0.;
         a22=-2.*(g_smeared_scalar_field[0][ix]-I*g_smeared_scalar_field[3][ix]);
       }
       else{
         a11=2.*(g_scalar_field[0][ix]+I*g_scalar_field[3][ix]);
         a12=0.;
         a21=0.;
         a22=-2*(g_scalar_field[0][ix]-I*g_scalar_field[3][ix]);
       }
     }
     else if (tauindex == 1){
       if (smearedcorrelator_BSM == 1){
         a11=+2.*I*g_smeared_scalar_field[2][ix];
         a12=-2.*I*g_smeared_scalar_field[0][ix];
         a21=+2.*I*g_smeared_scalar_field[0][ix];
         a22=+2.*I*g_smeared_scalar_field[2][ix];
       }
       else{
         a11=+2.*I*g_scalar_field[2][ix];
         a12=-2.*I*g_scalar_field[0][ix];
         a21=+2.*I*g_scalar_field[0][ix];
         a22=+2.*I*g_scalar_field[2][ix];
       }
     }
     else if (tauindex == 0){
       if (smearedcorrelator_BSM == 1){
         a11=+2.*I*g_smeared_scalar_field[1][ix];
         a12=+2.  *g_smeared_scalar_field[0][ix];
         a21=+2.  *g_smeared_scalar_field[0][ix];
         a22=+2.*I*g_smeared_scalar_field[1][ix];
       }
       else{
         a11=+2.*I*g_scalar_field[1][ix];
         a12=+2.  *g_scalar_field[0][ix];
         a21=+2.  *g_scalar_field[0][ix];
         a22=+2.*I*g_scalar_field[1][ix];
       }
     }
     else {
       if (g_cart_id == 0){
         printf("Wrong Pauli matrix index\n");
         exit(1);
       }
     }
   }
   else if (dagger==DAGGER){
     if (tauindex == 2){
       if (smearedcorrelator_BSM == 1){
         a11=+2.*(g_smeared_scalar_field[0][ix]-I*g_smeared_scalar_field[3][ix]);
         a12=0.;
         a21=0.;
         a22=-2.*(g_smeared_scalar_field[0][ix]+I*g_smeared_scalar_field[3][ix]);
       }
       else{
         a11=2.*(g_scalar_field[0][ix]-I*g_scalar_field[3][ix]);
         a12=0.;
         a21=0.;
         a22=-2*(g_scalar_field[0][ix]+I*g_scalar_field[3][ix]);
       }
     }
     else if (tauindex == 1){
       if (smearedcorrelator_BSM == 1){
         a11=-2.*I*g_smeared_scalar_field[2][ix];
         a12=-2.*I*g_smeared_scalar_field[0][ix];
         a21=+2.*I*g_smeared_scalar_field[0][ix];
         a22=-2.*I*g_smeared_scalar_field[2][ix];
       }
       else{
         a11=-2.*I*g_scalar_field[2][ix];
         a12=-2.*I*g_scalar_field[0][ix];
         a21=+2.*I*g_scalar_field[0][ix];
         a22=-2.*I*g_scalar_field[2][ix];
       }
     }
     else if (tauindex == 0){
       if (smearedcorrelator_BSM == 1){
         a11=-2.*I*g_smeared_scalar_field[1][ix];
         a12=+2.  *g_smeared_scalar_field[0][ix];
         a21=+2.  *g_smeared_scalar_field[0][ix];
         a22=-2.*I*g_smeared_scalar_field[1][ix];
       }
       else{
         a11=-2.*I*g_scalar_field[1][ix];
         a12=+2.  *g_scalar_field[0][ix];
         a21=+2.  *g_scalar_field[0][ix];
         a22=-2.*I*g_scalar_field[1][ix];
       }
     }
     else {
       if (g_cart_id == 0){
         printf("Wrong Pauli matrix index\n");
         exit(1);
       }
     }  
   }
   else{
     if (g_cart_id == 0){
       printf("Anticommutator phi tau has to be either dagger or not\n");
       exit(1);
     }
   }

   _spinor_null(tmpbi2.sp_up);
   _spinor_null(tmpbi2.sp_dn);

   if ( gamma5 == GAMMA_UP){
     _vector_mul_complex(    tmpbi2.sp_up.s0, a11, source_copy.sp_up.s0);
     _vector_add_mul_complex(tmpbi2.sp_up.s0, a12, source_copy.sp_dn.s0);

     _vector_mul_complex    (tmpbi2.sp_dn.s0, a21, source_copy.sp_up.s0);
     _vector_add_mul_complex(tmpbi2.sp_dn.s0, a22, source_copy.sp_dn.s0);

     _vector_mul_complex(    tmpbi2.sp_up.s1, a11, source_copy.sp_up.s1);
     _vector_add_mul_complex(tmpbi2.sp_up.s1, a12, source_copy.sp_dn.s1);

     _vector_mul_complex    (tmpbi2.sp_dn.s1, a21, source_copy.sp_up.s1);
     _vector_add_mul_complex(tmpbi2.sp_dn.s1, a22, source_copy.sp_dn.s1);
   }
   else if  ( gamma5 == GAMMA_DN ){
     _vector_mul_complex(    tmpbi2.sp_up.s2, a11, source_copy.sp_up.s2);
     _vector_add_mul_complex(tmpbi2.sp_up.s2, a12, source_copy.sp_dn.s2);

     _vector_mul_complex    (tmpbi2.sp_dn.s2, a21, source_copy.sp_up.s2);
     _vector_add_mul_complex(tmpbi2.sp_dn.s2, a22, source_copy.sp_dn.s2);

     _vector_mul_complex(    tmpbi2.sp_up.s3, a11, source_copy.sp_up.s3);
     _vector_add_mul_complex(tmpbi2.sp_up.s3, a12, source_copy.sp_dn.s3);

     _vector_mul_complex    (tmpbi2.sp_dn.s3, a21, source_copy.sp_up.s3);
     _vector_add_mul_complex(tmpbi2.sp_dn.s3, a22, source_copy.sp_dn.s3);
   }
   else if ( gamma5 == NO_GAMMA ){
     _spinor_mul_complex    (tmpbi2.sp_up,    a11, source_copy.sp_up);
     _spinor_add_mul_complex(tmpbi2.sp_up,    a12, source_copy.sp_dn);

     _spinor_mul_complex    (tmpbi2.sp_dn,    a21, source_copy.sp_up);
     _spinor_add_mul_complex(tmpbi2.sp_dn,    a22, source_copy.sp_dn);
   }

   _spinor_assign(dest->sp_up, tmpbi2.sp_up);
   _spinor_assign(dest->sp_dn, tmpbi2.sp_dn);

}


void phi0_taui_anticommutator( _Complex double *dest, int tauindex, int dagger ){

   _Complex double *source_copy;
   _Complex double a11=0.0, a12=0.0, a21=0.0, a22=0.0;
   int i;

   source_copy=(_Complex double *)malloc(sizeof(_Complex double)*2*T_global);
   if (source_copy == NULL){
     printf("Error in mem allcoation in phi0 tau3 commutator\n");
     exit(1);
   }
   for (i=0; i<2*T_global; ++i)
     source_copy[i]=dest[i];
   if ( tauindex == 2){
     if (dagger == NO_DAGG){
       if (smearedcorrelator_BSM == 1){
         a11=2.*(+1.*g_smeared_scalar_field[0][0]+1.*I*g_smeared_scalar_field[3][0]);
         a12=0.;
         a21=0.;
         a22=2.*(-1.*g_smeared_scalar_field[0][0]+1.*I*g_smeared_scalar_field[3][0]);
       }
       else{
         a11=2.*(+1.*g_scalar_field[0][0]+1.*I*g_scalar_field[3][0]);
         a12=0.;
         a21=0.;
         a22=2.*(-1.*g_scalar_field[0][0]+1.*I*g_scalar_field[3][0]);
       }
     }
     if (dagger == DAGGER){
       if (smearedcorrelator_BSM == 1){
         a11=2.*(+1.*g_smeared_scalar_field[0][0]-1.*I*g_smeared_scalar_field[3][0]);
         a12=0.;
         a21=0.;
         a22=2.*(-1.*g_smeared_scalar_field[0][0]-1.*I*g_smeared_scalar_field[3][0]);
       }
       else{
         a11=2.*(+1.*g_scalar_field[0][0]-1.*I*g_scalar_field[3][0]);
         a12=0.;
         a21=0.;
         a22=2.*(-1.*g_scalar_field[0][0]-1.*I*g_scalar_field[3][0]);
       } 

     }
   }
   if ( tauindex == 1){
     if (dagger == NO_DAGG){
       if (smearedcorrelator_BSM == 1){
         a11=+2.*I*g_smeared_scalar_field[2][0];
         a12=-2.*I*g_smeared_scalar_field[0][0];
         a21=+2.*I*g_smeared_scalar_field[0][0];
         a22=+2.*I*g_smeared_scalar_field[2][0];
       }
       else{
         a11=+2.*I*g_scalar_field[2][0];
         a12=-2.*I*g_scalar_field[0][0];
         a21=+2.*I*g_scalar_field[0][0];
         a22=+2.*I*g_scalar_field[2][0];
       }
     }
     if (dagger == DAGGER){
       if (smearedcorrelator_BSM == 1){
         a11=-2.*I*g_smeared_scalar_field[2][0];
         a12=-2.*I*g_smeared_scalar_field[0][0];
         a21=+2.*I*g_smeared_scalar_field[0][0];
         a22=-2.*I*g_smeared_scalar_field[2][0];
       }
       else{
         a11=-2.*I*g_scalar_field[2][0];
         a12=-2.*I*g_scalar_field[0][0];
         a21=+2.*I*g_scalar_field[0][0];
         a22=-2.*I*g_scalar_field[2][0];
       }

     } 
   }

   if ( tauindex == 0){
     if (dagger == NO_DAGG){
       if (smearedcorrelator_BSM == 1){
         a11=2.*I*g_smeared_scalar_field[1][0];
         a12=2.*  g_smeared_scalar_field[0][0];
         a21=2.*  g_smeared_scalar_field[0][0];
         a22=2.*I*g_smeared_scalar_field[1][0];
       }
       else{
         a11=2.*I*g_scalar_field[1][0];
         a12=2.*  g_scalar_field[0][0];
         a21=2.*  g_scalar_field[0][0];
         a22=2.*I*g_scalar_field[1][0];
       }
     }
     if (dagger == DAGGER){
       if (smearedcorrelator_BSM == 1){
         a11=-2.*I*g_smeared_scalar_field[1][0];
         a12=2.*  g_smeared_scalar_field[0][0];
         a21=2.*  g_smeared_scalar_field[0][0];
         a22=-2.*I*g_smeared_scalar_field[1][0];
       }
       else{
         a11=-2.*I*g_scalar_field[1][0];
         a12=2.*  g_scalar_field[0][0];
         a21=2.*  g_scalar_field[0][0];
         a22=-2.*I*g_scalar_field[1][0];
       }
     }
   }

//   printf("a11=%e %e\n", creal(a11), cimag(a11));
//   printf("a12=%e %e\n", creal(a12), cimag(a12));
//   printf("a21=%e %e\n", creal(a21), cimag(a21));
//   printf("a22=%e %e\n", creal(a22), cimag(a22));


   for (i=0; i<T_global; ++i){
     dest[2*i +0]= a11* source_copy[2*i + 0] + a12* source_copy[2*i + 1];
     dest[2*i +1]= a21* source_copy[2*i + 0] + a22* source_copy[2*i + 1];
   }
   free(source_copy);
}


void taui_scalarfield_spinor_s0s0( bispinor *dest, bispinor *source, int gamma5, int idx, int direction, int dagger){

  bispinor tmp;
  bispinor tmpbi2;
  _Complex double a11=0., a12=0., a21=0., a22=0.;

  int scalarcoord;

  _spinor_assign(tmp.sp_up, source->sp_up);
  _spinor_assign(tmp.sp_dn, source->sp_dn);

 if (direction == NODIR)
   scalarcoord=idx;
 else if (direction<4){
   scalarcoord= g_iup[idx][direction];
 }
 else if (direction<8){
   scalarcoord= g_idn[idx][7-direction];
 }
 else{
   scalarcoord=0;
   if (g_cart_id == 0) {printf("Wrong direction in tau scalar field spinor\n"); exit(1);}
 }
 if (dagger == DAGGER){
   if (smearedcorrelator_BSM == 1){
     a11=  +1.*g_smeared_scalar_field[0][scalarcoord] - I*g_smeared_scalar_field[3][scalarcoord];
     a12=  -1.*g_smeared_scalar_field[2][scalarcoord] - I*g_smeared_scalar_field[1][scalarcoord];

     a21=  +1.*g_smeared_scalar_field[2][scalarcoord] - I*g_smeared_scalar_field[1][scalarcoord];
     a22=  +1.*g_smeared_scalar_field[0][scalarcoord] + I*g_smeared_scalar_field[3][scalarcoord];

   }
   else{
     a11=  +1.*g_scalar_field[0][scalarcoord] - I*g_scalar_field[3][scalarcoord];
     a12=  -1.*g_scalar_field[2][scalarcoord] - I*g_scalar_field[1][scalarcoord];

     a21=  +1.*g_scalar_field[2][scalarcoord] - I*g_scalar_field[1][scalarcoord];
     a22=  +1.*g_scalar_field[0][scalarcoord] + I*g_scalar_field[3][scalarcoord];
   }
 }
 else if (dagger == NO_DAGG){
   if (smearedcorrelator_BSM == 1){
     a11=  +1.*g_smeared_scalar_field[0][scalarcoord] + I*g_smeared_scalar_field[3][scalarcoord];
     a12=  +1.*g_smeared_scalar_field[2][scalarcoord] + I*g_smeared_scalar_field[1][scalarcoord];

     a21=  -1.*g_smeared_scalar_field[2][scalarcoord] + I*g_smeared_scalar_field[1][scalarcoord];
     a22=  +1.*g_smeared_scalar_field[0][scalarcoord] - I*g_smeared_scalar_field[3][scalarcoord];

   }
   else{
     a11=  +1.*g_scalar_field[0][scalarcoord] + I*g_scalar_field[3][scalarcoord];
     a12=  +1.*g_scalar_field[2][scalarcoord] + I*g_scalar_field[1][scalarcoord];

     a21=  -1.*g_scalar_field[2][scalarcoord] + I*g_scalar_field[1][scalarcoord];
     a22=  +1.*g_scalar_field[0][scalarcoord] - I*g_scalar_field[3][scalarcoord];
   }
 }
 else {
   fprintf(stdout, "The sixth argument must be either DAGGER or NO_DAGG\n");
 }
 _spinor_null(tmpbi2.sp_up);
 _spinor_null(tmpbi2.sp_dn);

 if ( gamma5 == GAMMA_UP){
  _vector_mul_complex(    tmpbi2.sp_up.s0, a11, tmp.sp_up.s0);
  _vector_add_mul_complex(tmpbi2.sp_up.s0, a12, tmp.sp_dn.s0);

  _vector_mul_complex    (tmpbi2.sp_dn.s0, a21, tmp.sp_up.s0);
  _vector_add_mul_complex(tmpbi2.sp_dn.s0, a22, tmp.sp_dn.s0);

  _vector_mul_complex(    tmpbi2.sp_up.s1, a11, tmp.sp_up.s1);
  _vector_add_mul_complex(tmpbi2.sp_up.s1, a12, tmp.sp_dn.s1);

  _vector_mul_complex    (tmpbi2.sp_dn.s1, a21, tmp.sp_up.s1);
  _vector_add_mul_complex(tmpbi2.sp_dn.s1, a22, tmp.sp_dn.s1);
 }
 else if  ( gamma5 == GAMMA_DN ){
  _vector_mul_complex(    tmpbi2.sp_up.s2, a11, tmp.sp_up.s2);
  _vector_add_mul_complex(tmpbi2.sp_up.s2, a12, tmp.sp_dn.s2);

  _vector_mul_complex    (tmpbi2.sp_dn.s2, a21, tmp.sp_up.s2);
  _vector_add_mul_complex(tmpbi2.sp_dn.s2, a22, tmp.sp_dn.s2);

  _vector_mul_complex(    tmpbi2.sp_up.s3, a11, tmp.sp_up.s3);
  _vector_add_mul_complex(tmpbi2.sp_up.s3, a12, tmp.sp_dn.s3);

  _vector_mul_complex    (tmpbi2.sp_dn.s3, a21, tmp.sp_up.s3);
  _vector_add_mul_complex(tmpbi2.sp_dn.s3, a22, tmp.sp_dn.s3);
 }
 else if ( gamma5 == NO_GAMMA ){
  _spinor_mul_complex    (tmpbi2.sp_up,    a11, tmp.sp_up);
  _spinor_add_mul_complex(tmpbi2.sp_up,    a12, tmp.sp_dn);

  _spinor_mul_complex    (tmpbi2.sp_dn,    a21, tmp.sp_up);
  _spinor_add_mul_complex(tmpbi2.sp_dn,    a22, tmp.sp_dn);
 }

 _spinor_assign(dest->sp_up, tmpbi2.sp_up);
 _spinor_assign(dest->sp_dn, tmpbi2.sp_dn);

}
void taui_scalarfield_spinor( bispinor *dest, bispinor *source, int gamma5, int tauindex, int idx, int direction, int dagger){
    
  bispinor tmp;
  bispinor tmpbi2;
  _Complex double a11=0.0, a12=0.0, a21=0.0, a22=0.0;

  int scalarcoord;

  _spinor_assign(tmp.sp_up, source->sp_up);
  _spinor_assign(tmp.sp_dn, source->sp_dn);

 if (direction == NODIR)
   scalarcoord=idx;
 else if (direction == TUP ){
   scalarcoord= g_iup[idx][TUP];
 }
 else if (direction == TDOWN){
   scalarcoord= g_idn[idx][TUP];
 }
 else{
   scalarcoord=0;
   if (g_cart_id == 0) {printf("Wrong direction in tau scalar field spinor\n"); exit(1);}
 }
 if (dagger == DAGGER){
  if (tauindex == 0){
   if (smearedcorrelator_BSM  == 1){
     a11=  -1.*g_smeared_scalar_field[2][scalarcoord] - I*g_smeared_scalar_field[1][scalarcoord];
     a12=  +1.*g_smeared_scalar_field[0][scalarcoord] - I*g_smeared_scalar_field[3][scalarcoord];

     a21=  +1.*g_smeared_scalar_field[0][scalarcoord] + I*g_smeared_scalar_field[3][scalarcoord];
     a22=  +1.*g_smeared_scalar_field[2][scalarcoord] - I*g_smeared_scalar_field[1][scalarcoord];

   }
   else{
     a11=  -1.*g_scalar_field[2][scalarcoord] - I*g_scalar_field[1][scalarcoord];
     a12=  +1.*g_scalar_field[0][scalarcoord] - I*g_scalar_field[3][scalarcoord];

     a21=  +1.*g_scalar_field[0][scalarcoord] + I*g_scalar_field[3][scalarcoord];
     a22=  +1.*g_scalar_field[2][scalarcoord] - I*g_scalar_field[1][scalarcoord];
   }
  }
  else  if (tauindex == 1){
   if (smearedcorrelator_BSM  == 1){
     a11=  +1.*g_smeared_scalar_field[1][scalarcoord] - I*g_smeared_scalar_field[2][scalarcoord];
     a12=  -1.*g_smeared_scalar_field[3][scalarcoord] - I*g_smeared_scalar_field[0][scalarcoord];

     a21=  -1.*g_smeared_scalar_field[3][scalarcoord] + I*g_smeared_scalar_field[0][scalarcoord];
     a22=  -1.*g_smeared_scalar_field[1][scalarcoord] - I*g_smeared_scalar_field[2][scalarcoord];

   }
   else{
     a11=  +1.*g_scalar_field[1][scalarcoord] - I*g_scalar_field[2][scalarcoord];
     a12=  -1.*g_scalar_field[3][scalarcoord] - I*g_scalar_field[0][scalarcoord];

     a21=  -1.*g_scalar_field[3][scalarcoord] + I*g_scalar_field[0][scalarcoord];
     a22=  -1.*g_scalar_field[1][scalarcoord] - I*g_scalar_field[2][scalarcoord];
   }
  }
  else  if (tauindex == 2){
   if (smearedcorrelator_BSM  == 1){
     a11=  +1.*g_smeared_scalar_field[0][scalarcoord] - I*g_smeared_scalar_field[3][scalarcoord];
     a12=  +1.*g_smeared_scalar_field[2][scalarcoord] + I*g_smeared_scalar_field[1][scalarcoord];

     a21=  +1.*g_smeared_scalar_field[2][scalarcoord] - I*g_smeared_scalar_field[1][scalarcoord];
     a22=  -1.*g_smeared_scalar_field[0][scalarcoord] - I*g_smeared_scalar_field[3][scalarcoord];

   }
   else{
     a11=  +1.*g_scalar_field[0][scalarcoord] - I*g_scalar_field[3][scalarcoord];
     a12=  +1.*g_scalar_field[2][scalarcoord] + I*g_scalar_field[1][scalarcoord];

     a21=  +1.*g_scalar_field[2][scalarcoord] - I*g_scalar_field[1][scalarcoord];
     a22=  -1.*g_scalar_field[0][scalarcoord] - I*g_scalar_field[3][scalarcoord];
   }
  }
 }
 else if (dagger == NO_DAGG){
  if (tauindex == 0){
   if (smearedcorrelator_BSM  == 1){
     a11=  -1.*g_smeared_scalar_field[2][scalarcoord] + I*g_smeared_scalar_field[1][scalarcoord];
     a12=  +1.*g_smeared_scalar_field[0][scalarcoord] - I*g_smeared_scalar_field[3][scalarcoord];

     a21=  +1.*g_smeared_scalar_field[0][scalarcoord] + I*g_smeared_scalar_field[3][scalarcoord];
     a22=  +1.*g_smeared_scalar_field[2][scalarcoord] + I*g_smeared_scalar_field[1][scalarcoord];

   }
   else{
     a11=  -1.*g_scalar_field[2][scalarcoord] + I*g_scalar_field[1][scalarcoord];
     a12=  +1.*g_scalar_field[0][scalarcoord] - I*g_scalar_field[3][scalarcoord];

     a21=  +1.*g_scalar_field[0][scalarcoord] + I*g_scalar_field[3][scalarcoord];
     a22=  +1.*g_scalar_field[2][scalarcoord] + I*g_scalar_field[1][scalarcoord];
   }
  }
  else if (tauindex == 1){
   if (smearedcorrelator_BSM  == 1){
     a11=  +1.*g_smeared_scalar_field[1][scalarcoord] + I*g_smeared_scalar_field[2][scalarcoord];
     a12=  -1.*g_smeared_scalar_field[3][scalarcoord] - I*g_smeared_scalar_field[0][scalarcoord];

     a21=  -1.*g_smeared_scalar_field[3][scalarcoord] + I*g_smeared_scalar_field[0][scalarcoord];
     a22=  -1.*g_smeared_scalar_field[1][scalarcoord] + I*g_smeared_scalar_field[2][scalarcoord];
   }
   else{
     a11=  +1.*g_scalar_field[1][scalarcoord] + I*g_scalar_field[2][scalarcoord];
     a12=  -1.*g_scalar_field[3][scalarcoord] - I*g_scalar_field[0][scalarcoord];

     a21=  -1.*g_scalar_field[3][scalarcoord] + I*g_scalar_field[0][scalarcoord];
     a22=  -1.*g_scalar_field[1][scalarcoord] + I*g_scalar_field[2][scalarcoord];
   }
  }
  else if (tauindex == 2){
   if (smearedcorrelator_BSM  == 1){
     a11=  +1.*g_smeared_scalar_field[0][scalarcoord] + I*g_smeared_scalar_field[3][scalarcoord];
     a12=  +1.*g_smeared_scalar_field[2][scalarcoord] + I*g_smeared_scalar_field[1][scalarcoord];

     a21=  +1.*g_smeared_scalar_field[2][scalarcoord] - I*g_smeared_scalar_field[1][scalarcoord];
     a22=  -1.*g_smeared_scalar_field[0][scalarcoord] + I*g_smeared_scalar_field[3][scalarcoord];
   }
   else{
     a11=  +1.*g_scalar_field[0][scalarcoord] + I*g_scalar_field[3][scalarcoord];
     a12=  +1.*g_scalar_field[2][scalarcoord] + I*g_scalar_field[1][scalarcoord];

     a21=  +1.*g_scalar_field[2][scalarcoord] - I*g_scalar_field[1][scalarcoord];
     a22=  -1.*g_scalar_field[0][scalarcoord] + I*g_scalar_field[3][scalarcoord];
   }
  }
 }
 _spinor_null(tmpbi2.sp_up);
 _spinor_null(tmpbi2.sp_dn);
 
 if ( gamma5 == GAMMA_UP){
  _vector_mul_complex(    tmpbi2.sp_up.s0, a11, tmp.sp_up.s0);
  _vector_add_mul_complex(tmpbi2.sp_up.s0, a12, tmp.sp_dn.s0);

  _vector_mul_complex    (tmpbi2.sp_dn.s0, a21, tmp.sp_up.s0);
  _vector_add_mul_complex(tmpbi2.sp_dn.s0, a22, tmp.sp_dn.s0);

  _vector_mul_complex(    tmpbi2.sp_up.s1, a11, tmp.sp_up.s1);
  _vector_add_mul_complex(tmpbi2.sp_up.s1, a12, tmp.sp_dn.s1);

  _vector_mul_complex    (tmpbi2.sp_dn.s1, a21, tmp.sp_up.s1);
  _vector_add_mul_complex(tmpbi2.sp_dn.s1, a22, tmp.sp_dn.s1);
 }
 else if  ( gamma5 == GAMMA_DN ){
  _vector_mul_complex(    tmpbi2.sp_up.s2, a11, tmp.sp_up.s2);
  _vector_add_mul_complex(tmpbi2.sp_up.s2, a12, tmp.sp_dn.s2);

  _vector_mul_complex    (tmpbi2.sp_dn.s2, a21, tmp.sp_up.s2);
  _vector_add_mul_complex(tmpbi2.sp_dn.s2, a22, tmp.sp_dn.s2);

  _vector_mul_complex(    tmpbi2.sp_up.s3, a11, tmp.sp_up.s3);
  _vector_add_mul_complex(tmpbi2.sp_up.s3, a12, tmp.sp_dn.s3);

  _vector_mul_complex    (tmpbi2.sp_dn.s3, a21, tmp.sp_up.s3);
  _vector_add_mul_complex(tmpbi2.sp_dn.s3, a22, tmp.sp_dn.s3);
 }
 else if ( gamma5 == NO_GAMMA ){
  _spinor_mul_complex    (tmpbi2.sp_up,    a11, tmp.sp_up);
  _spinor_add_mul_complex(tmpbi2.sp_up,    a12, tmp.sp_dn);

  _spinor_mul_complex    (tmpbi2.sp_dn,    a21, tmp.sp_up);
  _spinor_add_mul_complex(tmpbi2.sp_dn,    a22, tmp.sp_dn);
 }

 _spinor_assign(dest->sp_up, tmpbi2.sp_up);
 _spinor_assign(dest->sp_dn, tmpbi2.sp_dn);

}
void trace_in_spinor( _Complex double *dest, _Complex double *src, int spinorindex){
   int tind, find;
   for (tind=0; tind<T_global; ++tind)
     for (find=0; find<2; ++find){ 
       dest[2*tind+find]+=src[8*tind+4*find+spinorindex];
     }
}
void trace_in_color(_Complex double *dest, bispinor *src, int colorindex){
   if      ( colorindex == 0 ){
     dest[0]+= src->sp_up.s0.c0;
     dest[1]+= src->sp_up.s1.c0;
     dest[2]+= src->sp_up.s2.c0;
     dest[3]+= src->sp_up.s3.c0;
     dest[4]+= src->sp_dn.s0.c0;
     dest[5]+= src->sp_dn.s1.c0;
     dest[6]+= src->sp_dn.s2.c0;
     dest[7]+= src->sp_dn.s3.c0;

   }
   else if ( colorindex == 1 ){
     dest[0]+= src->sp_up.s0.c1;
     dest[1]+= src->sp_up.s1.c1;
     dest[2]+= src->sp_up.s2.c1;
     dest[3]+= src->sp_up.s3.c1;
     dest[4]+= src->sp_dn.s0.c1;
     dest[5]+= src->sp_dn.s1.c1;
     dest[6]+= src->sp_dn.s2.c1;
     dest[7]+= src->sp_dn.s3.c1;
   }
   else if ( colorindex == 2 ){
     dest[0]+= src->sp_up.s0.c2;
     dest[1]+= src->sp_up.s1.c2;
     dest[2]+= src->sp_up.s2.c2;
     dest[3]+= src->sp_up.s3.c2;
     dest[4]+= src->sp_dn.s0.c2;
     dest[5]+= src->sp_dn.s1.c2;
     dest[6]+= src->sp_dn.s2.c2;
     dest[7]+= src->sp_dn.s3.c2;
   }
}
void trace_in_space(_Complex double *dest, _Complex double *source, int idx){
     int i;
     for (i=0; i<8;++i){
       dest[g_coord[idx][TUP]*8+i]+= source[i];
     }
}
void trace_in_flavor(_Complex double *dest, _Complex double *source, int f1){
     int i;
     for (i=0; i<T_global; ++i){
        dest[i]+= source[2*i+f1];
     }
}
#endif
