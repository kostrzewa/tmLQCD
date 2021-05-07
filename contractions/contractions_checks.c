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

static void trace_in_spinor_and_color( _Complex double *c, bispinor **prop, int ix, int f3, int f4, int f6, int f1){
     int alpha2;
     int c1;
     c[ix]=0.;
     for (alpha2=0; alpha2<2;++alpha2)
       for (c1=0; c1<3; ++c1){
          if ( (f6 == 0) && (f4==0) ){
             c[ix]+= prop[12*alpha2+4*c1+2*f1][ix].sp_up.s2.c0*conj(prop[12*alpha2+4*c1+2*f3+1][ix].sp_up.s2.c0)
                    +prop[12*alpha2+4*c1+2*f1][ix].sp_up.s2.c1*conj(prop[12*alpha2+4*c1+2*f3+1][ix].sp_up.s2.c1)
                    +prop[12*alpha2+4*c1+2*f1][ix].sp_up.s2.c2*conj(prop[12*alpha2+4*c1+2*f3+1][ix].sp_up.s2.c2)
                    +prop[12*alpha2+4*c1+2*f1][ix].sp_up.s3.c0*conj(prop[12*alpha2+4*c1+2*f3+1][ix].sp_up.s3.c0)
                    +prop[12*alpha2+4*c1+2*f1][ix].sp_up.s3.c1*conj(prop[12*alpha2+4*c1+2*f3+1][ix].sp_up.s3.c1)
                    +prop[12*alpha2+4*c1+2*f1][ix].sp_up.s3.c2*conj(prop[12*alpha2+4*c1+2*f3+1][ix].sp_up.s3.c2);
          }
          if ( (f6 == 1) && (f4==0) ){
             c[ix]+= prop[12*alpha2+4*c1+2*f1][ix].sp_dn.s2.c0*conj(prop[12*alpha2+4*c1+2*f3+1][ix].sp_up.s2.c0)
                    +prop[12*alpha2+4*c1+2*f1][ix].sp_dn.s2.c1*conj(prop[12*alpha2+4*c1+2*f3+1][ix].sp_up.s2.c1)
                    +prop[12*alpha2+4*c1+2*f1][ix].sp_dn.s2.c2*conj(prop[12*alpha2+4*c1+2*f3+1][ix].sp_up.s2.c2)
                    +prop[12*alpha2+4*c1+2*f1][ix].sp_dn.s3.c0*conj(prop[12*alpha2+4*c1+2*f3+1][ix].sp_up.s3.c0)
                    +prop[12*alpha2+4*c1+2*f1][ix].sp_dn.s3.c1*conj(prop[12*alpha2+4*c1+2*f3+1][ix].sp_up.s3.c1)
                    +prop[12*alpha2+4*c1+2*f1][ix].sp_dn.s3.c2*conj(prop[12*alpha2+4*c1+2*f3+1][ix].sp_up.s3.c2);
          }
          if ( (f6 == 0) && (f4==1) ){
             c[ix]+= prop[12*alpha2+4*c1+2*f1][ix].sp_up.s2.c0*conj(prop[12*alpha2+4*c1+2*f3+1][ix].sp_dn.s2.c0)
                    +prop[12*alpha2+4*c1+2*f1][ix].sp_up.s2.c1*conj(prop[12*alpha2+4*c1+2*f3+1][ix].sp_dn.s2.c1)
                    +prop[12*alpha2+4*c1+2*f1][ix].sp_up.s2.c2*conj(prop[12*alpha2+4*c1+2*f3+1][ix].sp_dn.s2.c2)
                    +prop[12*alpha2+4*c1+2*f1][ix].sp_up.s3.c0*conj(prop[12*alpha2+4*c1+2*f3+1][ix].sp_dn.s3.c0)
                    +prop[12*alpha2+4*c1+2*f1][ix].sp_up.s3.c1*conj(prop[12*alpha2+4*c1+2*f3+1][ix].sp_dn.s3.c1)
                    +prop[12*alpha2+4*c1+2*f1][ix].sp_up.s3.c2*conj(prop[12*alpha2+4*c1+2*f3+1][ix].sp_dn.s3.c2);
          }
          if ( (f6 == 1) && (f4==1) ){
             c[ix]+= prop[12*alpha2+4*c1+2*f1][ix].sp_dn.s2.c0*conj(prop[12*alpha2+4*c1+2*f3+1][ix].sp_dn.s2.c0)
                    +prop[12*alpha2+4*c1+2*f1][ix].sp_dn.s2.c1*conj(prop[12*alpha2+4*c1+2*f3+1][ix].sp_dn.s2.c1)
                    +prop[12*alpha2+4*c1+2*f1][ix].sp_dn.s2.c2*conj(prop[12*alpha2+4*c1+2*f3+1][ix].sp_dn.s2.c2)
                    +prop[12*alpha2+4*c1+2*f1][ix].sp_dn.s3.c0*conj(prop[12*alpha2+4*c1+2*f3+1][ix].sp_dn.s3.c0)
                    +prop[12*alpha2+4*c1+2*f1][ix].sp_dn.s3.c1*conj(prop[12*alpha2+4*c1+2*f3+1][ix].sp_dn.s3.c1)
                    +prop[12*alpha2+4*c1+2*f1][ix].sp_dn.s3.c2*conj(prop[12*alpha2+4*c1+2*f3+1][ix].sp_dn.s3.c2);
          }
       }
}
static void trace_in_spinor_and_color62a( _Complex double *c, bispinor **prop, int ix, int f3, int f4, int f6, int f1){
     int alpha2;
     int c1;
     c[ix]=0.;
     bispinor running;
     su3 * restrict upm;
     su3_vector tmpvec;
     for (alpha2=0; alpha2<2;++alpha2)
       for (c1=0; c1<3; ++c1){
          if ( (f6 == 0) && (f4==0) ){
              upm = &g_gauge_field[ix][TUP];

              _vector_null( running.sp_up.s0 );
              _vector_null( running.sp_up.s1 );
              _vector_null( running.sp_up.s2 );
              _vector_null( running.sp_up.s3 );


              _su3_multiply( running.sp_up.s2, (*upm), prop[12*alpha2+4*c1+2*f1][g_iup[ix][TUP]].sp_up.s2 );
              _su3_multiply( running.sp_up.s3, (*upm), prop[12*alpha2+4*c1+2*f1][g_iup[ix][TUP]].sp_up.s3 );

              upm = &g_gauge_field[g_idn[ix][TUP]][TUP];

              _vector_null( tmpvec );
              _su3_multiply(tmpvec, (*upm), running.sp_up.s2);
              _vector_assign( running.sp_up.s2, tmpvec);

              _vector_null( tmpvec );
              _su3_multiply(tmpvec, (*upm), running.sp_up.s3);
              _vector_assign( running.sp_up.s3, tmpvec);

              _complex_times_vector(running.sp_up.s2,phase_00,running.sp_up.s2);
              _complex_times_vector(running.sp_up.s3,phase_00,running.sp_up.s3);

 
              c[ix]+= running.sp_up.s2.c0*conj(prop[12*alpha2+4*c1+2*f3+1][g_idn[ix][TUP]].sp_up.s2.c0)
                     +running.sp_up.s2.c1*conj(prop[12*alpha2+4*c1+2*f3+1][g_idn[ix][TUP]].sp_up.s2.c1)
                     +running.sp_up.s2.c2*conj(prop[12*alpha2+4*c1+2*f3+1][g_idn[ix][TUP]].sp_up.s2.c2)
                     +running.sp_up.s3.c0*conj(prop[12*alpha2+4*c1+2*f3+1][g_idn[ix][TUP]].sp_up.s3.c0)
                     +running.sp_up.s3.c1*conj(prop[12*alpha2+4*c1+2*f3+1][g_idn[ix][TUP]].sp_up.s3.c1)
                     +running.sp_up.s3.c2*conj(prop[12*alpha2+4*c1+2*f3+1][g_idn[ix][TUP]].sp_up.s3.c2);
          }
          if ( (f6 == 1) && (f4==0) ){
              upm = &g_gauge_field[ix][TUP];

              _vector_null( running.sp_dn.s0 );
              _vector_null( running.sp_dn.s1 );
              _vector_null( running.sp_dn.s2 );
              _vector_null( running.sp_dn.s3 );


              _su3_multiply( running.sp_dn.s2, (*upm), prop[12*alpha2+4*c1+2*f1][g_iup[ix][TUP]].sp_dn.s2 );
              _su3_multiply( running.sp_dn.s3, (*upm), prop[12*alpha2+4*c1+2*f1][g_iup[ix][TUP]].sp_dn.s3 );

              upm = &g_gauge_field[g_idn[ix][TUP]][TUP];

              _vector_null( tmpvec );
              _su3_multiply(tmpvec, (*upm), running.sp_dn.s2);
              _vector_assign( running.sp_dn.s2, tmpvec);

              _vector_null( tmpvec );
              _su3_multiply(tmpvec, (*upm), running.sp_dn.s3);
              _vector_assign( running.sp_dn.s3, tmpvec);

              _complex_times_vector(running.sp_dn.s2,phase_00,running.sp_dn.s2);
              _complex_times_vector(running.sp_dn.s3,phase_00,running.sp_dn.s3);


              c[ix]+= running.sp_dn.s2.c0*conj(prop[12*alpha2+4*c1+2*f3+1][g_idn[ix][TUP]].sp_up.s2.c0)
                     +running.sp_dn.s2.c1*conj(prop[12*alpha2+4*c1+2*f3+1][g_idn[ix][TUP]].sp_up.s2.c1)
                     +running.sp_dn.s2.c2*conj(prop[12*alpha2+4*c1+2*f3+1][g_idn[ix][TUP]].sp_up.s2.c2)
                     +running.sp_dn.s3.c0*conj(prop[12*alpha2+4*c1+2*f3+1][g_idn[ix][TUP]].sp_up.s3.c0)
                     +running.sp_dn.s3.c1*conj(prop[12*alpha2+4*c1+2*f3+1][g_idn[ix][TUP]].sp_up.s3.c1)
                     +running.sp_dn.s3.c2*conj(prop[12*alpha2+4*c1+2*f3+1][g_idn[ix][TUP]].sp_up.s3.c2);
          }
          if ( (f6 == 0) && (f4==1) ){
              upm = &g_gauge_field[ix][TUP];

              _vector_null( running.sp_up.s0 );
              _vector_null( running.sp_up.s1 );
              _vector_null( running.sp_up.s2 );
              _vector_null( running.sp_up.s3 );


              _su3_multiply( running.sp_up.s2, (*upm), prop[12*alpha2+4*c1+2*f1][g_iup[ix][TUP]].sp_up.s2 );
              _su3_multiply( running.sp_up.s3, (*upm), prop[12*alpha2+4*c1+2*f1][g_iup[ix][TUP]].sp_up.s3 );

              upm = &g_gauge_field[g_idn[ix][TUP]][TUP];

              _vector_null( tmpvec );
              _su3_multiply(tmpvec, (*upm), running.sp_up.s2);
              _vector_assign( running.sp_up.s2, tmpvec);

              _vector_null( tmpvec );
              _su3_multiply(tmpvec, (*upm), running.sp_up.s3);
              _vector_assign( running.sp_up.s3, tmpvec);

              _complex_times_vector(running.sp_up.s2,phase_00,running.sp_up.s2);
              _complex_times_vector(running.sp_up.s3,phase_00,running.sp_up.s3);


              c[ix]+= running.sp_up.s2.c0*conj(prop[12*alpha2+4*c1+2*f3+1][g_idn[ix][TUP]].sp_dn.s2.c0)
                     +running.sp_up.s2.c1*conj(prop[12*alpha2+4*c1+2*f3+1][g_idn[ix][TUP]].sp_dn.s2.c1)
                     +running.sp_up.s2.c2*conj(prop[12*alpha2+4*c1+2*f3+1][g_idn[ix][TUP]].sp_dn.s2.c2)
                     +running.sp_up.s3.c0*conj(prop[12*alpha2+4*c1+2*f3+1][g_idn[ix][TUP]].sp_dn.s3.c0)
                     +running.sp_up.s3.c1*conj(prop[12*alpha2+4*c1+2*f3+1][g_idn[ix][TUP]].sp_dn.s3.c1)
                     +running.sp_up.s3.c2*conj(prop[12*alpha2+4*c1+2*f3+1][g_idn[ix][TUP]].sp_dn.s3.c2);
          }
          if ( (f6 == 1) && (f4==1) ){
              upm = &g_gauge_field[ix][TUP];

              _vector_null( running.sp_dn.s0 );
              _vector_null( running.sp_dn.s1 );
              _vector_null( running.sp_dn.s2 );
              _vector_null( running.sp_dn.s3 );


              _su3_multiply( running.sp_dn.s2, (*upm), prop[12*alpha2+4*c1+2*f1][g_iup[ix][TUP]].sp_dn.s2 );
              _su3_multiply( running.sp_dn.s3, (*upm), prop[12*alpha2+4*c1+2*f1][g_iup[ix][TUP]].sp_dn.s3 );

              upm = &g_gauge_field[g_idn[ix][TUP]][TUP];

              _vector_null( tmpvec );
              _su3_multiply(tmpvec, (*upm), running.sp_dn.s2);
              _vector_assign( running.sp_dn.s2, tmpvec);

              _vector_null( tmpvec );
              _su3_multiply(tmpvec, (*upm), running.sp_dn.s3);
              _vector_assign( running.sp_dn.s3, tmpvec);

              _complex_times_vector(running.sp_dn.s2,phase_00,running.sp_dn.s2);
              _complex_times_vector(running.sp_dn.s3,phase_00,running.sp_dn.s3);


              c[ix]+= running.sp_dn.s2.c0*conj(prop[12*alpha2+4*c1+2*f3+1][g_idn[ix][TUP]].sp_dn.s2.c0)
                     +running.sp_dn.s2.c1*conj(prop[12*alpha2+4*c1+2*f3+1][g_idn[ix][TUP]].sp_dn.s2.c1)
                     +running.sp_dn.s2.c2*conj(prop[12*alpha2+4*c1+2*f3+1][g_idn[ix][TUP]].sp_dn.s2.c2)
                     +running.sp_dn.s3.c0*conj(prop[12*alpha2+4*c1+2*f3+1][g_idn[ix][TUP]].sp_dn.s3.c0)
                     +running.sp_dn.s3.c1*conj(prop[12*alpha2+4*c1+2*f3+1][g_idn[ix][TUP]].sp_dn.s3.c1)
                     +running.sp_dn.s3.c2*conj(prop[12*alpha2+4*c1+2*f3+1][g_idn[ix][TUP]].sp_dn.s3.c2);
          }
       }
}
static void trace_in_spinor_and_color61b( _Complex double *c, bispinor **prop, int ix, int f3, int f4, int f6, int f1){
     int alpha2;
     int c1;
     c[ix]=0.;
     for (alpha2=2; alpha2<4;++alpha2)
       for (c1=0; c1<3; ++c1){
          if ( (f6 == 0) && (f4==0) ){
             c[ix]+= prop[12*alpha2+4*c1+2*f1][ix].sp_up.s2.c0*conj(prop[12*alpha2+4*c1+2*f3+1][ix].sp_up.s2.c0)
                    +prop[12*alpha2+4*c1+2*f1][ix].sp_up.s2.c1*conj(prop[12*alpha2+4*c1+2*f3+1][ix].sp_up.s2.c1)
                    +prop[12*alpha2+4*c1+2*f1][ix].sp_up.s2.c2*conj(prop[12*alpha2+4*c1+2*f3+1][ix].sp_up.s2.c2)
                    +prop[12*alpha2+4*c1+2*f1][ix].sp_up.s3.c0*conj(prop[12*alpha2+4*c1+2*f3+1][ix].sp_up.s3.c0)
                    +prop[12*alpha2+4*c1+2*f1][ix].sp_up.s3.c1*conj(prop[12*alpha2+4*c1+2*f3+1][ix].sp_up.s3.c1)
                    +prop[12*alpha2+4*c1+2*f1][ix].sp_up.s3.c2*conj(prop[12*alpha2+4*c1+2*f3+1][ix].sp_up.s3.c2);
          }
          if ( (f6 == 1) && (f4==0) ){
             c[ix]+= prop[12*alpha2+4*c1+2*f1][ix].sp_dn.s2.c0*conj(prop[12*alpha2+4*c1+2*f3+1][ix].sp_up.s2.c0)
                    +prop[12*alpha2+4*c1+2*f1][ix].sp_dn.s2.c1*conj(prop[12*alpha2+4*c1+2*f3+1][ix].sp_up.s2.c1)
                    +prop[12*alpha2+4*c1+2*f1][ix].sp_dn.s2.c2*conj(prop[12*alpha2+4*c1+2*f3+1][ix].sp_up.s2.c2)
                    +prop[12*alpha2+4*c1+2*f1][ix].sp_dn.s3.c0*conj(prop[12*alpha2+4*c1+2*f3+1][ix].sp_up.s3.c0)
                    +prop[12*alpha2+4*c1+2*f1][ix].sp_dn.s3.c1*conj(prop[12*alpha2+4*c1+2*f3+1][ix].sp_up.s3.c1)
                    +prop[12*alpha2+4*c1+2*f1][ix].sp_dn.s3.c2*conj(prop[12*alpha2+4*c1+2*f3+1][ix].sp_up.s3.c2);
          }
          if ( (f6 == 0) && (f4==1) ){
             c[ix]+= prop[12*alpha2+4*c1+2*f1][ix].sp_up.s2.c0*conj(prop[12*alpha2+4*c1+2*f3+1][ix].sp_dn.s2.c0)
                    +prop[12*alpha2+4*c1+2*f1][ix].sp_up.s2.c1*conj(prop[12*alpha2+4*c1+2*f3+1][ix].sp_dn.s2.c1)
                    +prop[12*alpha2+4*c1+2*f1][ix].sp_up.s2.c2*conj(prop[12*alpha2+4*c1+2*f3+1][ix].sp_dn.s2.c2)
                    +prop[12*alpha2+4*c1+2*f1][ix].sp_up.s3.c0*conj(prop[12*alpha2+4*c1+2*f3+1][ix].sp_dn.s3.c0)
                    +prop[12*alpha2+4*c1+2*f1][ix].sp_up.s3.c1*conj(prop[12*alpha2+4*c1+2*f3+1][ix].sp_dn.s3.c1)
                    +prop[12*alpha2+4*c1+2*f1][ix].sp_up.s3.c2*conj(prop[12*alpha2+4*c1+2*f3+1][ix].sp_dn.s3.c2);
          }
          if ( (f6 == 1) && (f4==1) ){
             c[ix]+= prop[12*alpha2+4*c1+2*f1][ix].sp_dn.s2.c0*conj(prop[12*alpha2+4*c1+2*f3+1][ix].sp_dn.s2.c0)
                    +prop[12*alpha2+4*c1+2*f1][ix].sp_dn.s2.c1*conj(prop[12*alpha2+4*c1+2*f3+1][ix].sp_dn.s2.c1)
                    +prop[12*alpha2+4*c1+2*f1][ix].sp_dn.s2.c2*conj(prop[12*alpha2+4*c1+2*f3+1][ix].sp_dn.s2.c2)
                    +prop[12*alpha2+4*c1+2*f1][ix].sp_dn.s3.c0*conj(prop[12*alpha2+4*c1+2*f3+1][ix].sp_dn.s3.c0)
                    +prop[12*alpha2+4*c1+2*f1][ix].sp_dn.s3.c1*conj(prop[12*alpha2+4*c1+2*f3+1][ix].sp_dn.s3.c1)
                    +prop[12*alpha2+4*c1+2*f1][ix].sp_dn.s3.c2*conj(prop[12*alpha2+4*c1+2*f3+1][ix].sp_dn.s3.c2);
          }
       }
}
static void trace_in_spinor_and_color1a( _Complex double *c, bispinor **prop, int ix, int f3, int f4, int f6, int f1){
     int alpha1;
     int c1;
     c[ix]=0.;
     bispinor running;
     su3 * restrict upm;
     for (alpha1=0; alpha1<2;++alpha1)
       for (c1=0; c1<3; ++c1){
          if ( (f6 == 0) && (f4==0) ){
              upm = &g_gauge_field[g_idn[ix][TUP]][TUP];

              _vector_null( running.sp_up.s0 );
              _vector_null( running.sp_up.s1 );
              _vector_null( running.sp_up.s2 );
              _vector_null( running.sp_up.s3 );


              _su3_multiply( running.sp_up.s2, (*upm), prop[12*alpha1+4*c1+2*f1][ix].sp_up.s2 );
              _su3_multiply( running.sp_up.s3, (*upm), prop[12*alpha1+4*c1+2*f1][ix].sp_up.s3 );

              _complex_times_vector(running.sp_up.s2,phase_0,running.sp_up.s2);
              _complex_times_vector(running.sp_up.s3,phase_0,running.sp_up.s3);
 

              _vector_add_assign(running.sp_up.s0, running.sp_up.s2);
              _vector_add_assign(running.sp_up.s1, running.sp_up.s3);
              _vector_null(running.sp_up.s2);
              _vector_null(running.sp_up.s3);

              c[ix]+= running.sp_up.s0.c0*conj(prop[12*alpha1+4*c1+2*f3+1][g_idn[ix][TUP]].sp_up.s0.c0)
                     +running.sp_up.s0.c1*conj(prop[12*alpha1+4*c1+2*f3+1][g_idn[ix][TUP]].sp_up.s0.c1)
                     +running.sp_up.s0.c2*conj(prop[12*alpha1+4*c1+2*f3+1][g_idn[ix][TUP]].sp_up.s0.c2)
                     +running.sp_up.s1.c0*conj(prop[12*alpha1+4*c1+2*f3+1][g_idn[ix][TUP]].sp_up.s1.c0)
                     +running.sp_up.s1.c1*conj(prop[12*alpha1+4*c1+2*f3+1][g_idn[ix][TUP]].sp_up.s1.c1)
                     +running.sp_up.s1.c2*conj(prop[12*alpha1+4*c1+2*f3+1][g_idn[ix][TUP]].sp_up.s1.c2);
          }
          if ( (f6 == 1) && (f4==0) ){
              upm = &g_gauge_field[g_idn[ix][TUP]][TUP];

              _vector_null( running.sp_dn.s0 );
              _vector_null( running.sp_dn.s1 );
              _vector_null( running.sp_dn.s2 );
              _vector_null( running.sp_dn.s3 );


              _su3_multiply( running.sp_dn.s2, (*upm), prop[12*alpha1+4*c1+2*f1][ix].sp_dn.s2 );
              _su3_multiply( running.sp_dn.s3, (*upm), prop[12*alpha1+4*c1+2*f1][ix].sp_dn.s3 );


              _complex_times_vector(running.sp_dn.s2,phase_0,running.sp_dn.s2);
              _complex_times_vector(running.sp_dn.s3,phase_0,running.sp_dn.s3);

              _vector_add_assign(running.sp_dn.s0, running.sp_dn.s2);
              _vector_add_assign(running.sp_dn.s1, running.sp_dn.s3);
              _vector_null(running.sp_dn.s2);
              _vector_null(running.sp_dn.s3);

              c[ix]+= running.sp_dn.s0.c0*conj(prop[12*alpha1+4*c1+2*f3+1][g_idn[ix][TUP]].sp_up.s0.c0)
                     +running.sp_dn.s0.c1*conj(prop[12*alpha1+4*c1+2*f3+1][g_idn[ix][TUP]].sp_up.s0.c1)
                     +running.sp_dn.s0.c2*conj(prop[12*alpha1+4*c1+2*f3+1][g_idn[ix][TUP]].sp_up.s0.c2)
                     +running.sp_dn.s1.c0*conj(prop[12*alpha1+4*c1+2*f3+1][g_idn[ix][TUP]].sp_up.s1.c0)
                     +running.sp_dn.s1.c1*conj(prop[12*alpha1+4*c1+2*f3+1][g_idn[ix][TUP]].sp_up.s1.c1)
                     +running.sp_dn.s1.c2*conj(prop[12*alpha1+4*c1+2*f3+1][g_idn[ix][TUP]].sp_up.s1.c2);
          }
          if ( (f6 == 0) && (f4==1) ){
              upm = &g_gauge_field[g_idn[ix][TUP]][TUP];

              _vector_null( running.sp_up.s0 );
              _vector_null( running.sp_up.s1 );
              _vector_null( running.sp_up.s2 );
              _vector_null( running.sp_up.s3 );


              _su3_multiply( running.sp_up.s2, (*upm), prop[12*alpha1+4*c1+2*f1][ix].sp_up.s2 );
              _su3_multiply( running.sp_up.s3, (*upm), prop[12*alpha1+4*c1+2*f1][ix].sp_up.s3 );

              _complex_times_vector(running.sp_up.s2,phase_0,running.sp_up.s2);
              _complex_times_vector(running.sp_up.s3,phase_0,running.sp_up.s3);

              _vector_add_assign(running.sp_up.s0, running.sp_up.s2);
              _vector_add_assign(running.sp_up.s1, running.sp_up.s3);
              _vector_null(running.sp_up.s2);
              _vector_null(running.sp_up.s3);

              c[ix]+= running.sp_up.s0.c0*conj(prop[12*alpha1+4*c1+2*f3+1][g_idn[ix][TUP]].sp_dn.s0.c0)
                     +running.sp_up.s0.c1*conj(prop[12*alpha1+4*c1+2*f3+1][g_idn[ix][TUP]].sp_dn.s0.c1)
                     +running.sp_up.s0.c2*conj(prop[12*alpha1+4*c1+2*f3+1][g_idn[ix][TUP]].sp_dn.s0.c2)
                     +running.sp_up.s1.c0*conj(prop[12*alpha1+4*c1+2*f3+1][g_idn[ix][TUP]].sp_dn.s1.c0)
                     +running.sp_up.s1.c1*conj(prop[12*alpha1+4*c1+2*f3+1][g_idn[ix][TUP]].sp_dn.s1.c1)
                     +running.sp_up.s1.c2*conj(prop[12*alpha1+4*c1+2*f3+1][g_idn[ix][TUP]].sp_dn.s1.c2);
          }
          if ( (f6 == 1) && (f4==1) ){
              upm = &g_gauge_field[g_idn[ix][TUP]][TUP];

              _vector_null( running.sp_dn.s0 );
              _vector_null( running.sp_dn.s1 );
              _vector_null( running.sp_dn.s2 );
              _vector_null( running.sp_dn.s3 );


              _su3_multiply( running.sp_dn.s2, (*upm), prop[12*alpha1+4*c1+2*f1][ix].sp_dn.s2 );
              _su3_multiply( running.sp_dn.s3, (*upm), prop[12*alpha1+4*c1+2*f1][ix].sp_dn.s3 );

              _complex_times_vector(running.sp_dn.s2,phase_0,running.sp_dn.s2);
              _complex_times_vector(running.sp_dn.s3,phase_0,running.sp_dn.s3);


              _vector_add_assign(running.sp_dn.s0, running.sp_dn.s2);
              _vector_add_assign(running.sp_dn.s1, running.sp_dn.s3);
              _vector_null(running.sp_dn.s2);
              _vector_null(running.sp_dn.s3);

              c[ix]+= running.sp_dn.s0.c0*conj(prop[12*alpha1+4*c1+2*f3+1][g_idn[ix][TUP]].sp_dn.s0.c0)
                     +running.sp_dn.s0.c1*conj(prop[12*alpha1+4*c1+2*f3+1][g_idn[ix][TUP]].sp_dn.s0.c1)
                     +running.sp_dn.s0.c2*conj(prop[12*alpha1+4*c1+2*f3+1][g_idn[ix][TUP]].sp_dn.s0.c2)
                     +running.sp_dn.s1.c0*conj(prop[12*alpha1+4*c1+2*f3+1][g_idn[ix][TUP]].sp_dn.s1.c0)
                     +running.sp_dn.s1.c1*conj(prop[12*alpha1+4*c1+2*f3+1][g_idn[ix][TUP]].sp_dn.s1.c1)
                     +running.sp_dn.s1.c2*conj(prop[12*alpha1+4*c1+2*f3+1][g_idn[ix][TUP]].sp_dn.s1.c2);
          }
       }
}



static void trace_in_spinor_and_color3a( _Complex double *c, bispinor **prop, int ix, int f3, int f4, int f6, int f1){
     int alpha1;
     int c1;
     c[ix]=0.;
     bispinor running;
     su3 * restrict upm;
     bispinor tmp;
     for (alpha1=0; alpha1<2;++alpha1)
       for (c1=0; c1<3; ++c1){
          if ( (f6 == 0) && (f4==0) ){
              upm = &g_gauge_field[ix][TUP];

              _vector_null( tmp.sp_up.s2 );
              _vector_null( tmp.sp_up.s3 );
              _vector_null( tmp.sp_up.s0 );
              _vector_null( tmp.sp_up.s1 );

              _vector_null( running.sp_up.s0 );
              _vector_null( running.sp_up.s1 );
              _vector_null( running.sp_up.s2 );
              _vector_null( running.sp_up.s3 );



              _su3_multiply( tmp.sp_up.s0, (*upm), prop[12*alpha1+4*c1+2*f1][g_iup[ix][TUP]].sp_up.s0 );
              _su3_multiply( tmp.sp_up.s1, (*upm), prop[12*alpha1+4*c1+2*f1][g_iup[ix][TUP]].sp_up.s1 );

              upm = &g_gauge_field[g_idn[ix][TUP]][TUP];

              _su3_multiply( running.sp_up.s0, (*upm), tmp.sp_up.s0 );
              _su3_multiply( running.sp_up.s1, (*upm), tmp.sp_up.s1 );

              _complex_times_vector(running.sp_up.s0,phase_00,running.sp_up.s0);
              _complex_times_vector(running.sp_up.s1,phase_00,running.sp_up.s1);



              c[ix]+= running.sp_up.s0.c0*conj(prop[12*alpha1+4*c1+2*f3+1][g_idn[ix][TUP]].sp_up.s0.c0)
                     +running.sp_up.s0.c1*conj(prop[12*alpha1+4*c1+2*f3+1][g_idn[ix][TUP]].sp_up.s0.c1)
                     +running.sp_up.s0.c2*conj(prop[12*alpha1+4*c1+2*f3+1][g_idn[ix][TUP]].sp_up.s0.c2)
                     +running.sp_up.s1.c0*conj(prop[12*alpha1+4*c1+2*f3+1][g_idn[ix][TUP]].sp_up.s1.c0)
                     +running.sp_up.s1.c1*conj(prop[12*alpha1+4*c1+2*f3+1][g_idn[ix][TUP]].sp_up.s1.c1)
                     +running.sp_up.s1.c2*conj(prop[12*alpha1+4*c1+2*f3+1][g_idn[ix][TUP]].sp_up.s1.c2);
          }
          if ( (f6 == 1) && (f4==0) ){
              upm = &g_gauge_field[ix][TUP];

              _vector_null( tmp.sp_dn.s2 );
              _vector_null( tmp.sp_dn.s3 );
              _vector_null( tmp.sp_dn.s0 );
              _vector_null( tmp.sp_dn.s1 );

              _vector_null( running.sp_dn.s0 );
              _vector_null( running.sp_dn.s1 );
              _vector_null( running.sp_dn.s2 );
              _vector_null( running.sp_dn.s3 );


              _su3_multiply( tmp.sp_dn.s0, (*upm), prop[12*alpha1+4*c1+2*f1][g_iup[ix][TUP]].sp_dn.s0 );
              _su3_multiply( tmp.sp_dn.s1, (*upm), prop[12*alpha1+4*c1+2*f1][g_iup[ix][TUP]].sp_dn.s1 );

              upm = &g_gauge_field[g_idn[ix][TUP]][TUP];

              _su3_multiply( running.sp_dn.s0, (*upm), tmp.sp_dn.s0 );
              _su3_multiply( running.sp_dn.s1, (*upm), tmp.sp_dn.s1 );


              _complex_times_vector(running.sp_dn.s0,phase_00,running.sp_dn.s0);
              _complex_times_vector(running.sp_dn.s1,phase_00,running.sp_dn.s1);


              c[ix]+= running.sp_dn.s0.c0*conj(prop[12*alpha1+4*c1+2*f3+1][g_idn[ix][TUP]].sp_up.s0.c0)
                     +running.sp_dn.s0.c1*conj(prop[12*alpha1+4*c1+2*f3+1][g_idn[ix][TUP]].sp_up.s0.c1)
                     +running.sp_dn.s0.c2*conj(prop[12*alpha1+4*c1+2*f3+1][g_idn[ix][TUP]].sp_up.s0.c2)
                     +running.sp_dn.s1.c0*conj(prop[12*alpha1+4*c1+2*f3+1][g_idn[ix][TUP]].sp_up.s1.c0)
                     +running.sp_dn.s1.c1*conj(prop[12*alpha1+4*c1+2*f3+1][g_idn[ix][TUP]].sp_up.s1.c1)
                     +running.sp_dn.s1.c2*conj(prop[12*alpha1+4*c1+2*f3+1][g_idn[ix][TUP]].sp_up.s1.c2);

          }
          if ( (f6 == 0) && (f4==1) ){
              upm = &g_gauge_field[ix][TUP];

              _vector_null( tmp.sp_up.s2 );
              _vector_null( tmp.sp_up.s3 );
              _vector_null( tmp.sp_up.s0 );
              _vector_null( tmp.sp_up.s1 );

              _vector_null( running.sp_up.s0 );
              _vector_null( running.sp_up.s1 );
              _vector_null( running.sp_up.s2 );
              _vector_null( running.sp_up.s3 );

              _su3_multiply( tmp.sp_up.s0, (*upm), prop[12*alpha1+4*c1+2*f1][g_iup[ix][TUP]].sp_up.s0 );
              _su3_multiply( tmp.sp_up.s1, (*upm), prop[12*alpha1+4*c1+2*f1][g_iup[ix][TUP]].sp_up.s1 );

              upm = &g_gauge_field[g_idn[ix][TUP]][TUP];

              _su3_multiply( running.sp_up.s0, (*upm), tmp.sp_up.s0 );
              _su3_multiply( running.sp_up.s1, (*upm), tmp.sp_up.s1 );


              _complex_times_vector(running.sp_up.s0,phase_00,running.sp_up.s0);
              _complex_times_vector(running.sp_up.s1,phase_00,running.sp_up.s1);


              c[ix]+= running.sp_up.s0.c0*conj(prop[12*alpha1+4*c1+2*f3+1][g_idn[ix][TUP]].sp_dn.s0.c0)
                     +running.sp_up.s0.c1*conj(prop[12*alpha1+4*c1+2*f3+1][g_idn[ix][TUP]].sp_dn.s0.c1)
                     +running.sp_up.s0.c2*conj(prop[12*alpha1+4*c1+2*f3+1][g_idn[ix][TUP]].sp_dn.s0.c2)
                     +running.sp_up.s1.c0*conj(prop[12*alpha1+4*c1+2*f3+1][g_idn[ix][TUP]].sp_dn.s1.c0)
                     +running.sp_up.s1.c1*conj(prop[12*alpha1+4*c1+2*f3+1][g_idn[ix][TUP]].sp_dn.s1.c1)
                     +running.sp_up.s1.c2*conj(prop[12*alpha1+4*c1+2*f3+1][g_idn[ix][TUP]].sp_dn.s1.c2);
          }
          if ( (f6 == 1) && (f4==1) ){
              upm = &g_gauge_field[ix][TUP];

              _vector_null( tmp.sp_dn.s2 );
              _vector_null( tmp.sp_dn.s3 );
              _vector_null( tmp.sp_dn.s0 );
              _vector_null( tmp.sp_dn.s1 );

              _vector_null( running.sp_dn.s0 );
              _vector_null( running.sp_dn.s1 );
              _vector_null( running.sp_dn.s2 );
              _vector_null( running.sp_dn.s3 );


              _su3_multiply( tmp.sp_dn.s0, (*upm), prop[12*alpha1+4*c1+2*f1][g_iup[ix][TUP]].sp_dn.s0 );
              _su3_multiply( tmp.sp_dn.s1, (*upm), prop[12*alpha1+4*c1+2*f1][g_iup[ix][TUP]].sp_dn.s1 );

              upm = &g_gauge_field[g_idn[ix][TUP]][TUP];

              _su3_multiply( running.sp_dn.s0, (*upm), tmp.sp_dn.s0 );
              _su3_multiply( running.sp_dn.s1, (*upm), tmp.sp_dn.s1 );


              _complex_times_vector(running.sp_dn.s0,phase_00,running.sp_dn.s0);
              _complex_times_vector(running.sp_dn.s1,phase_00,running.sp_dn.s1);
               
              c[ix]+= running.sp_dn.s0.c0*conj(prop[12*alpha1+4*c1+2*f3+1][g_idn[ix][TUP]].sp_dn.s0.c0)
                     +running.sp_dn.s0.c1*conj(prop[12*alpha1+4*c1+2*f3+1][g_idn[ix][TUP]].sp_dn.s0.c1)
                     +running.sp_dn.s0.c2*conj(prop[12*alpha1+4*c1+2*f3+1][g_idn[ix][TUP]].sp_dn.s0.c2)
                     +running.sp_dn.s1.c0*conj(prop[12*alpha1+4*c1+2*f3+1][g_idn[ix][TUP]].sp_dn.s1.c0)
                     +running.sp_dn.s1.c1*conj(prop[12*alpha1+4*c1+2*f3+1][g_idn[ix][TUP]].sp_dn.s1.c1)
                     +running.sp_dn.s1.c2*conj(prop[12*alpha1+4*c1+2*f3+1][g_idn[ix][TUP]].sp_dn.s1.c2);

          }
       }
}

void wilsoncurrent31a_petros( bispinor **propfields )
{

    _Complex double **phimatrix=(_Complex double **)malloc(sizeof(_Complex double *)*4);

    _Complex double *C0000=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    _Complex double *C0001=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    _Complex double *C0010=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    _Complex double *C0011=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    _Complex double *C0100=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    _Complex double *C0101=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    _Complex double *C0110=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    _Complex double *C0111=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    _Complex double *C1000=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    _Complex double *C1001=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    _Complex double *C1010=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    _Complex double *C1011=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    _Complex double *C1100=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    _Complex double *C1101=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    _Complex double *C1110=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    _Complex double *C1111=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);

    _Complex double *final_corr=(_Complex double *)malloc(sizeof(_Complex double)*T_global);

    _Complex double *phimatrixspatialnull=(_Complex double *)malloc(sizeof(_Complex double)*4);

    int ix;

// Doing the neccessary communication
#if defined TM_USE_MPI
   int s1,c1,f1;
   int count;
   MPI_Status  statuses[8];
   MPI_Request *request;
   request=( MPI_Request *) malloc(sizeof(MPI_Request)*8);
   for (s1=0; s1<2; ++s1)
      for (c1=0; c1<3; ++c1)
         for (f1=0; f1<2; ++f1){
            count=0;
            generic_exchange_direction_nonblocking( propfields[12*s1 + 4*c1 + 2*f1 + 0], sizeof(bispinor), TUP   , request, &count );
            MPI_Waitall( count, request, statuses);
            count=0;
            generic_exchange_direction_nonblocking( propfields[12*s1 + 4*c1 + 2*f1 + 0], sizeof(bispinor), TDOWN , request, &count );
            MPI_Waitall( count, request, statuses);
            count=0;
            generic_exchange_direction_nonblocking( propfields[12*s1 + 4*c1 + 2*f1 + 1], sizeof(bispinor), TDOWN , request, &count );
            MPI_Waitall( count, request, statuses);
         }
   free(request);
#endif

    for (ix=0;ix<4;++ix)
       phimatrix[ix]=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    for (ix=0;ix<VOLUME;++ix)
    {
       if (smearedcorrelator_BSM == 1){
         phimatrix[0][ix]= 1.*g_smeared_scalar_field[0][ix] + I*g_smeared_scalar_field[3][ix];
         phimatrix[1][ix]= 1.*g_smeared_scalar_field[2][ix] + I*g_smeared_scalar_field[1][ix];
         phimatrix[2][ix]=-1.*g_smeared_scalar_field[2][ix] + I*g_smeared_scalar_field[1][ix];
         phimatrix[3][ix]= 1.*g_smeared_scalar_field[0][ix] - I*g_smeared_scalar_field[3][ix];
       }
       else{
         phimatrix[0][ix]= 1.*g_scalar_field[0][ix] + I*g_scalar_field[3][ix];
         phimatrix[1][ix]= 1.*g_scalar_field[2][ix] + I*g_scalar_field[1][ix];
         phimatrix[2][ix]=-1.*g_scalar_field[2][ix] + I*g_scalar_field[1][ix];
         phimatrix[3][ix]= 1.*g_scalar_field[0][ix] - I*g_scalar_field[3][ix];
       }
    }

    for (ix=0;ix<4;++ix)
       phimatrixspatialnull[ix]=phimatrix[ix][0];

#if defined TM_USE_MPI
    for (ix=0;ix<4;++ix)
       MPI_Bcast(&phimatrixspatialnull[ix], 1, MPI_DOUBLE_COMPLEX, 0, g_cart_grid);
#endif

    for (ix=0; ix<VOLUME; ++ix){
       trace_in_spinor_and_color3a(C0000,propfields,ix,0,0,0,0);
       trace_in_spinor_and_color3a(C0001,propfields,ix,0,0,0,1);
       trace_in_spinor_and_color3a(C0010,propfields,ix,0,0,1,0);
       trace_in_spinor_and_color3a(C0011,propfields,ix,0,0,1,1);
       trace_in_spinor_and_color3a(C0100,propfields,ix,0,1,0,0);
       trace_in_spinor_and_color3a(C0101,propfields,ix,0,1,0,1);
       trace_in_spinor_and_color3a(C0110,propfields,ix,0,1,1,0);
       trace_in_spinor_and_color3a(C0111,propfields,ix,0,1,1,1);
       trace_in_spinor_and_color3a(C1000,propfields,ix,1,0,0,0);
       trace_in_spinor_and_color3a(C1001,propfields,ix,1,0,0,1);
       trace_in_spinor_and_color3a(C1010,propfields,ix,1,0,1,0);
       trace_in_spinor_and_color3a(C1011,propfields,ix,1,0,1,1);
       trace_in_spinor_and_color3a(C1100,propfields,ix,1,1,0,0);
       trace_in_spinor_and_color3a(C1101,propfields,ix,1,1,0,1);
       trace_in_spinor_and_color3a(C1110,propfields,ix,1,1,1,0);
       trace_in_spinor_and_color3a(C1111,propfields,ix,1,1,1,1);

    }
    for (ix=0; ix<T_global; ++ix)
       final_corr[ix]=0.; 
    for (ix=0; ix<VOLUME; ++ix){

//tau_1
       final_corr[g_coord[ix][TUP]]+=  1.*phimatrixspatialnull[1*2+0]*C0000[ix]*phimatrix[1*2+0][ix]
                                     + 1.*phimatrixspatialnull[1*2+0]*C0010[ix]*phimatrix[1*2+1][ix]
                                     + 1.*phimatrixspatialnull[1*2+0]*C0100[ix]*phimatrix[0*2+0][ix]
                                     + 1.*phimatrixspatialnull[1*2+0]*C0110[ix]*phimatrix[0*2+1][ix]

                                     + 1.*phimatrixspatialnull[1*2+1]*C1000[ix]*phimatrix[1*2+0][ix]
                                     + 1.*phimatrixspatialnull[1*2+1]*C1010[ix]*phimatrix[1*2+1][ix]
                                     + 1.*phimatrixspatialnull[1*2+1]*C1100[ix]*phimatrix[0*2+0][ix]
                                     + 1.*phimatrixspatialnull[1*2+1]*C1110[ix]*phimatrix[0*2+1][ix]

                                     + 1.*phimatrixspatialnull[0*2+0]*C0001[ix]*phimatrix[1*2+0][ix]
                                     + 1.*phimatrixspatialnull[0*2+0]*C0011[ix]*phimatrix[1*2+1][ix]
                                     + 1.*phimatrixspatialnull[0*2+0]*C0101[ix]*phimatrix[0*2+0][ix]
                                     + 1.*phimatrixspatialnull[0*2+0]*C0111[ix]*phimatrix[0*2+1][ix]

                                     + 1.*phimatrixspatialnull[0*2+1]*C1001[ix]*phimatrix[1*2+0][ix]
                                     + 1.*phimatrixspatialnull[0*2+1]*C1011[ix]*phimatrix[1*2+1][ix]
                                     + 1.*phimatrixspatialnull[0*2+1]*C1101[ix]*phimatrix[0*2+0][ix]
                                     + 1.*phimatrixspatialnull[0*2+1]*C1111[ix]*phimatrix[0*2+1][ix];

//tau2
       final_corr[g_coord[ix][TUP]]+= -1.*phimatrixspatialnull[1*2+0]*C0000[ix]*phimatrix[1*2+0][ix]
                                     +-1.*phimatrixspatialnull[1*2+0]*C0010[ix]*phimatrix[1*2+1][ix]
                                     + 1.*phimatrixspatialnull[1*2+0]*C0100[ix]*phimatrix[0*2+0][ix]
                                     + 1.*phimatrixspatialnull[1*2+0]*C0110[ix]*phimatrix[0*2+1][ix]

                                     +-1.*phimatrixspatialnull[1*2+1]*C1000[ix]*phimatrix[1*2+0][ix]
                                     +-1.*phimatrixspatialnull[1*2+1]*C1010[ix]*phimatrix[1*2+1][ix]
                                     + 1.*phimatrixspatialnull[1*2+1]*C1100[ix]*phimatrix[0*2+0][ix]
                                     + 1.*phimatrixspatialnull[1*2+1]*C1110[ix]*phimatrix[0*2+1][ix]

                                     + 1.*phimatrixspatialnull[0*2+0]*C0001[ix]*phimatrix[1*2+0][ix]
                                     + 1.*phimatrixspatialnull[0*2+0]*C0011[ix]*phimatrix[1*2+1][ix]
                                     +-1.*phimatrixspatialnull[0*2+0]*C0101[ix]*phimatrix[0*2+0][ix]
                                     +-1.*phimatrixspatialnull[0*2+0]*C0111[ix]*phimatrix[0*2+1][ix]

                                     + 1.*phimatrixspatialnull[0*2+1]*C1001[ix]*phimatrix[1*2+0][ix]
                                     + 1.*phimatrixspatialnull[0*2+1]*C1011[ix]*phimatrix[1*2+1][ix]
                                     +-1.*phimatrixspatialnull[0*2+1]*C1101[ix]*phimatrix[0*2+0][ix]
                                     +-1.*phimatrixspatialnull[0*2+1]*C1111[ix]*phimatrix[0*2+1][ix];
//tau3
       final_corr[g_coord[ix][TUP]]+=  1.*phimatrixspatialnull[0*2+0]*C0000[ix]*phimatrix[0*2+0][ix]
                                     + 1.*phimatrixspatialnull[0*2+0]*C0010[ix]*phimatrix[0*2+1][ix]
                                     +-1.*phimatrixspatialnull[0*2+0]*C0100[ix]*phimatrix[1*2+0][ix]
                                     +-1.*phimatrixspatialnull[0*2+0]*C0110[ix]*phimatrix[1*2+1][ix]

                                     + 1.*phimatrixspatialnull[0*2+1]*C1000[ix]*phimatrix[0*2+0][ix]
                                     + 1.*phimatrixspatialnull[0*2+1]*C1010[ix]*phimatrix[0*2+1][ix]
                                     +-1.*phimatrixspatialnull[0*2+1]*C1100[ix]*phimatrix[1*2+0][ix]
                                     +-1.*phimatrixspatialnull[0*2+1]*C1110[ix]*phimatrix[1*2+1][ix]

                                     +-1.*phimatrixspatialnull[1*2+0]*C0001[ix]*phimatrix[0*2+0][ix]
                                     +-1.*phimatrixspatialnull[1*2+0]*C0011[ix]*phimatrix[0*2+1][ix]
                                     + 1.*phimatrixspatialnull[1*2+0]*C0101[ix]*phimatrix[1*2+0][ix]
                                     + 1.*phimatrixspatialnull[1*2+0]*C0111[ix]*phimatrix[1*2+1][ix]

                                     +-1.*phimatrixspatialnull[1*2+1]*C1001[ix]*phimatrix[0*2+0][ix]
                                     +-1.*phimatrixspatialnull[1*2+1]*C1011[ix]*phimatrix[0*2+1][ix]
                                     + 1.*phimatrixspatialnull[1*2+1]*C1101[ix]*phimatrix[1*2+0][ix]
                                     + 1.*phimatrixspatialnull[1*2+1]*C1111[ix]*phimatrix[1*2+1][ix];

    }
#if defined TM_USE_MPI
    for (ix=0; ix<T_global; ++ix){
       _Complex double tmp;
       MPI_Allreduce(&final_corr[ix], &tmp, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
       final_corr[ix]= tmp;
    }
#endif 
    if (g_cart_id == 0){printf("Wilson current  Density correlator type a la Petros (1) results\n");}
      for (ix=0; ix<T_global; ++ix){
        if (g_cart_id == 0){
        printf("WCDPR1 0 0 %.3d %10.10e %10.10e\n", ix, creal(final_corr[ix])/4.,cimag(final_corr[ix])/4.);
      }
    }


    free(C0000);
    free(C0001);
    free(C0010);
    free(C0011);
    free(C0100);
    free(C0101);
    free(C0110);
    free(C0111);
    free(C1000);
    free(C1001);
    free(C1010);
    free(C1011);
    free(C1100);
    free(C1101);
    free(C1110);
    free(C1111);

    for (ix=0;ix<4;++ix)
       free(phimatrix[ix]);
    free(phimatrix);
    free(final_corr);

}


void density_density_1234_petros( bispinor **propfields )
{

    _Complex double **phimatrix=(_Complex double **)malloc(sizeof(_Complex double *)*4);

    _Complex double *C0000=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    _Complex double *C0001=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    _Complex double *C0010=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    _Complex double *C0011=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    _Complex double *C0100=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    _Complex double *C0101=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    _Complex double *C0110=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    _Complex double *C0111=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    _Complex double *C1000=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    _Complex double *C1001=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    _Complex double *C1010=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    _Complex double *C1011=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    _Complex double *C1100=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    _Complex double *C1101=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    _Complex double *C1110=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    _Complex double *C1111=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);

    _Complex double *final_corr=(_Complex double *)malloc(sizeof(_Complex double)*T_global);

    _Complex double *phimatrixspatialnull=(_Complex double *)malloc(sizeof(_Complex double)*4);

    int ix;


    for (ix=0;ix<4;++ix)
       phimatrix[ix]=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    for (ix=0;ix<VOLUME;++ix)
    {
       if (smearedcorrelator_BSM == 1){
         phimatrix[0][ix]= 1.*g_smeared_scalar_field[0][ix] + I*g_smeared_scalar_field[3][ix];
         phimatrix[1][ix]= 1.*g_smeared_scalar_field[2][ix] + I*g_smeared_scalar_field[1][ix];
         phimatrix[2][ix]=-1.*g_smeared_scalar_field[2][ix] + I*g_smeared_scalar_field[1][ix];
         phimatrix[3][ix]= 1.*g_smeared_scalar_field[0][ix] - I*g_smeared_scalar_field[3][ix];
       }
       else{
         phimatrix[0][ix]= 1.*g_scalar_field[0][ix] + I*g_scalar_field[3][ix];
         phimatrix[1][ix]= 1.*g_scalar_field[2][ix] + I*g_scalar_field[1][ix];
         phimatrix[2][ix]=-1.*g_scalar_field[2][ix] + I*g_scalar_field[1][ix];
         phimatrix[3][ix]= 1.*g_scalar_field[0][ix] - I*g_scalar_field[3][ix];
       }
    }

    for (ix=0;ix<4;++ix)
       phimatrixspatialnull[ix]=phimatrix[ix][0];

#if defined TM_USE_MPI
    for (ix=0;ix<4;++ix){
       MPI_Bcast(&phimatrixspatialnull[ix], 1, MPI_DOUBLE_COMPLEX, 0, g_cart_grid);
    }
#endif
    
    for (ix=0; ix<VOLUME; ++ix){
       trace_in_spinor_and_color(C0000,propfields,ix,0,0,0,0);
       trace_in_spinor_and_color(C0001,propfields,ix,0,0,0,1);
       trace_in_spinor_and_color(C0010,propfields,ix,0,0,1,0);
       trace_in_spinor_and_color(C0011,propfields,ix,0,0,1,1);
       trace_in_spinor_and_color(C0100,propfields,ix,0,1,0,0);
       trace_in_spinor_and_color(C0101,propfields,ix,0,1,0,1);
       trace_in_spinor_and_color(C0110,propfields,ix,0,1,1,0);
       trace_in_spinor_and_color(C0111,propfields,ix,0,1,1,1);
       trace_in_spinor_and_color(C1000,propfields,ix,1,0,0,0);
       trace_in_spinor_and_color(C1001,propfields,ix,1,0,0,1);
       trace_in_spinor_and_color(C1010,propfields,ix,1,0,1,0);
       trace_in_spinor_and_color(C1011,propfields,ix,1,0,1,1);
       trace_in_spinor_and_color(C1100,propfields,ix,1,1,0,0);
       trace_in_spinor_and_color(C1101,propfields,ix,1,1,0,1);
       trace_in_spinor_and_color(C1110,propfields,ix,1,1,1,0);
       trace_in_spinor_and_color(C1111,propfields,ix,1,1,1,1);

    }
    for (ix=0; ix<T_global; ++ix)
       final_corr[ix]=0.;
    for (ix=0; ix<VOLUME; ++ix){

//tau_1
       final_corr[g_coord[ix][TUP]]+=  1.*phimatrixspatialnull[1*2+0]*C0010[ix]*conj(phimatrix[0*2+0][ix])
                                     + 1.*phimatrixspatialnull[1*2+0]*C0000[ix]*conj(phimatrix[1*2+0][ix])
                                     + 1.*phimatrixspatialnull[1*2+0]*C0110[ix]*conj(phimatrix[0*2+1][ix])
                                     + 1.*phimatrixspatialnull[1*2+0]*C0100[ix]*conj(phimatrix[1*2+1][ix])

                                     + 1.*phimatrixspatialnull[1*2+1]*C1010[ix]*conj(phimatrix[0*2+0][ix])
                                     + 1.*phimatrixspatialnull[1*2+1]*C1000[ix]*conj(phimatrix[1*2+0][ix])
                                     + 1.*phimatrixspatialnull[1*2+1]*C1110[ix]*conj(phimatrix[0*2+1][ix])
                                     + 1.*phimatrixspatialnull[1*2+1]*C1100[ix]*conj(phimatrix[1*2+1][ix])

                                     + 1.*phimatrixspatialnull[0*2+0]*C0011[ix]*conj(phimatrix[0*2+0][ix])
                                     + 1.*phimatrixspatialnull[0*2+0]*C0001[ix]*conj(phimatrix[1*2+0][ix])
                                     + 1.*phimatrixspatialnull[0*2+0]*C0111[ix]*conj(phimatrix[0*2+1][ix])
                                     + 1.*phimatrixspatialnull[0*2+0]*C0101[ix]*conj(phimatrix[1*2+1][ix])

                                     + 1.*phimatrixspatialnull[0*2+1]*C1011[ix]*conj(phimatrix[0*2+0][ix])
                                     + 1.*phimatrixspatialnull[0*2+1]*C1001[ix]*conj(phimatrix[1*2+0][ix])
                                     + 1.*phimatrixspatialnull[0*2+1]*C1111[ix]*conj(phimatrix[0*2+1][ix])
                                     + 1.*phimatrixspatialnull[0*2+1]*C1101[ix]*conj(phimatrix[1*2+1][ix]);

//tau_2
       final_corr[g_coord[ix][TUP]]+= -1.*phimatrixspatialnull[1*2+0]*C0010[ix]*conj(phimatrix[0*2+0][ix])
                                     + 1.*phimatrixspatialnull[1*2+0]*C0000[ix]*conj(phimatrix[1*2+0][ix])
                                     +-1.*phimatrixspatialnull[1*2+0]*C0110[ix]*conj(phimatrix[0*2+1][ix])
                                     + 1.*phimatrixspatialnull[1*2+0]*C0100[ix]*conj(phimatrix[1*2+1][ix])

                                     +-1.*phimatrixspatialnull[1*2+1]*C1010[ix]*conj(phimatrix[0*2+0][ix])
                                     + 1.*phimatrixspatialnull[1*2+1]*C1000[ix]*conj(phimatrix[1*2+0][ix])
                                     +-1.*phimatrixspatialnull[1*2+1]*C1110[ix]*conj(phimatrix[0*2+1][ix])
                                     + 1.*phimatrixspatialnull[1*2+1]*C1100[ix]*conj(phimatrix[1*2+1][ix])

                                     + 1.*phimatrixspatialnull[0*2+0]*C0011[ix]*conj(phimatrix[0*2+0][ix])
                                     +-1.*phimatrixspatialnull[0*2+0]*C0001[ix]*conj(phimatrix[1*2+0][ix])
                                     + 1.*phimatrixspatialnull[0*2+0]*C0111[ix]*conj(phimatrix[0*2+1][ix])
                                     +-1.*phimatrixspatialnull[0*2+0]*C0101[ix]*conj(phimatrix[1*2+1][ix])

                                     + 1.*phimatrixspatialnull[0*2+1]*C1011[ix]*conj(phimatrix[0*2+0][ix])
                                     +-1.*phimatrixspatialnull[0*2+1]*C1001[ix]*conj(phimatrix[1*2+0][ix])
                                     + 1.*phimatrixspatialnull[0*2+1]*C1111[ix]*conj(phimatrix[0*2+1][ix])
                                     +-1.*phimatrixspatialnull[0*2+1]*C1101[ix]*conj(phimatrix[1*2+1][ix]);

//tau3
       final_corr[g_coord[ix][TUP]]+=  1.*phimatrixspatialnull[0*2+0]*C0000[ix]*conj(phimatrix[0*2+0][ix])
                                     +-1.*phimatrixspatialnull[0*2+0]*C0010[ix]*conj(phimatrix[1*2+0][ix])
                                     + 1.*phimatrixspatialnull[0*2+0]*C0100[ix]*conj(phimatrix[0*2+1][ix])
                                     +-1.*phimatrixspatialnull[0*2+0]*C0110[ix]*conj(phimatrix[1*2+1][ix])

                                     + 1.*phimatrixspatialnull[0*2+1]*C1000[ix]*conj(phimatrix[0*2+0][ix])
                                     +-1.*phimatrixspatialnull[0*2+1]*C1010[ix]*conj(phimatrix[1*2+0][ix])
                                     + 1.*phimatrixspatialnull[0*2+1]*C1100[ix]*conj(phimatrix[0*2+1][ix])
                                     +-1.*phimatrixspatialnull[0*2+1]*C1110[ix]*conj(phimatrix[1*2+1][ix])

                                     +-1.*phimatrixspatialnull[1*2+0]*C0001[ix]*conj(phimatrix[0*2+0][ix])
                                     + 1.*phimatrixspatialnull[1*2+0]*C0011[ix]*conj(phimatrix[1*2+0][ix])
                                     +-1.*phimatrixspatialnull[1*2+0]*C0101[ix]*conj(phimatrix[0*2+1][ix])
                                     + 1.*phimatrixspatialnull[1*2+0]*C0111[ix]*conj(phimatrix[1*2+1][ix])

                                     +-1.*phimatrixspatialnull[1*2+1]*C1001[ix]*conj(phimatrix[0*2+0][ix])
                                     + 1.*phimatrixspatialnull[1*2+1]*C1011[ix]*conj(phimatrix[1*2+0][ix])
                                     +-1.*phimatrixspatialnull[1*2+1]*C1101[ix]*conj(phimatrix[0*2+1][ix])
                                     + 1.*phimatrixspatialnull[1*2+1]*C1111[ix]*conj(phimatrix[1*2+1][ix]);

    }
#if defined TM_USE_MPI
    for (ix=0; ix<T_global; ++ix){
       _Complex double tmp;
       MPI_Allreduce(&final_corr[ix], &tmp, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
       final_corr[ix]= tmp;
    }
#endif
    if (g_cart_id == 0){printf("Density Density correlator type a la Petros (1) results\n");}
      for (ix=0; ix<T_global; ++ix){
        if (g_cart_id == 0){
        printf("DD 1 %.3d %10.10e %10.10e\n", ix, creal(final_corr[ix])/4.,cimag(final_corr[ix])/4.);
      }
    }


    free(C0000);
    free(C0001);
    free(C0010);
    free(C0011);
    free(C0100);
    free(C0101);
    free(C0110);
    free(C0111);
    free(C1000);
    free(C1001);
    free(C1010);
    free(C1011);
    free(C1100);
    free(C1101);
    free(C1110);
    free(C1111);

    for (ix=0;ix<4;++ix)
       free(phimatrix[ix]);
    free(phimatrix);
    free(final_corr);

}



void diraccurrent1a_petros( bispinor **propfields )
{

    _Complex double **phimatrix=(_Complex double **)malloc(sizeof(_Complex double *)*4);

    _Complex double *C0000=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    _Complex double *C0001=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    _Complex double *C0010=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    _Complex double *C0011=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    _Complex double *C0100=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    _Complex double *C0101=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    _Complex double *C0110=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    _Complex double *C0111=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    _Complex double *C1000=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    _Complex double *C1001=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    _Complex double *C1010=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    _Complex double *C1011=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    _Complex double *C1100=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    _Complex double *C1101=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    _Complex double *C1110=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    _Complex double *C1111=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);

    _Complex double *final_corr=(_Complex double *)malloc(sizeof(_Complex double)*T_global);

    _Complex double *phimatrixspatialnull=(_Complex double *)malloc(sizeof(_Complex double)*4);

    int ix;


    for (ix=0;ix<4;++ix)
       phimatrix[ix]=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    for (ix=0;ix<VOLUME;++ix)
    {
       if (smearedcorrelator_BSM == 1){
         phimatrix[0][ix]= 1.*g_smeared_scalar_field[0][ix] + I*g_smeared_scalar_field[3][ix];
         phimatrix[1][ix]= 1.*g_smeared_scalar_field[2][ix] + I*g_smeared_scalar_field[1][ix];
         phimatrix[2][ix]=-1.*g_smeared_scalar_field[2][ix] + I*g_smeared_scalar_field[1][ix];
         phimatrix[3][ix]= 1.*g_smeared_scalar_field[0][ix] - I*g_smeared_scalar_field[3][ix];
       }
       else{
         phimatrix[0][ix]= 1.*g_scalar_field[0][ix] + I*g_scalar_field[3][ix];
         phimatrix[1][ix]= 1.*g_scalar_field[2][ix] + I*g_scalar_field[1][ix];
         phimatrix[2][ix]=-1.*g_scalar_field[2][ix] + I*g_scalar_field[1][ix];
         phimatrix[3][ix]= 1.*g_scalar_field[0][ix] - I*g_scalar_field[3][ix];
       }
    }

    for (ix=0;ix<4;++ix)
       phimatrixspatialnull[ix]=phimatrix[ix][0];

#if defined TM_USE_MPI
    for (ix=0;ix<4;++ix){
       MPI_Bcast(&phimatrixspatialnull[ix], 1, MPI_DOUBLE_COMPLEX, 0, g_cart_grid);
    }
#endif

    for (ix=0; ix<VOLUME; ++ix){
       trace_in_spinor_and_color1a(C0000,propfields,ix,0,0,0,0);
       trace_in_spinor_and_color1a(C0001,propfields,ix,0,0,0,1);
       trace_in_spinor_and_color1a(C0010,propfields,ix,0,0,1,0);
       trace_in_spinor_and_color1a(C0011,propfields,ix,0,0,1,1);
       trace_in_spinor_and_color1a(C0100,propfields,ix,0,1,0,0);
       trace_in_spinor_and_color1a(C0101,propfields,ix,0,1,0,1);
       trace_in_spinor_and_color1a(C0110,propfields,ix,0,1,1,0);
       trace_in_spinor_and_color1a(C0111,propfields,ix,0,1,1,1);
       trace_in_spinor_and_color1a(C1000,propfields,ix,1,0,0,0);
       trace_in_spinor_and_color1a(C1001,propfields,ix,1,0,0,1);
       trace_in_spinor_and_color1a(C1010,propfields,ix,1,0,1,0);
       trace_in_spinor_and_color1a(C1011,propfields,ix,1,0,1,1);
       trace_in_spinor_and_color1a(C1100,propfields,ix,1,1,0,0);
       trace_in_spinor_and_color1a(C1101,propfields,ix,1,1,0,1);
       trace_in_spinor_and_color1a(C1110,propfields,ix,1,1,1,0);
       trace_in_spinor_and_color1a(C1111,propfields,ix,1,1,1,1);
    }
    for (ix=0; ix<T_global; ++ix)
       final_corr[ix]=0.; 
    for (ix=0; ix<VOLUME; ++ix){

//tau_1
       final_corr[g_coord[ix][TUP]]+=  1.*phimatrixspatialnull[1*2+0]*C0010[ix]
                                     + 1.*phimatrixspatialnull[1*2+0]*C0100[ix]
                                     + 1.*phimatrixspatialnull[1*2+1]*C1010[ix]
                                     + 1.*phimatrixspatialnull[1*2+1]*C1100[ix]

                                     + 1.*phimatrixspatialnull[0*2+0]*C0011[ix]
                                     + 1.*phimatrixspatialnull[0*2+0]*C0101[ix]
                                     + 1.*phimatrixspatialnull[0*2+1]*C1011[ix]
                                     + 1.*phimatrixspatialnull[0*2+1]*C1101[ix];

//tau_2
       final_corr[g_coord[ix][TUP]]+= -1.*phimatrixspatialnull[1*2+0]*C0010[ix]
                                     + 1.*phimatrixspatialnull[1*2+0]*C0100[ix]
                                     +-1.*phimatrixspatialnull[1*2+1]*C1010[ix]
                                     + 1.*phimatrixspatialnull[1*2+1]*C1100[ix]

                                     + 1.*phimatrixspatialnull[0*2+0]*C0011[ix]
                                     +-1.*phimatrixspatialnull[0*2+0]*C0101[ix]
                                     + 1.*phimatrixspatialnull[0*2+1]*C1011[ix]
                                     +-1.*phimatrixspatialnull[0*2+1]*C1101[ix];
//tau_3
       final_corr[g_coord[ix][TUP]]+=  1.*phimatrixspatialnull[0*2+0]*C0000[ix]
                                     +-1.*phimatrixspatialnull[0*2+0]*C0110[ix]
                                     + 1.*phimatrixspatialnull[0*2+1]*C1000[ix]
                                     +-1.*phimatrixspatialnull[0*2+1]*C1110[ix]

                                     +-1.*phimatrixspatialnull[1*2+0]*C0001[ix]
                                     + 1.*phimatrixspatialnull[1*2+0]*C0111[ix]
                                     +-1.*phimatrixspatialnull[1*2+1]*C1001[ix]
                                     + 1.*phimatrixspatialnull[1*2+1]*C1111[ix];

    }
#if defined TM_USE_MPI
    for (ix=0; ix<T_global; ++ix){
       _Complex double tmp;
       MPI_Allreduce(&final_corr[ix], &tmp, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
       final_corr[ix]= tmp;
    }
#endif 
    if (g_cart_id == 0){printf("Dirac Current Density correlator type a la Petros (1) results\n");}
      for (ix=0; ix<T_global; ++ix){
        if (g_cart_id == 0){
        printf("DCD 0 0  %.3d %10.10e %10.10e\n", ix, creal(final_corr[ix])/4.,cimag(final_corr[ix])/4.);
      }
    }


    free(C0000);
    free(C0001);
    free(C0010);
    free(C0011);
    free(C0100);
    free(C0101);
    free(C0110);
    free(C0111);
    free(C1000);
    free(C1001);
    free(C1010);
    free(C1011);
    free(C1100);
    free(C1101);
    free(C1110);
    free(C1111);

    for (ix=0;ix<4;++ix)
       free(phimatrix[ix]);
    free(phimatrix);
    free(final_corr);

}

void wilsoncurrent61a_petros( bispinor **propfields )
{
#if defined TM_USE_MPI
    int count;
    MPI_Status  statuses[8];
    MPI_Request *request;
    request=( MPI_Request *) malloc(sizeof(MPI_Request)*8);
#endif

    _Complex double **phimatrix=(_Complex double **)malloc(sizeof(_Complex double *)*4);

    _Complex double *C0000=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    _Complex double *C0001=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    _Complex double *C0010=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    _Complex double *C0011=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    _Complex double *C0100=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    _Complex double *C0101=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    _Complex double *C0110=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    _Complex double *C0111=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    _Complex double *C1000=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    _Complex double *C1001=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    _Complex double *C1010=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    _Complex double *C1011=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    _Complex double *C1100=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    _Complex double *C1101=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    _Complex double *C1110=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);
    _Complex double *C1111=(_Complex double *)malloc(sizeof(_Complex double)*VOLUME);

    _Complex double *final_corr=(_Complex double *)malloc(sizeof(_Complex double)*T_global);

    _Complex double *phimatrixspatialnull=(_Complex double *)malloc(sizeof(_Complex double)*4);

    int ix;


    for (ix=0;ix<4;++ix)
       phimatrix[ix]=(_Complex double *)malloc(sizeof(_Complex double)*VOLUMEPLUSRAND);
    for (ix=0;ix<VOLUME;++ix)
    {
       if (smearedcorrelator_BSM == 1){
         phimatrix[0][ix]= 1.*g_smeared_scalar_field[0][ix] + I*g_smeared_scalar_field[3][ix];
         phimatrix[1][ix]= 1.*g_smeared_scalar_field[2][ix] + I*g_smeared_scalar_field[1][ix];
         phimatrix[2][ix]=-1.*g_smeared_scalar_field[2][ix] + I*g_smeared_scalar_field[1][ix];
         phimatrix[3][ix]= 1.*g_smeared_scalar_field[0][ix] - I*g_smeared_scalar_field[3][ix];
       }
       else{
         phimatrix[0][ix]= 1.*g_scalar_field[0][ix] + I*g_scalar_field[3][ix];
         phimatrix[1][ix]= 1.*g_scalar_field[2][ix] + I*g_scalar_field[1][ix];
         phimatrix[2][ix]=-1.*g_scalar_field[2][ix] + I*g_scalar_field[1][ix];
         phimatrix[3][ix]= 1.*g_scalar_field[0][ix] - I*g_scalar_field[3][ix];
       }
    }

    for (ix=0;ix<4;++ix)
       phimatrixspatialnull[ix]=phimatrix[ix][0];

#if defined TM_USE_MPI
    for (ix=0; ix<4; ++ix){
      count=0;
      generic_exchange_direction_nonblocking( phimatrix[ix], sizeof(_Complex double), TDOWN   , request, &count );
      MPI_Waitall( count, request, statuses);
      count=0;
    }
    for (ix=0;ix<4;++ix){
       MPI_Bcast(&phimatrixspatialnull[ix], 1, MPI_DOUBLE_COMPLEX, 0, g_cart_grid);
    }
#endif

    for (ix=0; ix<VOLUME; ++ix){
       trace_in_spinor_and_color(C0000,propfields,ix,0,0,0,0);
       trace_in_spinor_and_color(C0001,propfields,ix,0,0,0,1);
       trace_in_spinor_and_color(C0010,propfields,ix,0,0,1,0);
       trace_in_spinor_and_color(C0011,propfields,ix,0,0,1,1);
       trace_in_spinor_and_color(C0100,propfields,ix,0,1,0,0);
       trace_in_spinor_and_color(C0101,propfields,ix,0,1,0,1);
       trace_in_spinor_and_color(C0110,propfields,ix,0,1,1,0);
       trace_in_spinor_and_color(C0111,propfields,ix,0,1,1,1);
       trace_in_spinor_and_color(C1000,propfields,ix,1,0,0,0);
       trace_in_spinor_and_color(C1001,propfields,ix,1,0,0,1);
       trace_in_spinor_and_color(C1010,propfields,ix,1,0,1,0);
       trace_in_spinor_and_color(C1011,propfields,ix,1,0,1,1);
       trace_in_spinor_and_color(C1100,propfields,ix,1,1,0,0);
       trace_in_spinor_and_color(C1101,propfields,ix,1,1,0,1);
       trace_in_spinor_and_color(C1110,propfields,ix,1,1,1,0);
       trace_in_spinor_and_color(C1111,propfields,ix,1,1,1,1);

    }
    for (ix=0; ix<T_global; ++ix)
       final_corr[ix]=0.;
    for (ix=0; ix<VOLUME; ++ix){

//tau_1
       final_corr[g_coord[ix][TUP]]+=  1.*phimatrixspatialnull[1*2+0]*C0010[ix]*conj(phimatrix[0*2+0][g_idn[ix][TUP]])
                                     + 1.*phimatrixspatialnull[1*2+0]*C0000[ix]*conj(phimatrix[1*2+0][g_idn[ix][TUP]])
                                     + 1.*phimatrixspatialnull[1*2+0]*C0110[ix]*conj(phimatrix[0*2+1][g_idn[ix][TUP]])
                                     + 1.*phimatrixspatialnull[1*2+0]*C0100[ix]*conj(phimatrix[1*2+1][g_idn[ix][TUP]])

                                     + 1.*phimatrixspatialnull[1*2+1]*C1010[ix]*conj(phimatrix[0*2+0][g_idn[ix][TUP]])
                                     + 1.*phimatrixspatialnull[1*2+1]*C1000[ix]*conj(phimatrix[1*2+0][g_idn[ix][TUP]])
                                     + 1.*phimatrixspatialnull[1*2+1]*C1110[ix]*conj(phimatrix[0*2+1][g_idn[ix][TUP]])
                                     + 1.*phimatrixspatialnull[1*2+1]*C1100[ix]*conj(phimatrix[1*2+1][g_idn[ix][TUP]])

                                     + 1.*phimatrixspatialnull[0*2+0]*C0011[ix]*conj(phimatrix[0*2+0][g_idn[ix][TUP]])
                                     + 1.*phimatrixspatialnull[0*2+0]*C0001[ix]*conj(phimatrix[1*2+0][g_idn[ix][TUP]])
                                     + 1.*phimatrixspatialnull[0*2+0]*C0111[ix]*conj(phimatrix[0*2+1][g_idn[ix][TUP]])
                                     + 1.*phimatrixspatialnull[0*2+0]*C0101[ix]*conj(phimatrix[1*2+1][g_idn[ix][TUP]])

                                     + 1.*phimatrixspatialnull[0*2+1]*C1011[ix]*conj(phimatrix[0*2+0][g_idn[ix][TUP]])
                                     + 1.*phimatrixspatialnull[0*2+1]*C1001[ix]*conj(phimatrix[1*2+0][g_idn[ix][TUP]])
                                     + 1.*phimatrixspatialnull[0*2+1]*C1111[ix]*conj(phimatrix[0*2+1][g_idn[ix][TUP]])
                                     + 1.*phimatrixspatialnull[0*2+1]*C1101[ix]*conj(phimatrix[1*2+1][g_idn[ix][TUP]]);

//tau_2
       final_corr[g_coord[ix][TUP]]+= -1.*phimatrixspatialnull[1*2+0]*C0010[ix]*conj(phimatrix[0*2+0][g_idn[ix][TUP]])
                                     + 1.*phimatrixspatialnull[1*2+0]*C0000[ix]*conj(phimatrix[1*2+0][g_idn[ix][TUP]])
                                     +-1.*phimatrixspatialnull[1*2+0]*C0110[ix]*conj(phimatrix[0*2+1][g_idn[ix][TUP]])
                                     + 1.*phimatrixspatialnull[1*2+0]*C0100[ix]*conj(phimatrix[1*2+1][g_idn[ix][TUP]])

                                     +-1.*phimatrixspatialnull[1*2+1]*C1010[ix]*conj(phimatrix[0*2+0][g_idn[ix][TUP]])
                                     + 1.*phimatrixspatialnull[1*2+1]*C1000[ix]*conj(phimatrix[1*2+0][g_idn[ix][TUP]])
                                     +-1.*phimatrixspatialnull[1*2+1]*C1110[ix]*conj(phimatrix[0*2+1][g_idn[ix][TUP]])
                                     + 1.*phimatrixspatialnull[1*2+1]*C1100[ix]*conj(phimatrix[1*2+1][g_idn[ix][TUP]])

                                     + 1.*phimatrixspatialnull[0*2+0]*C0011[ix]*conj(phimatrix[0*2+0][g_idn[ix][TUP]])
                                     +-1.*phimatrixspatialnull[0*2+0]*C0001[ix]*conj(phimatrix[1*2+0][g_idn[ix][TUP]])
                                     + 1.*phimatrixspatialnull[0*2+0]*C0111[ix]*conj(phimatrix[0*2+1][g_idn[ix][TUP]])
                                     +-1.*phimatrixspatialnull[0*2+0]*C0101[ix]*conj(phimatrix[1*2+1][g_idn[ix][TUP]])

                                     + 1.*phimatrixspatialnull[0*2+1]*C1011[ix]*conj(phimatrix[0*2+0][g_idn[ix][TUP]])
                                     +-1.*phimatrixspatialnull[0*2+1]*C1001[ix]*conj(phimatrix[1*2+0][g_idn[ix][TUP]])
                                     + 1.*phimatrixspatialnull[0*2+1]*C1111[ix]*conj(phimatrix[0*2+1][g_idn[ix][TUP]])
                                     +-1.*phimatrixspatialnull[0*2+1]*C1101[ix]*conj(phimatrix[1*2+1][g_idn[ix][TUP]]);

//tau3
       final_corr[g_coord[ix][TUP]]+=  1.*phimatrixspatialnull[0*2+0]*C0000[ix]*conj(phimatrix[0*2+0][g_idn[ix][TUP]])
                                     +-1.*phimatrixspatialnull[0*2+0]*C0010[ix]*conj(phimatrix[1*2+0][g_idn[ix][TUP]])
                                     + 1.*phimatrixspatialnull[0*2+0]*C0100[ix]*conj(phimatrix[0*2+1][g_idn[ix][TUP]])
                                     +-1.*phimatrixspatialnull[0*2+0]*C0110[ix]*conj(phimatrix[1*2+1][g_idn[ix][TUP]])

                                     + 1.*phimatrixspatialnull[0*2+1]*C1000[ix]*conj(phimatrix[0*2+0][g_idn[ix][TUP]])
                                     +-1.*phimatrixspatialnull[0*2+1]*C1010[ix]*conj(phimatrix[1*2+0][g_idn[ix][TUP]])
                                     + 1.*phimatrixspatialnull[0*2+1]*C1100[ix]*conj(phimatrix[0*2+1][g_idn[ix][TUP]])
                                     +-1.*phimatrixspatialnull[0*2+1]*C1110[ix]*conj(phimatrix[1*2+1][g_idn[ix][TUP]])

                                     +-1.*phimatrixspatialnull[1*2+0]*C0001[ix]*conj(phimatrix[0*2+0][g_idn[ix][TUP]])
                                     + 1.*phimatrixspatialnull[1*2+0]*C0011[ix]*conj(phimatrix[1*2+0][g_idn[ix][TUP]])
                                     +-1.*phimatrixspatialnull[1*2+0]*C0101[ix]*conj(phimatrix[0*2+1][g_idn[ix][TUP]])
                                     + 1.*phimatrixspatialnull[1*2+0]*C0111[ix]*conj(phimatrix[1*2+1][g_idn[ix][TUP]])

                                     +-1.*phimatrixspatialnull[1*2+1]*C1001[ix]*conj(phimatrix[0*2+0][g_idn[ix][TUP]])
                                     + 1.*phimatrixspatialnull[1*2+1]*C1011[ix]*conj(phimatrix[1*2+0][g_idn[ix][TUP]])
                                     +-1.*phimatrixspatialnull[1*2+1]*C1101[ix]*conj(phimatrix[0*2+1][g_idn[ix][TUP]])
                                     + 1.*phimatrixspatialnull[1*2+1]*C1111[ix]*conj(phimatrix[1*2+1][g_idn[ix][TUP]]);

    }
#if defined TM_USE_MPI
    for (ix=0; ix<T_global; ++ix){
       _Complex double tmp;
       MPI_Allreduce(&final_corr[ix], &tmp, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
       final_corr[ix]= tmp;
    }
#endif
    if (g_cart_id == 0){printf("Wilson Current Density correlator type 61a a la Petros (1) results\n");}
      for (ix=0; ix<T_global; ++ix){
        if (g_cart_id == 0){
        printf("WCDPL2 1 %.3d %10.10e %10.10e\n", ix, creal(final_corr[ix])/4.,cimag(final_corr[ix])/4.);
      }
    }


    free(C0000);
    free(C0001);
    free(C0010);
    free(C0011);
    free(C0100);
    free(C0101);
    free(C0110);
    free(C0111);
    free(C1000);
    free(C1001);
    free(C1010);
    free(C1011);
    free(C1100);
    free(C1101);
    free(C1110);
    free(C1111);

    for (ix=0;ix<4;++ix)
       free(phimatrix[ix]);
    free(phimatrix);
    free(final_corr);
#if defined TM_USE_MPI
    free(request);
#endif

}


void wilsoncurrent62a_petros( bispinor **propfields )
{
#if defined TM_USE_MPI
    int count;  
    MPI_Status  statuses[8];
    MPI_Request *request; 
    request=( MPI_Request *) malloc(sizeof(MPI_Request)*8);
#endif

    _Complex double **phimatrix=(_Complex double **)malloc(sizeof(_Complex double *)*4);

    _Complex double *C0000=(_Complex double *)malloc(sizeof(_Complex double)*VOLUMEPLUSRAND);
    _Complex double *C0001=(_Complex double *)malloc(sizeof(_Complex double)*VOLUMEPLUSRAND);
    _Complex double *C0010=(_Complex double *)malloc(sizeof(_Complex double)*VOLUMEPLUSRAND);
    _Complex double *C0011=(_Complex double *)malloc(sizeof(_Complex double)*VOLUMEPLUSRAND);
    _Complex double *C0100=(_Complex double *)malloc(sizeof(_Complex double)*VOLUMEPLUSRAND);
    _Complex double *C0101=(_Complex double *)malloc(sizeof(_Complex double)*VOLUMEPLUSRAND);
    _Complex double *C0110=(_Complex double *)malloc(sizeof(_Complex double)*VOLUMEPLUSRAND);
    _Complex double *C0111=(_Complex double *)malloc(sizeof(_Complex double)*VOLUMEPLUSRAND);
    _Complex double *C1000=(_Complex double *)malloc(sizeof(_Complex double)*VOLUMEPLUSRAND);
    _Complex double *C1001=(_Complex double *)malloc(sizeof(_Complex double)*VOLUMEPLUSRAND);
    _Complex double *C1010=(_Complex double *)malloc(sizeof(_Complex double)*VOLUMEPLUSRAND);
    _Complex double *C1011=(_Complex double *)malloc(sizeof(_Complex double)*VOLUMEPLUSRAND);
    _Complex double *C1100=(_Complex double *)malloc(sizeof(_Complex double)*VOLUMEPLUSRAND);
    _Complex double *C1101=(_Complex double *)malloc(sizeof(_Complex double)*VOLUMEPLUSRAND);
    _Complex double *C1110=(_Complex double *)malloc(sizeof(_Complex double)*VOLUMEPLUSRAND);
    _Complex double *C1111=(_Complex double *)malloc(sizeof(_Complex double)*VOLUMEPLUSRAND);

    _Complex double *final_corr=(_Complex double *)malloc(sizeof(_Complex double)*T_global);

    _Complex double *phimatrixspatialnull=(_Complex double *)malloc(sizeof(_Complex double)*4);

    int ix;


    for (ix=0;ix<4;++ix)
       phimatrix[ix]=(_Complex double *)malloc(sizeof(_Complex double)*VOLUMEPLUSRAND);
    for (ix=0;ix<VOLUME;++ix)
    {
       if (smearedcorrelator_BSM == 1){
         phimatrix[0][ix]= 1.*g_smeared_scalar_field[0][ix] + I*g_smeared_scalar_field[3][ix];
         phimatrix[1][ix]= 1.*g_smeared_scalar_field[2][ix] + I*g_smeared_scalar_field[1][ix];
         phimatrix[2][ix]=-1.*g_smeared_scalar_field[2][ix] + I*g_smeared_scalar_field[1][ix];
         phimatrix[3][ix]= 1.*g_smeared_scalar_field[0][ix] - I*g_smeared_scalar_field[3][ix];
       }
       else{
         phimatrix[0][ix]= 1.*g_scalar_field[0][ix] + I*g_scalar_field[3][ix];
         phimatrix[1][ix]= 1.*g_scalar_field[2][ix] + I*g_scalar_field[1][ix];
         phimatrix[2][ix]=-1.*g_scalar_field[2][ix] + I*g_scalar_field[1][ix];
         phimatrix[3][ix]= 1.*g_scalar_field[0][ix] - I*g_scalar_field[3][ix];
       }
    }

    for (ix=0;ix<4;++ix)
       phimatrixspatialnull[ix]=phimatrix[ix][0];

#if defined TM_USE_MPI
    for (ix=0; ix<4; ++ix){
      count=0;
      generic_exchange_direction_nonblocking( phimatrix[ix], sizeof(_Complex double), TDOWN   , request, &count );
      MPI_Waitall( count, request, statuses);
      count=0;
    }
    for (ix=0;ix<4;++ix){
       MPI_Bcast(&phimatrixspatialnull[ix], 1, MPI_DOUBLE_COMPLEX, 0, g_cart_grid);
    }
   int s1,c1,f1;
   for (s1=0; s1<2; ++s1)
      for (c1=0; c1<3; ++c1)
         for (f1=0; f1<2; ++f1){
            count=0;
            generic_exchange_direction_nonblocking( propfields[12*s1 + 4*c1 + 2*f1 + 0], sizeof(bispinor), TUP   , request, &count );
            MPI_Waitall( count, request, statuses);
            count=0;
            generic_exchange_direction_nonblocking( propfields[12*s1 + 4*c1 + 2*f1 + 1], sizeof(bispinor), TDOWN , request, &count );
            MPI_Waitall( count, request, statuses);
         }
#endif


    for (ix=0; ix<VOLUME; ++ix){
       trace_in_spinor_and_color62a(C0000,propfields,ix,0,0,0,0);
       trace_in_spinor_and_color62a(C0001,propfields,ix,0,0,0,1);
       trace_in_spinor_and_color62a(C0010,propfields,ix,0,0,1,0);
       trace_in_spinor_and_color62a(C0011,propfields,ix,0,0,1,1);
       trace_in_spinor_and_color62a(C0100,propfields,ix,0,1,0,0);
       trace_in_spinor_and_color62a(C0101,propfields,ix,0,1,0,1);
       trace_in_spinor_and_color62a(C0110,propfields,ix,0,1,1,0);
       trace_in_spinor_and_color62a(C0111,propfields,ix,0,1,1,1);
       trace_in_spinor_and_color62a(C1000,propfields,ix,1,0,0,0);
       trace_in_spinor_and_color62a(C1001,propfields,ix,1,0,0,1);
       trace_in_spinor_and_color62a(C1010,propfields,ix,1,0,1,0);
       trace_in_spinor_and_color62a(C1011,propfields,ix,1,0,1,1);
       trace_in_spinor_and_color62a(C1100,propfields,ix,1,1,0,0);
       trace_in_spinor_and_color62a(C1101,propfields,ix,1,1,0,1);
       trace_in_spinor_and_color62a(C1110,propfields,ix,1,1,1,0);
       trace_in_spinor_and_color62a(C1111,propfields,ix,1,1,1,1);

    }
#if defined TM_USE_MPI
    count=0;
    generic_exchange_direction_nonblocking( C0000, sizeof(_Complex double), TDOWN   , request, &count );
    MPI_Waitall( count, request, statuses);
    count=0;
    generic_exchange_direction_nonblocking( C0001, sizeof(_Complex double), TDOWN   , request, &count );
    MPI_Waitall( count, request, statuses);
    count=0;
    generic_exchange_direction_nonblocking( C0010, sizeof(_Complex double), TDOWN   , request, &count );
    MPI_Waitall( count, request, statuses);
    count=0;
    generic_exchange_direction_nonblocking( C0011, sizeof(_Complex double), TDOWN   , request, &count );
    MPI_Waitall( count, request, statuses);
    count=0;
    generic_exchange_direction_nonblocking( C0100, sizeof(_Complex double), TDOWN   , request, &count );
    MPI_Waitall( count, request, statuses);
    count=0;
    generic_exchange_direction_nonblocking( C0101, sizeof(_Complex double), TDOWN   , request, &count );
    MPI_Waitall( count, request, statuses);
    count=0;
    generic_exchange_direction_nonblocking( C0110, sizeof(_Complex double), TDOWN   , request, &count );
    MPI_Waitall( count, request, statuses);
    count=0;
    generic_exchange_direction_nonblocking( C0111, sizeof(_Complex double), TDOWN   , request, &count );
    MPI_Waitall( count, request, statuses);
    count=0;
    generic_exchange_direction_nonblocking( C1000, sizeof(_Complex double), TDOWN   , request, &count );
    MPI_Waitall( count, request, statuses);
    count=0;
    generic_exchange_direction_nonblocking( C1001, sizeof(_Complex double), TDOWN   , request, &count );
    MPI_Waitall( count, request, statuses);
    count=0;
    generic_exchange_direction_nonblocking( C1010, sizeof(_Complex double), TDOWN   , request, &count );
    MPI_Waitall( count, request, statuses);
    count=0;
    generic_exchange_direction_nonblocking( C1011, sizeof(_Complex double), TDOWN   , request, &count );
    MPI_Waitall( count, request, statuses);
    count=0;
    generic_exchange_direction_nonblocking( C1100, sizeof(_Complex double), TDOWN   , request, &count );
    MPI_Waitall( count, request, statuses);
    count=0;
    generic_exchange_direction_nonblocking( C1101, sizeof(_Complex double), TDOWN   , request, &count );
    MPI_Waitall( count, request, statuses);
    count=0;
    generic_exchange_direction_nonblocking( C1110, sizeof(_Complex double), TDOWN   , request, &count );
    MPI_Waitall( count, request, statuses);
    count=0;
    generic_exchange_direction_nonblocking( C1111, sizeof(_Complex double), TDOWN   , request, &count );
    MPI_Waitall( count, request, statuses);
    count=0;
#endif


    for (ix=0; ix<T_global; ++ix)
       final_corr[ix]=0.; 
    for (ix=0; ix<VOLUME; ++ix){

//tau_1
       final_corr[g_coord[ix][TUP]]+=  1.*phimatrixspatialnull[1*2+0]*C0010[g_idn[ix][TUP]]*conj(phimatrix[0*2+0][g_idn[ix][TUP]])
                                     + 1.*phimatrixspatialnull[1*2+0]*C0000[g_idn[ix][TUP]]*conj(phimatrix[1*2+0][g_idn[ix][TUP]])
                                     + 1.*phimatrixspatialnull[1*2+0]*C0110[g_idn[ix][TUP]]*conj(phimatrix[0*2+1][g_idn[ix][TUP]])
                                     + 1.*phimatrixspatialnull[1*2+0]*C0100[g_idn[ix][TUP]]*conj(phimatrix[1*2+1][g_idn[ix][TUP]])

                                     + 1.*phimatrixspatialnull[1*2+1]*C1010[g_idn[ix][TUP]]*conj(phimatrix[0*2+0][g_idn[ix][TUP]])
                                     + 1.*phimatrixspatialnull[1*2+1]*C1000[g_idn[ix][TUP]]*conj(phimatrix[1*2+0][g_idn[ix][TUP]])
                                     + 1.*phimatrixspatialnull[1*2+1]*C1110[g_idn[ix][TUP]]*conj(phimatrix[0*2+1][g_idn[ix][TUP]])
                                     + 1.*phimatrixspatialnull[1*2+1]*C1100[g_idn[ix][TUP]]*conj(phimatrix[1*2+1][g_idn[ix][TUP]])

                                     + 1.*phimatrixspatialnull[0*2+0]*C0011[g_idn[ix][TUP]]*conj(phimatrix[0*2+0][g_idn[ix][TUP]])
                                     + 1.*phimatrixspatialnull[0*2+0]*C0001[g_idn[ix][TUP]]*conj(phimatrix[1*2+0][g_idn[ix][TUP]])
                                     + 1.*phimatrixspatialnull[0*2+0]*C0111[g_idn[ix][TUP]]*conj(phimatrix[0*2+1][g_idn[ix][TUP]])
                                     + 1.*phimatrixspatialnull[0*2+0]*C0101[g_idn[ix][TUP]]*conj(phimatrix[1*2+1][g_idn[ix][TUP]])

                                     + 1.*phimatrixspatialnull[0*2+1]*C1011[g_idn[ix][TUP]]*conj(phimatrix[0*2+0][g_idn[ix][TUP]])
                                     + 1.*phimatrixspatialnull[0*2+1]*C1001[g_idn[ix][TUP]]*conj(phimatrix[1*2+0][g_idn[ix][TUP]])
                                     + 1.*phimatrixspatialnull[0*2+1]*C1111[g_idn[ix][TUP]]*conj(phimatrix[0*2+1][g_idn[ix][TUP]])
                                     + 1.*phimatrixspatialnull[0*2+1]*C1101[g_idn[ix][TUP]]*conj(phimatrix[1*2+1][g_idn[ix][TUP]]);

//tau_2
       final_corr[g_coord[ix][TUP]]+= -1.*phimatrixspatialnull[1*2+0]*C0010[g_idn[ix][TUP]]*conj(phimatrix[0*2+0][g_idn[ix][TUP]])
                                     + 1.*phimatrixspatialnull[1*2+0]*C0000[g_idn[ix][TUP]]*conj(phimatrix[1*2+0][g_idn[ix][TUP]])
                                     +-1.*phimatrixspatialnull[1*2+0]*C0110[g_idn[ix][TUP]]*conj(phimatrix[0*2+1][g_idn[ix][TUP]])
                                     + 1.*phimatrixspatialnull[1*2+0]*C0100[g_idn[ix][TUP]]*conj(phimatrix[1*2+1][g_idn[ix][TUP]])

                                     +-1.*phimatrixspatialnull[1*2+1]*C1010[g_idn[ix][TUP]]*conj(phimatrix[0*2+0][g_idn[ix][TUP]])
                                     + 1.*phimatrixspatialnull[1*2+1]*C1000[g_idn[ix][TUP]]*conj(phimatrix[1*2+0][g_idn[ix][TUP]])
                                     +-1.*phimatrixspatialnull[1*2+1]*C1110[g_idn[ix][TUP]]*conj(phimatrix[0*2+1][g_idn[ix][TUP]])
                                     + 1.*phimatrixspatialnull[1*2+1]*C1100[g_idn[ix][TUP]]*conj(phimatrix[1*2+1][g_idn[ix][TUP]])

                                     + 1.*phimatrixspatialnull[0*2+0]*C0011[g_idn[ix][TUP]]*conj(phimatrix[0*2+0][g_idn[ix][TUP]])
                                     +-1.*phimatrixspatialnull[0*2+0]*C0001[g_idn[ix][TUP]]*conj(phimatrix[1*2+0][g_idn[ix][TUP]])
                                     + 1.*phimatrixspatialnull[0*2+0]*C0111[g_idn[ix][TUP]]*conj(phimatrix[0*2+1][g_idn[ix][TUP]])
                                     +-1.*phimatrixspatialnull[0*2+0]*C0101[g_idn[ix][TUP]]*conj(phimatrix[1*2+1][g_idn[ix][TUP]])

                                     + 1.*phimatrixspatialnull[0*2+1]*C1011[g_idn[ix][TUP]]*conj(phimatrix[0*2+0][g_idn[ix][TUP]])
                                     +-1.*phimatrixspatialnull[0*2+1]*C1001[g_idn[ix][TUP]]*conj(phimatrix[1*2+0][g_idn[ix][TUP]])
                                     + 1.*phimatrixspatialnull[0*2+1]*C1111[g_idn[ix][TUP]]*conj(phimatrix[0*2+1][g_idn[ix][TUP]])
                                     +-1.*phimatrixspatialnull[0*2+1]*C1101[g_idn[ix][TUP]]*conj(phimatrix[1*2+1][g_idn[ix][TUP]]);

//tau3
       final_corr[g_coord[ix][TUP]]+=  1.*phimatrixspatialnull[0*2+0]*C0000[g_idn[ix][TUP]]*conj(phimatrix[0*2+0][g_idn[ix][TUP]])
                                     +-1.*phimatrixspatialnull[0*2+0]*C0010[g_idn[ix][TUP]]*conj(phimatrix[1*2+0][g_idn[ix][TUP]])
                                     + 1.*phimatrixspatialnull[0*2+0]*C0100[g_idn[ix][TUP]]*conj(phimatrix[0*2+1][g_idn[ix][TUP]])
                                     +-1.*phimatrixspatialnull[0*2+0]*C0110[g_idn[ix][TUP]]*conj(phimatrix[1*2+1][g_idn[ix][TUP]])

                                     + 1.*phimatrixspatialnull[0*2+1]*C1000[g_idn[ix][TUP]]*conj(phimatrix[0*2+0][g_idn[ix][TUP]])
                                     +-1.*phimatrixspatialnull[0*2+1]*C1010[g_idn[ix][TUP]]*conj(phimatrix[1*2+0][g_idn[ix][TUP]])
                                     + 1.*phimatrixspatialnull[0*2+1]*C1100[g_idn[ix][TUP]]*conj(phimatrix[0*2+1][g_idn[ix][TUP]])
                                     +-1.*phimatrixspatialnull[0*2+1]*C1110[g_idn[ix][TUP]]*conj(phimatrix[1*2+1][g_idn[ix][TUP]])

                                     +-1.*phimatrixspatialnull[1*2+0]*C0001[g_idn[ix][TUP]]*conj(phimatrix[0*2+0][g_idn[ix][TUP]])
                                     + 1.*phimatrixspatialnull[1*2+0]*C0011[g_idn[ix][TUP]]*conj(phimatrix[1*2+0][g_idn[ix][TUP]])
                                     +-1.*phimatrixspatialnull[1*2+0]*C0101[g_idn[ix][TUP]]*conj(phimatrix[0*2+1][g_idn[ix][TUP]])
                                     + 1.*phimatrixspatialnull[1*2+0]*C0111[g_idn[ix][TUP]]*conj(phimatrix[1*2+1][g_idn[ix][TUP]])

                                     +-1.*phimatrixspatialnull[1*2+1]*C1001[g_idn[ix][TUP]]*conj(phimatrix[0*2+0][g_idn[ix][TUP]])
                                     + 1.*phimatrixspatialnull[1*2+1]*C1011[g_idn[ix][TUP]]*conj(phimatrix[1*2+0][g_idn[ix][TUP]])
                                     +-1.*phimatrixspatialnull[1*2+1]*C1101[g_idn[ix][TUP]]*conj(phimatrix[0*2+1][g_idn[ix][TUP]])
                                     + 1.*phimatrixspatialnull[1*2+1]*C1111[g_idn[ix][TUP]]*conj(phimatrix[1*2+1][g_idn[ix][TUP]]);

    }
#if defined TM_USE_MPI
    for (ix=0; ix<T_global; ++ix){
       _Complex double tmp;
       MPI_Allreduce(&final_corr[ix], &tmp, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
       final_corr[ix]= tmp;
    }
#endif 
    if (g_cart_id == 0){printf("Wilson Current Density correlator type 62a a la Petros (1) results\n");}
      for (ix=0; ix<T_global; ++ix){
        if (g_cart_id == 0){
        printf("WCDPL2 1 %.3d %10.10e %10.10e\n", ix, creal(final_corr[ix])/4.,cimag(final_corr[ix])/4.);
      }
    }


    free(C0000);
    free(C0001);
    free(C0010);
    free(C0011);
    free(C0100);
    free(C0101);
    free(C0110);
    free(C0111);
    free(C1000);
    free(C1001);
    free(C1010);
    free(C1011);
    free(C1100);
    free(C1101);
    free(C1110);
    free(C1111);

    for (ix=0;ix<4;++ix)
       free(phimatrix[ix]);
    free(phimatrix);
    free(final_corr);
#if defined TM_USE_MPI
    free(request);
#endif

}
#endif
