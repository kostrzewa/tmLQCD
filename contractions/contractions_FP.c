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
#include "contractions/contractions_helper.h"

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


void density_density_1234_s0s0( bispinor ** propfields, int type_1234, _Complex double **results ){
   int ix,i;
   int f1,c1,s1;
   int spinorstart=0, spinorend=4;
   bispinor running;

   _Complex double *colortrace;
   _Complex double *spacetrace;
   _Complex double *spinortrace;
   _Complex double *flavortrace;

   int type;
   colortrace=(_Complex double *)malloc(sizeof(_Complex double) *8);
   spacetrace=(_Complex double *)malloc(sizeof(_Complex double) *8*T_global);
   spinortrace=(_Complex double *)malloc(sizeof(_Complex double)*2*T_global);
   flavortrace=(_Complex double *)malloc(sizeof(_Complex double)*T_global);

   *results=(_Complex double *)malloc(sizeof(_Complex double)*T_global);
   if (*results == NULL){
     printf("Error in memory allocation for results in s0s0\n");
     exit(1);
   }

   if ( (colortrace == NULL) || (spacetrace == NULL) || (spinortrace == NULL) || (flavortrace == NULL) )
   {
     printf("Error in mem allocation in density_density_1234_s0s0\n");
     exit(1);
   } 

   if ( ( type_1234 == TYPE_1 )|| ( type_1234 == TYPE_3 ) ) {
     spinorstart=0;
     spinorend  =2;
   }
   else if ( ( type_1234 == TYPE_2) || (type_1234 == TYPE_4) ){
     spinorstart=2;
     spinorend  =4;
   }
   else{
     fprintf(stdout,"Wrong argument for type_ab, it can only be TYPE_1, TYPE_2, TYPE_3 or TYPE_4\n");
     exit(1);
   }

   for (i=0; i<T_global; ++i)
      flavortrace[i]=0.;
//Trace over flavor space
   for (f1=0; f1<2; ++f1){
//Trace over the spinor indices
      for (i=0; i<2*T_global; ++i)
         spinortrace[i]=0.;

      for (s1= spinorstart; s1<spinorend; ++s1){

//Trace over the spatial indices
         for (i=0; i<8*T_global; ++i)
            spacetrace[i]=0.;

         for (ix = 0; ix< VOLUME; ++ix){

//Trace over the color indices for each sites
            for (i=0; i<8; ++i)
               colortrace[i]=0.;
            for (c1=0; c1<3; ++c1){
/*   
       TYPE  1 OR  2            (1-g5)/2*S(x  ,ytilde) fixed indices (c1, s1, f1)
       TYPE  3 OR  4            (1+g5)/2*S(x  ,ytilde) running indices bispinor
*/

//for the up quark
               if ( (type_1234 == TYPE_1) || (type_1234 == TYPE_2) ){
                 _vector_null( running.sp_up.s0 );
                 _vector_null( running.sp_up.s1 );
                 _vector_assign( running.sp_up.s2, propfields[12*s1+4*c1+2*f1][ix].sp_up.s2 );
                 _vector_assign( running.sp_up.s3, propfields[12*s1+4*c1+2*f1][ix].sp_up.s3 );
                 _vector_null( running.sp_dn.s0 );
                 _vector_null( running.sp_dn.s1 );
                 _vector_assign( running.sp_dn.s2, propfields[12*s1+4*c1+2*f1][ix].sp_dn.s2 );
                 _vector_assign( running.sp_dn.s3, propfields[12*s1+4*c1+2*f1][ix].sp_dn.s3 );
               }
               else if ((type_1234 == TYPE_3) || ( type_1234 == TYPE_4) ){
                 _vector_null( running.sp_up.s2 );
                 _vector_null( running.sp_up.s3 );
                 _vector_assign( running.sp_up.s0, propfields[12*s1+4*c1+2*f1][ix].sp_up.s0 );
                 _vector_assign( running.sp_up.s1, propfields[12*s1+4*c1+2*f1][ix].sp_up.s1 );
                 _vector_null( running.sp_dn.s2 );
                 _vector_null( running.sp_dn.s3 );
                 _vector_assign( running.sp_dn.s0, propfields[12*s1+4*c1+2*f1][ix].sp_dn.s0 );
                 _vector_assign( running.sp_dn.s1, propfields[12*s1+4*c1+2*f1][ix].sp_dn.s1 );
               }

/*   
       TYPE  1 OR  2     phi^dagger(x)*tau_i*  (1-g5)/2*S(x  ,ytilde)
       TYPE  3 OR  4     tau_i*phi(x)          (1+g5)/2*S(x  ,ytilde)
*/
               if ( (type_1234 == TYPE_1) || (type_1234 == TYPE_2)){
                 taui_scalarfield_spinor_s0s0( &running, &running, GAMMA_DN, ix, NODIR, DAGGER );
               }
               else if ( (type_1234 == TYPE_3) || (type_1234 == TYPE_4) ){
                 taui_scalarfield_spinor_s0s0( &running, &running, GAMMA_UP, ix, NODIR, NO_DAGG);
               }
/*   
       TYPE  1 OR  2     S(ytilde, x)*phi^dagger(x)*tau_i*  (1-g5)/2*S(x  ,ytilde)
       TYPE  3 OR  4     S(ytilde, x)*tau_i*phi(x)          (1+g5)/2*S(x  ,ytilde)
*/
               multiply_backward_propagator(&running, propfields, &running, ix, NODIR );

               trace_in_color(colortrace,&running,c1);

            }  //End of trace color

            trace_in_space(spacetrace,colortrace,ix);

         } //End of trace space
 
//Gather the results from all nodes to complete the trace in space

#if defined TM_USE_MPI
         for (i=0; i<8*T_global; ++i){
            _Complex double tmp;
            MPI_Allreduce(&spacetrace[i], &tmp, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
            spacetrace[i]= tmp;
         }
#endif
         trace_in_spinor(spinortrace, spacetrace, s1);
      }

//End of trace in spinor space
/*   
       TYPE  1      tau_i*phi(ytilde)*       (1+gamma5)/2*S(ytilde, x)*phi^dagger(x)*tau_i*  (1-g5)/2*S(x  ,ytilde)
       TYPE  2      phi(ytilde)^dagger*tau_i*(1-gamma5)/2*S(ytilde, x)*phi^dagger(x)*tau_i*  (1-g5)/2*S(x  ,ytilde)
       TYPE  3      tau_i*phi(ytilde)*       (1+gamma5)/2*S(ytilde, x)*tau_i*phi(x)          (1+g5)/2*S(x  ,ytilde)
       TYPE  4      phi(ytilde)^dagger*tau_i*(1-gamma5)/2*S(ytilde, x)*tau_i*phi(x)          (1+g5)/2*S(x  ,ytilde)
*/
      if ( (type_1234 == TYPE_1) || (type_1234 == TYPE_3) ){
         taui_scalarfield_flavoronly_s0s0( spinortrace, NO_DAGG );
      }
      else if ( (type_1234 == TYPE_4) || ( type_1234 == TYPE_2) ){
         taui_scalarfield_flavoronly_s0s0( spinortrace, DAGGER  );
      }

      trace_in_flavor( flavortrace, spinortrace, f1 );
   } //End of traCe in flavor space

   type = type_1234 == TYPE_1 ? 1 : type_1234 == TYPE_2 ? 2 : type_1234 == TYPE_3 ? 3 : 4 ;
   if (g_cart_id == 0){ printf( "Density Density correlator type (%s) results\n", type_1234 == TYPE_1 ? "1" : type_1234 == TYPE_2 ? "2" : type_1234 == TYPE_3 ? "3" : "4");}
   for (i=0; i<T_global; ++i){
      if (g_cart_id == 0){
        printf("DDS0S0 %d %.3d %10.10e %10.10e\n", type, i, creal(flavortrace[i])/4.,cimag(flavortrace[i])/4.);
        fflush(stdout);
      }
      (*results)[i]=flavortrace[i]/4.;
   }
   
   free(flavortrace);
   free(spacetrace);
   free(spinortrace);
   free(colortrace);

}


void density_density_1234( bispinor ** propfields, int type_1234, _Complex double  **results ){
   int ix,i;
   int f1,c1,s1,tauindex;
   int spinorstart=0, spinorend=4;
   bispinor running;

   _Complex double *colortrace;
   _Complex double *spacetrace;
   _Complex double *spinortrace;
   _Complex double *flavortrace;
   _Complex double *paulitrace;
   int type;

   colortrace=(_Complex double *)malloc(sizeof(_Complex double) *8);
   spacetrace=(_Complex double *)malloc(sizeof(_Complex double) *8*T_global);
   spinortrace=(_Complex double *)malloc(sizeof(_Complex double)*2*T_global);
   flavortrace=(_Complex double *)malloc(sizeof(_Complex double)*T_global);
   paulitrace= (_Complex double *)malloc(sizeof(_Complex double)*T_global);

   if ( (colortrace == NULL) || (spacetrace == NULL) || (spinortrace == NULL) || (flavortrace == NULL) || (paulitrace == NULL) )
   {
     printf("Error in mem allocation in density_density_1234\n");
     exit(1);
   }
   *results=(_Complex double *)malloc(sizeof(_Complex double)*4*T_global);
   if (*results == NULL){
     printf("Not enough memory for the results\n"); 
     exit(1);
   }

   if ( ( type_1234 == TYPE_1 )|| ( type_1234 == TYPE_3 ) ) {
     spinorstart=0; 
     spinorend  =2;
   }
   else if ( ( type_1234 == TYPE_2) || (type_1234 == TYPE_4) ){
     spinorstart=2;
     spinorend  =4;
   }
   else{
     if (g_cart_id ==0) fprintf(stdout, "Wrong arument for type_1234, it can only be TYPE_1, TYPE_2, TYPE_3, TYPE_4\n");
     exit(1);
  }

//Trace over the Pauli matrices
   for (i=0; i<T_global; ++i)
      paulitrace[i]=0.;

   for (tauindex=0; tauindex<3; ++tauindex){

//Trace over up and down flavors
      for (i=0; i<T_global; ++i)
         flavortrace[i]=0.;

      for (f1=0; f1<2; ++f1){
//Trace over the spinor indices you have to trace only over those two spinor 
//component that appear in the final spinor
         for (i=0; i<2*T_global; ++i)
            spinortrace[i]=0.;

         for (s1= spinorstart; s1<spinorend; ++s1){

//Trace over the spatial indices
            for (i=0; i<8*T_global; ++i)
               spacetrace[i]=0.;

            for (ix = 0; ix< VOLUME; ++ix){

//Trace over the color indices for each sites

               for (i=0; i<8; ++i)
                  colortrace[i]=0.;
               for (c1=0; c1<3; ++c1){
/*   
       TYPE  1 OR  2            (1-g5)/2*S(x  ,ytilde) fixed indices (c1, s1, f1)
       TYPE  3 OR  4            (1+g5)/2*S(x  ,ytilde) running indices bispinor
*/

//for the up quark
                  if ( (type_1234 == TYPE_1) || (type_1234 == TYPE_2) ){
                    _vector_null( running.sp_up.s0 );
                    _vector_null( running.sp_up.s1 );
                    _vector_assign( running.sp_up.s2, propfields[12*s1+4*c1+2*f1][ix].sp_up.s2 );
                    _vector_assign( running.sp_up.s3, propfields[12*s1+4*c1+2*f1][ix].sp_up.s3 );
                    _vector_null( running.sp_dn.s0 );
                    _vector_null( running.sp_dn.s1 );
                    _vector_assign( running.sp_dn.s2, propfields[12*s1+4*c1+2*f1][ix].sp_dn.s2 );
                    _vector_assign( running.sp_dn.s3, propfields[12*s1+4*c1+2*f1][ix].sp_dn.s3 );
                  }
                  else if ((type_1234 == TYPE_3) || ( type_1234 == TYPE_4)){
                    _vector_null( running.sp_up.s2 );
                    _vector_null( running.sp_up.s3 );
                    _vector_assign( running.sp_up.s0, propfields[12*s1+4*c1+2*f1][ix].sp_up.s0 );
                    _vector_assign( running.sp_up.s1, propfields[12*s1+4*c1+2*f1][ix].sp_up.s1 );
                    _vector_null( running.sp_dn.s2 );
                    _vector_null( running.sp_dn.s3 );
                    _vector_assign( running.sp_dn.s0, propfields[12*s1+4*c1+2*f1][ix].sp_dn.s0 );
                    _vector_assign( running.sp_dn.s1, propfields[12*s1+4*c1+2*f1][ix].sp_dn.s1 );
                  }

/*   
       TYPE  1 OR  2     phi^dagger(x)*tau_i*  (1-g5)/2*S(x  ,ytilde)
       TYPE  3 OR  4     tau_i*phi(x)          (1+g5)/2*S(x  ,ytilde)
*/
                  if ( (type_1234 == TYPE_1) || (type_1234 == TYPE_2)){
                    taui_scalarfield_spinor( &running, &running, GAMMA_DN, tauindex, ix, NODIR, DAGGER );
                  }
                  else if ( (type_1234 == TYPE_3) || (type_1234 == TYPE_4) ){
                    taui_scalarfield_spinor( &running, &running, GAMMA_UP, tauindex, ix, NODIR, NO_DAGG);
                  }
/*   
       TYPE  1 OR  2     S(ytilde, x)*phi^dagger(x)*tau_i*  (1-g5)/2*S(x  ,ytilde)
       TYPE  3 OR  4     S(ytilde, x)*tau_i*phi(x)          (1+g5)/2*S(x  ,ytilde)
*/
                  multiply_backward_propagator(&running, propfields, &running, ix, NODIR );

                  //delta( color component of bispinor running, c1) for all spinor and flavor indices
                  trace_in_color(colortrace,&running,c1);

               }  //End of trace color
               //sum over all lattice sites the result of the color trace
               trace_in_space(spacetrace,colortrace,ix);

            } //End of trace space

//Gather the results from all nodes to complete the trace in space
#if defined TM_USE_MPI
            for (i=0; i<8*T_global; ++i){
               _Complex double tmp;
               MPI_Allreduce(&spacetrace[i], &tmp, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, g_cart_grid);
               spacetrace[i]= tmp;
            }
#endif
            // delta (spinor components of spacetrace, s1) for all time slices and flavor indices
            trace_in_spinor(spinortrace, spacetrace, s1);
         }//End of trace in spinor space
/*   
       TYPE  1      tau_i*phi(ytilde)*       (1+gamma5)/2*S(ytilde, x)*phi^dagger(x)*tau_i*  (1-g5)/2*S(x  ,ytilde)
       TYPE  2      phi(ytilde)^dagger*tau_i*(1-gamma5)/2*S(ytilde, x)*phi^dagger(x)*tau_i*  (1-g5)/2*S(x  ,ytilde)
       TYPE  3      tau_i*phi(ytilde)*       (1+gamma5)/2*S(ytilde, x)*tau_i*phi(x)          (1+g5)/2*S(x  ,ytilde)
       TYPE  4      phi(ytilde)^dagger*tau_i*(1-gamma5)/2*S(ytilde, x)*tau_i*phi(x)          (1+g5)/2*S(x  ,ytilde)
*/
         if ( (type_1234 == TYPE_1) || (type_1234 == TYPE_3) ){
           taui_scalarfield_flavoronly( spinortrace, tauindex, NO_DAGG, LEFT );
         }
         else if ( (type_1234 == TYPE_4) || ( type_1234 == TYPE_2) ){
           taui_scalarfield_flavoronly( spinortrace, tauindex, DAGGER, LEFT  );
         }
         //delta(flavor component in spinortrace, f1) for all time slices
         trace_in_flavor( flavortrace, spinortrace, f1 );
      } //End of trace in flavor space
      type = type_1234 == TYPE_1 ? 1 : type_1234 == TYPE_2 ? 2 : type_1234 == TYPE_3 ? 3 : 4 ;
      if (g_cart_id == 0){printf("Density Density correlator type (%s) for tau matrix %d results\n", type_1234 == TYPE_1 ? "1" : type_1234 == TYPE_2 ? "2" : type_1234 == TYPE_3 ? "3" : "4",tauindex);}
      for (i=0; i<T_global; ++i){
        if (g_cart_id == 0){
         printf( "DDTAU%dTAU%d %d %.3d %10.10e %10.10e\n", tauindex,tauindex,type, i, creal(flavortrace[i])/4.,cimag(flavortrace[i])/4.);
         fflush(stdout);
        }
        (*results)[i+T_global*tauindex]=flavortrace[i]/4.;
      }
      //sum for all Pauli matrices
      for (i=0;i<T_global; ++i)
         paulitrace[i]+=flavortrace[i];
   } //End of trace for Pauli matrices

   type = type_1234 == TYPE_1 ? 1 : type_1234 == TYPE_2 ? 2 : type_1234 == TYPE_3 ? 3 : 4 ;
   if (g_cart_id == 0){printf("Density Density correlator type (%s) results\n", type_1234 == TYPE_1 ? "1" : type_1234 == TYPE_2 ? "2" : type_1234 == TYPE_3 ? "3" : "4");}
   for (i=0; i<T_global; ++i){
      if (g_cart_id == 0){
        printf("DD %d %.3d %10.10e %10.10e\n", type, i, creal(paulitrace[i])/4.,cimag(paulitrace[i])/4.);
        fflush(stdout);
      }
      (*results)[i+3*T_global]=paulitrace[i]/4.;
   }
   free(flavortrace);
   free(paulitrace);
   free(spacetrace);
   free(spinortrace);
   free(colortrace);

}
void giancarlodensity( bispinor ** propfields, int tau3, _Complex double  **results ){
   int ix,i;
   int f1,c1,s1;
   int spinorstart=0, spinorend=4;
   bispinor running;

   _Complex double *colortrace;
   _Complex double *spacetrace;
   _Complex double *spinortrace;
   _Complex double *flavortrace;

   colortrace= (_Complex double *)malloc(sizeof(_Complex double) *8);
   spacetrace= (_Complex double *)malloc(sizeof(_Complex double) *8*T_global);
   spinortrace=(_Complex double *)malloc(sizeof(_Complex double)*2*T_global);
   flavortrace=(_Complex double *)malloc(sizeof(_Complex double)*T_global);

   if ( (colortrace == NULL) || (spacetrace == NULL) || (spinortrace == NULL) || (flavortrace == NULL) )
   {
     printf("Error in mem allocation in giancarlo\n");
     exit(1);
   }
   *results=(_Complex double *)malloc(sizeof(_Complex double)*T_global);
   if (*results == NULL){
     printf("Not enough memory for the results\n");
     exit(1);
   }
   spinorstart=2;
   spinorend  =4;
   
//Trace over up and down flavors
   for (i=0; i<T_global; ++i)
     flavortrace[i]=0.;
   for (f1=0; f1<2; ++f1){
//Trace over the spinor indices you have to trace only over those two spinor 
//component that appear in the final spinor
     for (i=0; i<2*T_global; ++i)
       spinortrace[i]=0.;
     for (s1= spinorstart; s1<spinorend; ++s1){
//Trace over the spatial indices
       for (i=0; i<8*T_global; ++i)
          spacetrace[i]=0.;
       for (ix = 0; ix< VOLUME; ++ix){
//Trace over the color indices for each sites
         for (i=0; i<8; ++i)
           colortrace[i]=0.;
         for (c1=0; c1<3; ++c1){
//for the up quark
           _vector_null( running.sp_up.s2 );
           _vector_null( running.sp_up.s3 );
           _vector_assign( running.sp_up.s0, propfields[12*s1+4*c1+2*f1][ix].sp_up.s0 );
           _vector_assign( running.sp_up.s1, propfields[12*s1+4*c1+2*f1][ix].sp_up.s1 );
           _vector_null( running.sp_dn.s2 );
           _vector_null( running.sp_dn.s3 );
           _vector_assign( running.sp_dn.s0, propfields[12*s1+4*c1+2*f1][ix].sp_dn.s0 );
           _vector_assign( running.sp_dn.s1, propfields[12*s1+4*c1+2*f1][ix].sp_dn.s1 );

           if (tau3 == 1){
             taui_spinor( &running, &running, 2);              
           }

           mult_phi(&running, &running, ix, NO_DAGG);

           if (tau3 == 1){
             taui_spinor( &running, &running, 2);
           }

           multiply_backward_propagator(&running, propfields, &running, ix, NODIR );

            //delta( color component of bispinor running, c1) for all spinor and flavor indices
           trace_in_color(colortrace,&running,c1);

         }  //End of trace color
         //sum over all lattice sites the result of the color trace
         trace_in_space(spacetrace,colortrace,ix);
       } //End of trace space
//Gather the results from all nodes to complete the trace in space
#if defined TM_USE_MPI
       for (i=0; i<8*T_global; ++i){
          _Complex double tmp;
          MPI_Allreduce(&spacetrace[i], &tmp, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, g_cart_grid);
          spacetrace[i]= tmp;
       }
#endif
       // delta (spinor components of spacetrace, s1) for all time slices and flavor indices
       trace_in_spinor(spinortrace, spacetrace, s1);
     }//End of trace in spinor space

     if (tau3 == 1){
       mult_taui_flavoronly(spinortrace, 2);
     }
     mult_phi_flavoronly(spinortrace, DAGGER);
     if (tau3 == 1){
       mult_taui_flavoronly(spinortrace, 2);
     }
     trace_in_flavor( flavortrace, spinortrace, f1 );
   }
   if (g_cart_id == 0){printf("Giancarlo correlator  (%s) for tau3 results\n", tau3 == 1 ? "with" : "without" );fflush(stdout);}
   for (i=0; i<T_global; ++i){
     if (g_cart_id == 0){
       printf( "GIANCARLO %.3d %10.10e %10.10e\n", i, creal(flavortrace[i])/4.,cimag(flavortrace[i])/4.);
       fflush(stdout);
     }
     (*results)[i]=flavortrace[i]/4.;
   }
   free(flavortrace);
   free(spacetrace);
   free(spinortrace);
   free(colortrace);

}
void density_ptau_density_vector( bispinor **propfields, int type_12,_Complex double **results){

   int ix,i;
   int f1,c1,s1,tauindex;
   int spinorstart=0, spinorend=4;
   bispinor running;

   _Complex double *colortrace;
   _Complex double *spacetrace;
   _Complex double *spinortrace;
   _Complex double *flavortrace;
   _Complex double *paulitrace;
   int type;

   colortrace= (_Complex double *)malloc(sizeof(_Complex double)*8);
   spacetrace= (_Complex double *)malloc(sizeof(_Complex double)*8*T_global);
   spinortrace=(_Complex double *)malloc(sizeof(_Complex double)*2*T_global);
   flavortrace=(_Complex double *)malloc(sizeof(_Complex double)*T_global);
   paulitrace= (_Complex double *)malloc(sizeof(_Complex double)*T_global);

   if ( (colortrace == NULL) || (spacetrace == NULL) || (spinortrace == NULL) || (flavortrace == NULL) || (paulitrace == NULL) )
   {
     printf("Error in mem allocation in density_density_1234_sxsx\n");
     exit(1);
   }
   *results=(_Complex double *)malloc(sizeof(_Complex double)*4*T_global);
   if (*results == NULL){
     printf("Error in ptau anti commutator memory allocation routine\n");
     exit(1);
   }

//Trace over the Pauli matrices
   for (i=0; i<T_global; ++i)
      paulitrace[i]=0.;
   for (tauindex=0; tauindex<3; ++tauindex){
//Trace over up and down flavors
     for (i=0; i<T_global; ++i)
       flavortrace[i]=0.;
     for (f1=0; f1<2; ++f1){
//Trace over the spinor indices you have to trace only over those two spinor 
//component that appear in the final spinor
       for (i=0; i<2*T_global; ++i)
         spinortrace[i]=0.;
       for (s1= spinorstart; s1<spinorend; ++s1){
//Trace over the spatial indices
         for (i=0; i<8*T_global; ++i)
           spacetrace[i]=0.;
         for (ix = 0; ix< VOLUME; ++ix){
//Trace over the color indices for each sites
           for (i=0; i<8; ++i)
             colortrace[i]=0.;
           for (c1=0; c1<3; ++c1){
/*   
       TYPE  1         (1+g5)/2*S(x  ,ytilde) fixed indices (c1, s1, f1)
       TYPE  2         (1-g5)/2*S(x  ,ytilde) running indices bispinor
*/
             if ( (type_12 == TYPE_1) ){
                 _vector_null( running.sp_up.s2 );
                 _vector_null( running.sp_up.s3 );
                 _vector_assign( running.sp_up.s0, propfields[12*s1+4*c1+2*f1][ix].sp_up.s0 );
                 _vector_assign( running.sp_up.s1, propfields[12*s1+4*c1+2*f1][ix].sp_up.s1 );
                 _vector_null( running.sp_dn.s2 );
                 _vector_null( running.sp_dn.s3 );
                 _vector_assign( running.sp_dn.s0, propfields[12*s1+4*c1+2*f1][ix].sp_dn.s0 );
                 _vector_assign( running.sp_dn.s1, propfields[12*s1+4*c1+2*f1][ix].sp_dn.s1 );
             }
             else if ((type_12 == TYPE_2)){
                 _vector_null( running.sp_up.s0 );
                 _vector_null( running.sp_up.s1 );
                 _vector_assign( running.sp_up.s2, propfields[12*s1+4*c1+2*f1][ix].sp_up.s2 );
                 _vector_assign( running.sp_up.s3, propfields[12*s1+4*c1+2*f1][ix].sp_up.s3 );
                 _vector_null( running.sp_dn.s0 );
                 _vector_null( running.sp_dn.s1 );
                 _vector_assign( running.sp_dn.s2, propfields[12*s1+4*c1+2*f1][ix].sp_dn.s2 );
                 _vector_assign( running.sp_dn.s3, propfields[12*s1+4*c1+2*f1][ix].sp_dn.s3 );
             }
             else{
                 if (g_cart_id ==0) {printf("Wrong type of argument in ptau density contraction\n");exit(1);}
             }
/*
       TYPE  1         {tau_i;phi(x)}       *(1+g5)/2*S(x  ,ytilde) fixed indices (c1, s1, f1)
       TYPE  2         {tau_i;phi^dagger(x)}*(1-g5)/2*S(x  ,ytilde) running indices bispinor
*/

             if ( (type_12 == TYPE_1)){
                 phix_taui_anti_commutator_bispinor( &running,tauindex, GAMMA_UP, NO_DAGG, ix );
             }
             else if ((type_12 == TYPE_2)){
                 phix_taui_anti_commutator_bispinor( &running,tauindex, GAMMA_DN, DAGGER, ix );
             }
/*   
       TYPE  1     S(ytilde, x)*{tau_i;phi(x)}       *(1+g5)/2*S(x  ,ytilde) fixed indices (c1, s1, f1)
       TYPE  2     S(ytilde, x)*{tau_i;phi^dagger(x)}*(1-g5)/2*S(x  ,ytilde) running indices bispinor
*/
             multiply_backward_propagator(&running, propfields, &running, ix, NODIR );

/*   
       TYPE  1     gamma5*S(ytilde, x)*{tau_i;phi(x)}       *(1+g5)/2*S(x  ,ytilde) fixed indices (c1, s1, f1)
       TYPE  2     gamma5*S(ytilde, x)*{tau_i;phi^dagger(x)}*(1-g5)/2*S(x  ,ytilde) running indices bispinor
*/
             bispinor_timesgamma5( &running );

/*   
       TYPE  1     tau_i*gamma5*S(ytilde, x)*{tau_i;phi(x)}       *(1+g5)/2*S(x  ,ytilde) fixed indices (c1, s1, f1)
       TYPE  2     tau_i*gamma5*S(ytilde, x)*{tau_i;phi^dagger(x)}*(1-g5)/2*S(x  ,ytilde) running indices bispinor
*/
             bispinor_taui( &running, tauindex);


                  //delta( color component of bispinor running, c1) for all spinor and flavor indices
             trace_in_color(colortrace,&running,c1);

           }  //End of trace color
               //sum over all lattice sites the result of the color trace
           trace_in_space(spacetrace,colortrace,ix);

         } //End of trace space

//Gather the results from all nodes to complete the trace in space
#if defined TM_USE_MPI
         for (i=0; i<8*T_global; ++i){
           _Complex double tmp;
           MPI_Allreduce(&spacetrace[i], &tmp, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, g_cart_grid);
           spacetrace[i]= tmp;
         }
#endif
            // delta (spinor components of spacetrace, s1) for all time slices and flavor indices
         trace_in_spinor(spinortrace, spacetrace, s1);
       }//End of trace in spinor space

         //delta(flavor component in spinortrace, f1) for all time slices
       trace_in_flavor( flavortrace, spinortrace, f1 );
     } //End of trace in flavor space
     type = type_12 == TYPE_1 ? 1 : type_12 == TYPE_2 ? 2 : 0;
     if (type==0 && g_cart_id==0) {printf("Error in calculating the ptau vector density contractions\n");exit(1);}
     if (g_cart_id == 0){printf("DensityP Vector Density correlator type (%s) results\n", type_12 == TYPE_1 ? "1" : "2");}
     for (i=0; i<T_global; ++i){
       if (g_cart_id == 0){
         printf("P%dDP%d %d %.3d %10.10e %10.10e\n", tauindex+1, tauindex+1, type, i, creal(flavortrace[i])/4.,cimag(flavortrace[i])/4.);
         fflush(stdout);
       }
     }
     //sum for all Pauli matrices
     for (i=0;i<T_global; ++i){
       paulitrace[i]+=flavortrace[i];
       (*results)[i+tauindex*T_global]=flavortrace[i]/4.;
     }
   } //End of trace for Pauli matrices

   type = type_12 == TYPE_1 ? 1 : type_12 == TYPE_2 ? 2 : 0 ;
   if (type==0 && g_cart_id==0) {printf("Error in calculating the ptau vector density contractions\n");exit(1);}
   if (g_cart_id == 0){printf("DensityP Vector Density correlator type (%s) results\n", type_12 == TYPE_1 ? "1" : "2");}
   for (i=0; i<T_global; ++i){
      if (g_cart_id == 0){
        printf("PDP %d %.3d %10.10e %10.10e\n", type, i, creal(paulitrace[i])/4.,cimag(paulitrace[i])/4.);
        fflush(stdout);
      }
      (*results)[i+3*T_global]=paulitrace[i]/4.;
   }
   free(flavortrace);
   free(paulitrace);
   free(spacetrace);
   free(spinortrace);
   free(colortrace);
}
void density_density_1234_sxsx( bispinor ** propfields, int type_1234, _Complex double **results ){
   int ix,i;
   int f1,c1,s1,tauindex;
   int spinorstart=0, spinorend=4;
   bispinor running;

   _Complex double *colortrace;
   _Complex double *spacetrace;
   _Complex double *spinortrace;
   _Complex double *flavortrace;
   _Complex double *paulitrace;
   int type;

   colortrace= (_Complex double *)malloc(sizeof(_Complex double)*8);
   spacetrace= (_Complex double *)malloc(sizeof(_Complex double)*8*T_global);
   spinortrace=(_Complex double *)malloc(sizeof(_Complex double)*2*T_global);
   flavortrace=(_Complex double *)malloc(sizeof(_Complex double)*T_global);
   paulitrace= (_Complex double *)malloc(sizeof(_Complex double)*T_global);

   if ( (colortrace == NULL) || (spacetrace == NULL) || (spinortrace == NULL) || (flavortrace == NULL) || (paulitrace == NULL) )
   {
     printf("Error in mem allocation in density_density_1234_sxsx\n");
     exit(1);
   }
   *results=(_Complex double *)malloc(sizeof(_Complex double)*4*T_global);
   if (*results == NULL){
     printf("Error in sxsx\n");
     exit(1);
   }
   if ( ( type_1234 == TYPE_1 )|| ( type_1234 == TYPE_3 ) ) {
     spinorstart=0;
     spinorend  =2;
   }
   else if ( ( type_1234 == TYPE_2) || (type_1234 == TYPE_4) ){
     spinorstart=2;
     spinorend  =4;
   }
   else{
     if (g_cart_id ==0) fprintf(stdout, "Wrong arument for type_1234, it can only be TYPE_1, TYPE_2, TYPE_3, TYPE_4\n");
     exit(1);
   }

//Trace over the Pauli matrices
   for (i=0; i<T_global; ++i)
      paulitrace[i]=0.;
   for (tauindex=0; tauindex<3; ++tauindex){
//Trace over up and down flavors
     for (i=0; i<T_global; ++i)
       flavortrace[i]=0.;
     for (f1=0; f1<2; ++f1){

//Trace over the spinor indices you have to trace only over those two spinor 
//component that appear in the final spinor
       for (i=0; i<2*T_global; ++i)
         spinortrace[i]=0.;
       for (s1= spinorstart; s1<spinorend; ++s1){

//Trace over the spatial indices
         for (i=0; i<8*T_global; ++i)
           spacetrace[i]=0.;

         for (ix = 0; ix< VOLUME; ++ix){

//Trace over the color indices for each sites
           for (i=0; i<8; ++i)
             colortrace[i]=0.;
           for (c1=0; c1<3; ++c1){
/*   
       TYPE  1 OR  2            (1-g5)/2*S(x  ,ytilde) fixed indices (c1, s1, f1)
       TYPE  3 OR  4            (1+g5)/2*S(x  ,ytilde) running indices bispinor
*/

//for the up quark
             if ( (type_1234 == TYPE_1) || (type_1234 == TYPE_2) ){
                 _vector_null( running.sp_up.s0 );
                 _vector_null( running.sp_up.s1 );
                 _vector_assign( running.sp_up.s2, propfields[12*s1+4*c1+2*f1][ix].sp_up.s2 );
                 _vector_assign( running.sp_up.s3, propfields[12*s1+4*c1+2*f1][ix].sp_up.s3 );
                 _vector_null( running.sp_dn.s0 );
                 _vector_null( running.sp_dn.s1 );
                 _vector_assign( running.sp_dn.s2, propfields[12*s1+4*c1+2*f1][ix].sp_dn.s2 );
                 _vector_assign( running.sp_dn.s3, propfields[12*s1+4*c1+2*f1][ix].sp_dn.s3 );
             }
             else if ((type_1234 == TYPE_3) || ( type_1234 == TYPE_4)){
                 _vector_null( running.sp_up.s2 );
                 _vector_null( running.sp_up.s3 );
                 _vector_assign( running.sp_up.s0, propfields[12*s1+4*c1+2*f1][ix].sp_up.s0 );
                 _vector_assign( running.sp_up.s1, propfields[12*s1+4*c1+2*f1][ix].sp_up.s1 );
                 _vector_null( running.sp_dn.s2 );
                 _vector_null( running.sp_dn.s3 );
                 _vector_assign( running.sp_dn.s0, propfields[12*s1+4*c1+2*f1][ix].sp_dn.s0 );
                 _vector_assign( running.sp_dn.s1, propfields[12*s1+4*c1+2*f1][ix].sp_dn.s1 );
                }

/*   
       TYPE  1 OR  2     phi^dagger(x)*tau_i*  (1-g5)/2*S(x  ,ytilde)
       TYPE  3 OR  4     tau_i*phi(x)          (1+g5)/2*S(x  ,ytilde)
*/
             if ( (type_1234 == TYPE_1) || (type_1234 == TYPE_2)){
                 taui_scalarfield_spinor( &running, &running, GAMMA_DN, tauindex, ix, NODIR, DAGGER );
             }
             else if ( (type_1234 == TYPE_3) || (type_1234 == TYPE_4) ){
                 taui_scalarfield_spinor( &running, &running, GAMMA_UP, tauindex, ix, NODIR, NO_DAGG);
             }
/*   
       TYPE  1 OR  2     S(ytilde, x)*phi^dagger(x)*tau_i*  (1-g5)/2*S(x  ,ytilde)
       TYPE  3 OR  4     S(ytilde, x)*tau_i*phi(x)          (1+g5)/2*S(x  ,ytilde)
*/
             multiply_backward_propagator(&running, propfields, &running, ix, NODIR );

                  //delta( color component of bispinor running, c1) for all spinor and flavor indices
             trace_in_color(colortrace,&running,c1);

           }  //End of trace color
               //sum over all lattice sites the result of the color trace
           trace_in_space(spacetrace,colortrace,ix);

         } //End of trace space

//Gather the results from all nodes to complete the trace in space
#if defined TM_USE_MPI
         for (i=0; i<8*T_global; ++i){
           _Complex double tmp;
           MPI_Allreduce(&spacetrace[i], &tmp, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, g_cart_grid);
           spacetrace[i]= tmp;
         }
#endif
            // delta (spinor components of spacetrace, s1) for all time slices and flavor indices
         trace_in_spinor(spinortrace, spacetrace, s1);
       }//End of trace in spinor space
/*   
       TYPE  1      tau_i*phi(ytilde)*       (1+gamma5)/2*S(ytilde, x)*phi^dagger(x)*tau_i*  (1-g5)/2*S(x  ,ytilde)
       TYPE  2      phi(ytilde)^dagger*tau_i*(1-gamma5)/2*S(ytilde, x)*phi^dagger(x)*tau_i*  (1-g5)/2*S(x  ,ytilde)
       TYPE  3      tau_i*phi(ytilde)*       (1+gamma5)/2*S(ytilde, x)*tau_i*phi(x)          (1+g5)/2*S(x  ,ytilde)
       TYPE  4      phi(ytilde)^dagger*tau_i*(1-gamma5)/2*S(ytilde, x)*tau_i*phi(x)          (1+g5)/2*S(x  ,ytilde)
*/
       if ( (type_1234 == TYPE_1) || (type_1234 == TYPE_3) ){
         taui_scalarfield_flavoronly( spinortrace, tauindex, NO_DAGG, LEFT );
       }
       else if ( (type_1234 == TYPE_4) || ( type_1234 == TYPE_2) ){
         taui_scalarfield_flavoronly( spinortrace, tauindex, DAGGER, LEFT  );
       }
         //delta(flavor component in spinortrace, f1) for all time slices
       trace_in_flavor( flavortrace, spinortrace, f1 );
      } //End of trace in flavor space
      type = type_1234 == TYPE_1 ? 1 : type_1234 == TYPE_2 ? 2 : type_1234 == TYPE_3 ? 3 : 4 ;
      if (g_cart_id == 0){printf("Density Density correlator type (%s) results\n", type_1234 == TYPE_1 ? "1" : type_1234 == TYPE_2 ? "2" : type_1234 == TYPE_3 ? "3" : "4");}
      for (i=0; i<T_global; ++i){
        if (g_cart_id == 0){
          printf("DDS%dS%d %d %.3d %10.10e %10.10e\n", tauindex+1, tauindex+1, type, i, creal(flavortrace[i])/4.,cimag(flavortrace[i])/4.);
          fflush(stdout);
        }
      }
      //sum for all Pauli matrices
      for (i=0;i<T_global; ++i){
        paulitrace[i]+=flavortrace[i];
        (*results)[i+tauindex*T_global]=flavortrace[i]/4.;
      }
      
   } //End of trace for Pauli matrices

   type = type_1234 == TYPE_1 ? 1 : type_1234 == TYPE_2 ? 2 : type_1234 == TYPE_3 ? 3 : 4 ;
   if (g_cart_id == 0){printf("Density Density correlator type (%s) results\n", type_1234 == TYPE_1 ? "1" : type_1234 == TYPE_2 ? "2" : type_1234 == TYPE_3 ? "3" : "4");}
   for (i=0; i<T_global; ++i){
      if (g_cart_id == 0){
        printf("DD %d %.3d %10.10e %10.10e\n", type, i, creal(paulitrace[i])/4.,cimag(paulitrace[i])/4.);
        fflush(stdout);
      }
      (*results)[i+3*T_global]=paulitrace[i]/4.;
   }

   free(flavortrace);
   free(paulitrace);
   free(spacetrace);
   free(spinortrace);
   free(colortrace);
}




void vector_axial_current_density_1234( bispinor ** propfields, int type_1234,int taudensity, int taucurrent, int vectororaxial, int scalarorpseudoscalar, _Complex double **results ){
   int ix,i;
   int f1,c1,s1;
   int spinorstart=0, spinorend=4;
   bispinor running;

   _Complex double *colortrace;
   _Complex double *spacetrace;
   _Complex double *spinortrace;
   _Complex double *flavortrace;
   int type;

#if defined TM_USE_MPI
   int count;
   MPI_Status  statuses[8];
   MPI_Request *request;
   request=( MPI_Request *) malloc(sizeof(MPI_Request)*8);
#endif


   colortrace=(_Complex double *)malloc(sizeof(_Complex double) *8);
   spacetrace=(_Complex double *)malloc(sizeof(_Complex double) *8*T_global);
   spinortrace=(_Complex double *)malloc(sizeof(_Complex double)*2*T_global);
   flavortrace=(_Complex double *)malloc(sizeof(_Complex double)*T_global);

   if ( (colortrace == NULL) || (spacetrace == NULL) || (spinortrace == NULL) || (flavortrace == NULL) )
   {
     printf("Error in mem allocation in density_density_1234\n");
     exit(1);
   }
   *results=(_Complex double *)malloc(sizeof(_Complex double)*T_global);
   if (*results == NULL){
     printf("Error in vector current density\n");
     exit(1);
   }


   if ( ( type_1234 == TYPE_1 )|| ( type_1234 == TYPE_3 ) ) {
     spinorstart=0;
     spinorend  =2;
   }
   else if ( ( type_1234 == TYPE_2) || (type_1234 == TYPE_4) ){
     spinorstart=2;
     spinorend  =4;
   }
   else{
     if (g_cart_id ==0) fprintf(stdout, "Wrong arument for type_1234, it can only be TYPE_1, TYPE_2, TYPE_3, TYPE_4\n");
     exit(1);
   }

//Doing the neccessary communication
#if defined TM_USE_MPI
   for (s1=spinorstart; s1<spinorend; ++s1)
      for (c1=0; c1<3; ++c1)
         for (f1=0; f1<2; ++f1){
            count=0;
            generic_exchange_direction_nonblocking( propfields[12*s1 + 4*c1 + 2*f1 + 0], sizeof(bispinor), TDOWN, request, &count );
            MPI_Waitall( count, request, statuses);
            count=0;
            generic_exchange_direction_nonblocking( propfields[12*s1 + 4*c1 + 2*f1 + 1], sizeof(bispinor), TDOWN, request, &count);
            MPI_Waitall( count, request, statuses);
         }
   free(request);
#endif


//Trace over up and down flavors
   for (i=0; i<T_global; ++i)
     flavortrace[i]=0.;
   for (f1=0; f1<2; ++f1){
//Trace over the spinor indices you have to trace only over those two spinor 
//component that appear in the final spinor
     for (i=0; i<2*T_global; ++i)
       spinortrace[i]=0.;

     for (s1= spinorstart; s1<spinorend; ++s1){
//Trace over the spatial indices
       for (i=0; i<8*T_global; ++i)
         spacetrace[i]=0.;
       for (ix = 0; ix< VOLUME; ++ix){
//Trace over the color indices for each sites
         for (i=0; i<8; ++i)
           colortrace[i]=0.;
         for (c1=0; c1<3; ++c1){
           if ( (type_1234 == TYPE_1) || (type_1234 == TYPE_2) ){
            bispinor_mult_su3matrix( &running, &propfields[12*s1+4*c1+2*f1][ix], &g_gauge_field[g_idn[ix][TUP]][TUP], NO_DAGG);
           }
           else if ((type_1234 == TYPE_3) || ( type_1234 == TYPE_4)){
            bispinor_mult_su3matrix( &running, &propfields[12*s1+4*c1+2*f1][g_idn[ix][TUP]],  &g_gauge_field[g_idn[ix][TUP]][TUP], DAGGER);
           }

           if (vectororaxial == 1){
            bispinor_timesgamma5(&running);
           }
//Multiplication with gamma0
           bispinor_timesgamma0(&running);

//Multiplication with tau_i input parameter for the current
           bispinor_taui(&running, taucurrent);

//Backward propagator multiplication
           if (type_1234 == TYPE_1 || type_1234 == TYPE_2){
              multiply_backward_propagator(&running, propfields, &running, ix, TDOWN );
           }
           else if (type_1234 == TYPE_3 || type_1234 == TYPE_4){
              multiply_backward_propagator(&running, propfields, &running, ix, NODIR );
           }
           trace_in_color(colortrace,&running,c1);
         }  //End of trace color
         trace_in_space(spacetrace,colortrace,ix);
       } //End of trace space
//Gather the results from all nodes to complete the trace in space
#if defined TM_USE_MPI
       for (i=0; i<8*T_global; ++i){
         _Complex double tmp;
        MPI_Allreduce(&spacetrace[i], &tmp, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, g_cart_grid);
        spacetrace[i]= tmp;
       }
#endif
           // delta (spinor components of spacetrace, s1) for all time slices and flavor indices
       trace_in_spinor(spinortrace, spacetrace, s1);

     }//End of trace in spinor space
/*
     if ( type_1234 == TYPE_1 || type_1234 == TYPE_3 ){
       taui_scalarfield_flavoronly( spinortrace, taudensity, NO_DAGG, RIGHT );
     }
     else if ( type_1234 == TYPE_2 || type_1234 == TYPE_4 ){
       taui_scalarfield_flavoronly( spinortrace, taudensity, DAGGER, RIGHT  );
     }*/
     if (scalarorpseudoscalar == 0){
       phi0_taui_commutator( spinortrace, taudensity );
     }
     else if (scalarorpseudoscalar == 1){
      if ( type_1234 == TYPE_1 || type_1234 == TYPE_3 ){
       phi0_taui_anticommutator( spinortrace, taudensity, NO_DAGG );
      }
      else if ( type_1234 == TYPE_2 || type_1234 == TYPE_4 ){
       phi0_taui_anticommutator( spinortrace, taudensity, DAGGER );
      }
     }
     //delta(flavor component in spinortrace, f1) for all time slices
     trace_in_flavor( flavortrace, spinortrace, f1 );
   } //End of trace in flavor space
   type = type_1234 == TYPE_1 ? 1 : type_1234 == TYPE_2 ? 2 : type_1234 == TYPE_3 ? 3 : 4 ;
   if (g_cart_id == 0){printf("Vector current Density correlator type (%s) for tau matrix current %d density %d results\n", type_1234 == TYPE_1 ? "1" : type_1234 == TYPE_2 ? "2" : type_1234 == TYPE_3 ? "3" : "4", taucurrent, taudensity);}
   for (i=0; i<T_global; ++i){
     if (g_cart_id == 0){
      if (vectororaxial == 0)
        printf("VECTORCURRENT%dDENSITY%d %d %.3d %10.10e %10.10e\n", taucurrent,taudensity, type, i, creal(flavortrace[i])/4.,cimag(flavortrace[i])/4.);
      else
       printf("AXIALCURRENT%dDENSITY%d %d %.3d %10.10e %10.10e\n", taucurrent,taudensity, type, i, creal(flavortrace[i])/4.,cimag(flavortrace[i])/4.);
      fflush(stdout);
     }
     (*results)[i]=flavortrace[i]/4.;
   }
   free(flavortrace);
   free(spacetrace);
   free(spinortrace);
   free(colortrace);
}



void vector_2_psueodoscalar_1_12( bispinor ** propfields, int type_12,_Complex double **results ){
   int ix,i;
   int f1,c1,s1;
   int spinorstart=0, spinorend=4;
   bispinor running;

   _Complex double *colortrace;
   _Complex double *spacetrace;
   _Complex double *spinortrace;
   _Complex double *flavortrace;
   int type;

#if defined TM_USE_MPI
   int count;
   MPI_Status  statuses[8];
   MPI_Request *request;
   request=( MPI_Request *) malloc(sizeof(MPI_Request)*8);
#endif


   colortrace=(_Complex double *)malloc(sizeof(_Complex double) *8);
   spacetrace=(_Complex double *)malloc(sizeof(_Complex double) *8*T_global);
   spinortrace=(_Complex double *)malloc(sizeof(_Complex double)*2*T_global);
   flavortrace=(_Complex double *)malloc(sizeof(_Complex double)*T_global);

   if ( (colortrace == NULL) || (spacetrace == NULL) || (spinortrace == NULL) || (flavortrace == NULL) )
   {
     printf("Error in mem allocation in density_density_1234\n");
     exit(1);
   }
   *results=(_Complex double *)malloc(sizeof(_Complex double)*T_global);
   if (*results == NULL){
     printf("Error in vector current density\n");
     exit(1);
   }

   spinorstart=0;
   spinorend  =4;

//Doing the neccessary communication
#if defined TM_USE_MPI
   for (s1=spinorstart; s1<spinorend; ++s1)
      for (c1=0; c1<3; ++c1)
         for (f1=0; f1<2; ++f1){
            count=0;
            generic_exchange_direction_nonblocking( propfields[12*s1 + 4*c1 + 2*f1 + 0], sizeof(bispinor), TDOWN, request, &count );
            MPI_Waitall( count, request, statuses);
            count=0;
            generic_exchange_direction_nonblocking( propfields[12*s1 + 4*c1 + 2*f1 + 1], sizeof(bispinor), TDOWN, request, &count);
            MPI_Waitall( count, request, statuses);
         }
   free(request);
#endif


//Trace over up and down flavors
   for (i=0; i<T_global; ++i)
     flavortrace[i]=0.;
   for (f1=0; f1<2; ++f1){
//Trace over the spinor indices you have to trace only over those two spinor 
//component that appear in the final spinor
     for (i=0; i<2*T_global; ++i)
       spinortrace[i]=0.;

     for (s1= spinorstart; s1<spinorend; ++s1){
//Trace over the spatial indices
       for (i=0; i<8*T_global; ++i)
         spacetrace[i]=0.;
       for (ix = 0; ix< VOLUME; ++ix){
//Trace over the color indices for each sites
         for (i=0; i<8; ++i)
           colortrace[i]=0.;
         for (c1=0; c1<3; ++c1){
           if ( type_12 == TYPE_1 ){
            bispinor_mult_su3matrix( &running, &propfields[12*s1+4*c1+2*f1][ix], &g_gauge_field[g_idn[ix][TUP]][TUP], NO_DAGG);
           }
           else if (type_12 == TYPE_2 ){
            bispinor_mult_su3matrix( &running, &propfields[12*s1+4*c1+2*f1][g_idn[ix][TUP]],  &g_gauge_field[g_idn[ix][TUP]][TUP], DAGGER);
           }
           else{
            if (g_cart_id == 0) printf("Error in indexing in test routine\n");
           }

//Multiplication with gamma0
           bispinor_timesgamma0(&running);

//Multiplication with tau_i input parameter for the current
           bispinor_taui(&running, 1);

//Backward propagator multiplication
           if ( type_12 == TYPE_1 ){
              multiply_backward_propagator(&running, propfields, &running, ix, TDOWN );
           }
           else if (type_12 == TYPE_2 ){
              multiply_backward_propagator(&running, propfields, &running, ix, NODIR );
           }
           else{
            if (g_cart_id == 0) printf("Error in indexing in test routine\n");
           }
           trace_in_color(colortrace,&running,c1);
         }  //End of trace color
         trace_in_space(spacetrace,colortrace,ix);
       } //End of trace space
//Gather the results from all nodes to complete the trace in space
#if defined TM_USE_MPI
       for (i=0; i<8*T_global; ++i){
         _Complex double tmp;
        MPI_Allreduce(&spacetrace[i], &tmp, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, g_cart_grid);
        spacetrace[i]= tmp;
       }
#endif
       int fx;
       for(i=0;i<T_global;++i){
         for (fx=0;fx<2;++fx){
           spacetrace[8*i+4*fx+0]*=+1.;
           spacetrace[8*i+4*fx+1]*=+1.;
           spacetrace[8*i+4*fx+2]*=-1.;
           spacetrace[8*i+4*fx+3]*=-1.;
         }
       }
      // delta (spinor components of spacetrace, s1) for all time slices and flavor indices
       trace_in_spinor(spinortrace, spacetrace, s1);

     }//End of trace in spinor space
     for (i=0;i<T_global;++i){
       _Complex double tmpup=spinortrace[2*i+0];
       _Complex double tmpdn=spinortrace[2*i+1];
       spinortrace[2*i+0]=tmpdn;
       spinortrace[2*i+1]=tmpup;
     }

     //delta(flavor component in spinortrace, f1) for all time slices
     trace_in_flavor( flavortrace, spinortrace, f1 );
   } //End of trace in flavor space
   type = type_12 == TYPE_1 ? 1 : 2;
   if (g_cart_id == 0){printf("TESTING Vector current Density correlator type (%s) for tau matrix current %d density %d results\n", type_12 == TYPE_1 ? "1" : "2", 1, 0);}
   for (i=0; i<T_global; ++i){
     if (g_cart_id == 0){
       printf("VECTORCURRENT%dDENSITY%d %d %.3d %10.10e %10.10e\n", 1,0, type, i, creal(flavortrace[i])/4.,cimag(flavortrace[i])/4.);
       fflush(stdout);
     }
     (*results)[i]=flavortrace[i]/4.;
   }
   free(flavortrace);
   free(spacetrace);
   free(spinortrace);
   free(colortrace);
}
void vector_density_density_1234( bispinor ** propfields, int type_1234,int taudensity, _Complex double **results ){
   int ix,i;
   int f1,c1,s1;
   int spinorstart=0, spinorend=4;
   bispinor running;

   _Complex double *colortrace;
   _Complex double *spacetrace;
   _Complex double *spinortrace;
   _Complex double *flavortrace;
   int type;



   colortrace=(_Complex double *)malloc(sizeof(_Complex double) *8);
   spacetrace=(_Complex double *)malloc(sizeof(_Complex double) *8*T_global);
   spinortrace=(_Complex double *)malloc(sizeof(_Complex double)*2*T_global);
   flavortrace=(_Complex double *)malloc(sizeof(_Complex double)*T_global);

   if ( (colortrace == NULL) || (spacetrace == NULL) || (spinortrace == NULL) || (flavortrace == NULL) )
   {
     printf("Error in mem allocation in density_density_1234\n");
     exit(1);
   }
   *results=(_Complex double *)malloc(sizeof(_Complex double)*T_global);
   if (*results == NULL){
     printf("Error in vector current density\n");
     exit(1);
   }


   if ( ( type_1234 == TYPE_1 )|| ( type_1234 == TYPE_3 ) ) {
     spinorstart=0;
     spinorend  =2;
   }
   else if ( ( type_1234 == TYPE_2) || (type_1234 == TYPE_4) ){
     spinorstart=2;
     spinorend  =4;
   }
   else{
     if (g_cart_id ==0) fprintf(stdout, "Wrong arument for type_1234, it can only be TYPE_1, TYPE_2, TYPE_3, TYPE_4\n");
     exit(1);
   }


//Trace over up and down flavors
   for (i=0; i<T_global; ++i)
     flavortrace[i]=0.;
   for (f1=0; f1<2; ++f1){
//Trace over the spinor indices you have to trace only over those two spinor 
//component that appear in the final spinor
     for (i=0; i<2*T_global; ++i)
       spinortrace[i]=0.;

     for (s1= spinorstart; s1<spinorend; ++s1){
//Trace over the spatial indices
       for (i=0; i<8*T_global; ++i)
         spacetrace[i]=0.;
       for (ix = 0; ix< VOLUME; ++ix){
//Trace over the color indices for each sites
         for (i=0; i<8; ++i)
           colortrace[i]=0.;
         for (c1=0; c1<3; ++c1){

           _bispinor_null(running);

           if ( (type_1234 == TYPE_1) || (type_1234 == TYPE_2) ){

             _vector_assign(running.sp_up.s0, propfields[12*s1+4*c1+2*f1][ix].sp_up.s0);
             _vector_assign(running.sp_up.s1, propfields[12*s1+4*c1+2*f1][ix].sp_up.s1);

             _vector_assign(running.sp_dn.s0, propfields[12*s1+4*c1+2*f1][ix].sp_dn.s0);
             _vector_assign(running.sp_dn.s1, propfields[12*s1+4*c1+2*f1][ix].sp_dn.s1);


              phix_taui_commutator_bispinor( &running, taudensity, GAMMA_UP, ix );

           }
           else if ((type_1234 == TYPE_3) || ( type_1234 == TYPE_4)){

             _vector_assign(running.sp_up.s2, propfields[12*s1+4*c1+2*f1][ix].sp_up.s2);
             _vector_assign(running.sp_up.s3, propfields[12*s1+4*c1+2*f1][ix].sp_up.s3);

             _vector_assign(running.sp_dn.s2, propfields[12*s1+4*c1+2*f1][ix].sp_dn.s2);
             _vector_assign(running.sp_dn.s3, propfields[12*s1+4*c1+2*f1][ix].sp_dn.s3);
              
              phix_taui_commutator_bispinor( &running, taudensity, GAMMA_DN, ix );

           }

           multiply_backward_propagator(&running, propfields, &running, ix, NODIR );

           trace_in_color(colortrace,&running,c1);

         }  
       
         trace_in_space(spacetrace,colortrace,ix);

       } 
//Gather the results from all nodes to complete the trace in space
#if defined TM_USE_MPI
       for (i=0; i<8*T_global; ++i){
         _Complex double tmp;
        MPI_Allreduce(&spacetrace[i], &tmp, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, g_cart_grid);
        spacetrace[i]= tmp;
       }
#endif
           // delta (spinor components of spacetrace, s1) for all time slices and flavor indices
       trace_in_spinor(spinortrace, spacetrace, s1);
     }//End of trace in spinor space

     phi0_taui_commutator( spinortrace, taudensity );
     
     //delta(flavor component in spinortrace, f1) for all time slices
     trace_in_flavor( flavortrace, spinortrace, f1 );
   } //End of trace in flavor space
   type = type_1234 == TYPE_1 ? 1 : type_1234 == TYPE_2 ? 2 : type_1234 == TYPE_3 ? 3 : 4 ;
   if (g_cart_id == 0){printf("Vector Density Density correlator type (%s) for tau matrix density %d results\n", type_1234 == TYPE_1 ? "1" : type_1234 == TYPE_2 ? "2" : type_1234 == TYPE_3 ? "3" : "4", taudensity);}
   for (i=0; i<T_global; ++i){
     if (g_cart_id == 0){
      printf("VECTORDENSITY%dDENSITY%d %d %.3d %10.10e %10.10e\n", taudensity,taudensity, type, i, creal(flavortrace[i])/4.,cimag(flavortrace[i])/4.);
      fflush(stdout);
     }
     (*results)[i]=flavortrace[i]/4.;
   }
   free(flavortrace);
   free(spacetrace);
   free(spinortrace);
   free(colortrace);
}



void naivedirac_current_density_12ab( bispinor ** propfields, int type_12, int type_ab, _Complex double **results ){
   int ix,i;
   int f1,c1,s1,tauindex;
   int spinorstart=0, spinorend=4;
   su3 * restrict upm;
   bispinor running;
#if defined TM_USE_MPI
   int count;
   MPI_Status  statuses[8];
   MPI_Request *request;
   request=( MPI_Request *) malloc(sizeof(MPI_Request)*8);
#endif
   _Complex double *colortrace;
   _Complex double *spacetrace;
   _Complex double *spinortrace;
   _Complex double *flavortrace;
   _Complex double *paulitrace;

   colortrace=(_Complex double *)malloc(sizeof(_Complex double) *8);
   spacetrace=(_Complex double *)malloc(sizeof(_Complex double) *8*T_global);
   spinortrace=(_Complex double *)malloc(sizeof(_Complex double)*2*T_global);
   flavortrace=(_Complex double *)malloc(sizeof(_Complex double)*T_global);
   paulitrace= (_Complex double *)malloc(sizeof(_Complex double)*T_global);

   if ( (colortrace == NULL) || (spacetrace == NULL) || (spinortrace == NULL) || (flavortrace == NULL) || (paulitrace == NULL) )
   {
     printf("Error in mem allocation in naivedirac_current_density_12ab\n");
     exit(1);
   }
   *results=(_Complex double *)malloc(sizeof(_Complex double)*T_global*4);
   if ( *results == NULL){
     printf("Not enough memory for results in current density naive\n");
     exit(1);
   }
   if ( type_ab == TYPE_A ) {
     spinorstart=0;
     spinorend  =2;
   }
   else if ( type_ab == TYPE_B ){
     spinorstart=2;
     spinorend  =4;
   }
   else{
    if (g_cart_id == 0){
      fprintf(stdout, "Wrong argument for type_ab, it can only be TYPE_A or TYPE_B\n");exit(1);}
   }


//Doing the neccessary communication
#if defined TM_USE_MPI
   for (s1=spinorstart; s1<spinorend; ++s1)
      for (c1=0; c1<3; ++c1)
         for (f1=0; f1<2; ++f1){
            count=0;
            generic_exchange_direction_nonblocking( propfields[12*s1 + 4*c1 + 2*f1 + 0], sizeof(bispinor), TDOWN, request, &count );
            MPI_Waitall( count, request, statuses);
            count=0;
            generic_exchange_direction_nonblocking( propfields[12*s1 + 4*c1 + 2*f1 + 1], sizeof(bispinor), TDOWN, request, &count);
            MPI_Waitall( count, request, statuses);
         }
   free(request);
#endif

//Trace over the Pauli matrices
   for (i=0; i<T_global; ++i)
      paulitrace[i]=0.;

   for (tauindex=0; tauindex<3; ++tauindex){
      for (i=0; i<T_global; ++i)
         flavortrace[i]=0.;

      for (f1=0; f1<2; ++f1){

//Trace over the spinor indices
         for (i=0; i<2*T_global; ++i)
            spinortrace[i]=0.;

         for (s1= spinorstart; s1<spinorend; ++s1){

//Trace over the spatial indices
            for (i=0; i<8*T_global; ++i)
               spacetrace[i]=0.;
            for (ix = 0; ix< VOLUME; ++ix){

//Trace over the color indices for each sites

               for (i=0; i<8; ++i)
                  colortrace[i]=0.;
               for (c1=0; c1<3; ++c1){
/*   
       TYPE  IA OR  IB     U0(x-0)*       (1-g5)/2*S(x  ,ytilde) fixed indices (c1, s1, f1)
       TYPE IIA OR IIB     U0^dagger(x-0)*(1-g5)/2*S(x-0,ytilde) running indices bispinor
*/
                  upm = &g_gauge_field[g_idn[ix][TUP]][TUP];

//for the up quark
                  _vector_null( running.sp_up.s0 );
                  _vector_null( running.sp_up.s1 );

                  if  ( type_12 == TYPE_I ){
                    _su3_multiply( running.sp_up.s2, (*upm), propfields[12*s1+4*c1+2*f1][ix].sp_up.s2 );
                    _su3_multiply( running.sp_up.s3, (*upm), propfields[12*s1+4*c1+2*f1][ix].sp_up.s3 );


                    _complex_times_vector(running.sp_up.s2,phase_0,running.sp_up.s2);
                    _complex_times_vector(running.sp_up.s3,phase_0,running.sp_up.s3);
                  }
                  else if ( type_12 == TYPE_II ){
                    _su3_inverse_multiply( running.sp_up.s2, (*upm), propfields[12*s1+4*c1+2*f1][g_idn[ix][TUP]].sp_up.s2 );
                    _su3_inverse_multiply( running.sp_up.s3, (*upm), propfields[12*s1+4*c1+2*f1][g_idn[ix][TUP]].sp_up.s3 );

                    _complexcjg_times_vector(running.sp_up.s2,phase_0,running.sp_up.s2);
                    _complexcjg_times_vector(running.sp_up.s3,phase_0,running.sp_up.s3);

                  }


//for the dn quark
                  _vector_null( running.sp_dn.s0 );
                  _vector_null( running.sp_dn.s1 );
                  if  ( type_12 == TYPE_I ){
                    _su3_multiply( running.sp_dn.s2, (*upm), propfields[12*s1+4*c1+2*f1][ix].sp_dn.s2 );
                    _su3_multiply( running.sp_dn.s3, (*upm), propfields[12*s1+4*c1+2*f1][ix].sp_dn.s3 );

                    _complex_times_vector(running.sp_dn.s2,phase_0,running.sp_dn.s2);
                    _complex_times_vector(running.sp_dn.s3,phase_0,running.sp_dn.s3);

                  }
                  else if ( type_12 == TYPE_II ){
                    _su3_inverse_multiply( running.sp_dn.s2, (*upm), propfields[12*s1+4*c1+2*f1][g_idn[ix][TUP]].sp_dn.s2 );
                    _su3_inverse_multiply( running.sp_dn.s3, (*upm), propfields[12*s1+4*c1+2*f1][g_idn[ix][TUP]].sp_dn.s3 );

                    _complexcjg_times_vector(running.sp_dn.s2,phase_0,running.sp_dn.s2);
                    _complexcjg_times_vector(running.sp_dn.s3,phase_0,running.sp_dn.s3);

                  }



/*   
       TYPE  IA OR  IB     gamma0*U0(x-0)*       (1-g5)/2*S(x  ,ytilde)
       TYPE IIA OR IIB     gamma0*U0^dagger(x-0)*(1-g5)/2*S(x-0,ytilde)
*/
                  _vector_add_assign(running.sp_up.s0, running.sp_up.s2);
                  _vector_add_assign(running.sp_up.s1, running.sp_up.s3);
                  _vector_null(running.sp_up.s2);
                  _vector_null(running.sp_up.s3);

                  _vector_add_assign(running.sp_dn.s0, running.sp_dn.s2);
                  _vector_add_assign(running.sp_dn.s1, running.sp_dn.s3);
                  _vector_null(running.sp_dn.s2);
                  _vector_null(running.sp_dn.s3);

/*   
       TYPE  IA OR  IB     tau_i*gamma0*U0(x-0)*       (1-g5)/2*S(x  ,ytilde)
       TYPE IIA OR IIB     tau_i*gamma0*U0^dagger(x-0)*(1-g5)/2*S(x-0,ytilde)
*/
                  taui_spinor( &running, &running, tauindex  );

/*   
       TYPE  IA OR  IB     S(ytilde, x-0)* tau_i*gamma0*U0(x-0)*       (1-g5)/2*S(x  ,ytilde)
       TYPE IIA OR IIB     S(ytilde, x  )* tau_i*gamma0*U0^dagger(x-0)*(1-g5)/2*S(x-0,ytilde)
*/
                  if ( type_12 == TYPE_I ){
                    multiply_backward_propagator(&running, propfields, &running, ix, TDOWN);
                  }
                  else if ( type_12 == TYPE_II ){
                    multiply_backward_propagator(&running, propfields, &running, ix,NODIR);
                  }
                  //delta( color component of bispinor running, c1) for all spinor and flavor indices
                  trace_in_color(colortrace,&running,c1);

               }  //End of trace color
               //sum over all lattice sites the result of the color trace
               trace_in_space(spacetrace,colortrace,ix);

            } //End of trace space

//Gather the results from all nodes to complete the trace in space
#if defined TM_USE_MPI
            for (i=0; i<8*T_global; ++i){
               _Complex double tmp;
               MPI_Allreduce(&spacetrace[i], &tmp, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, g_cart_grid);
               spacetrace[i]= tmp;
            }
#endif
            // delta (spinor components of spacetrace, s1) for all time slices and flavor components
            trace_in_spinor(spinortrace, spacetrace, s1);

         }//End of trace in spinor space

/*   
       TYPE  IA tau_i*phi(ytilde)        *  (1+gamma5)/2  *   S(ytilde, x-0)*   tau_i*gamma0*U0(x-0)*       (1-g5)/2*   S(x  ,ytilde)
       TYPE  IB phi ^dagger(ytilde)*tau_i *  (1-gamma5)/2  *   S(ytilde, x-0)*   tau_i*gamma0*U0(x-0)*       (1-g5)/2*   S(x  ,ytilde)

       TYPE IIA tau_i*phi(ytilde)        *  (1+gamma5)/2  *   S(ytilde, x  )*   tau_i*gamma0*U0^dagger(x-0)*(1-g5)/2*   S(x-0,ytilde)
       TYPE IIB phi^dagger(ytilde)*tau_i *  (1-gamma5)/2  *   S(ytilde, x  )*   tau_i*gamma0*U0^dagger(x-0)*(1-g5)/2*   S(x-0,ytilde)

*/
         if ( type_ab == TYPE_A ){
           taui_scalarfield_flavoronly( spinortrace, tauindex, NO_DAGG, LEFT );
         }
         else if ( type_ab == TYPE_B){
           taui_scalarfield_flavoronly( spinortrace, tauindex, DAGGER, LEFT  );
         }
         //delta(flavor component in spinortrace, f1) for all time slices 
         trace_in_flavor( flavortrace, spinortrace, f1 );
      } //End of trace in flavor space
      //sum for all Pauli matrices
      if (g_cart_id == 0){printf("NaiveDirac Current Density correlator type (%s %s) for tau matrixes %d results\n", type_12 == TYPE_I ? "I" : "II",type_ab == TYPE_A ? "a" :"b", tauindex);}
      for (i=0; i<T_global; ++i){
        if (g_cart_id == 0){
          printf("DCDTAU%dTAU%d %d %d %.3d %10.10e %10.10e\n",tauindex, tauindex,type_12, type_ab,  i, creal(flavortrace[i])/4.,cimag(flavortrace[i])/4.0);
        }
        (*results)[i+tauindex*T_global]=flavortrace[i]/4.;
      }
      for (i=0;i<T_global; ++i)
         paulitrace[i]+=flavortrace[i];
   } //End of trace for Pauli matrices

   if (g_cart_id == 0){printf( "NaiveDirac Current Density correlator type (%s %s) results\n", type_12 == TYPE_I ? "I" : "II",type_ab == TYPE_A ? "a" :"b");}
   for (i=0; i<T_global; ++i){
      if (g_cart_id == 0){
        printf("DCD %d %d %.3d %10.10e %10.10e\n",type_12, type_ab,  i, creal(paulitrace[i])/4.,cimag(paulitrace[i])/4.0);
      }
      (*results)[i+tauindex*T_global]=paulitrace[i]/4.;
   }
   free(flavortrace);
   free(paulitrace);
   free(spacetrace);
   free(spinortrace);
   free(colortrace);

}

void vector_axial_current_current_1234( bispinor ** propfields_source_zero, bispinor ** propfields_source_ntmone, int type_1234, int taucurrent, int vectororaxial, _Complex double **results ){
   int ix,i;
   int f1,c1,s1;
   int spinorstart=0, spinorend=4;
   bispinor running;

   _Complex double *colortrace;
   _Complex double *spacetrace;
   _Complex double *spinortrace;
   _Complex double *flavortrace;
   int type;
   su3 untminusonet;

#if defined TM_USE_MPI
   int count;
   MPI_Status  statuses[8];
   MPI_Request *request;
   request=( MPI_Request *) malloc(sizeof(MPI_Request)*8);
#endif

   untminusonet.c00=g_gauge_field[g_idn[0][TUP]][TUP].c00;
   untminusonet.c01=g_gauge_field[g_idn[0][TUP]][TUP].c01;
   untminusonet.c02=g_gauge_field[g_idn[0][TUP]][TUP].c02;
   untminusonet.c10=g_gauge_field[g_idn[0][TUP]][TUP].c10;
   untminusonet.c11=g_gauge_field[g_idn[0][TUP]][TUP].c11;
   untminusonet.c12=g_gauge_field[g_idn[0][TUP]][TUP].c12;
   untminusonet.c20=g_gauge_field[g_idn[0][TUP]][TUP].c20;
   untminusonet.c21=g_gauge_field[g_idn[0][TUP]][TUP].c21;
   untminusonet.c22=g_gauge_field[g_idn[0][TUP]][TUP].c22;

#if defined TM_USE_MPI
   MPI_Bcast(&untminusonet.c00, 1, MPI_DOUBLE_COMPLEX, 0, g_cart_grid);
   MPI_Bcast(&untminusonet.c01, 1, MPI_DOUBLE_COMPLEX, 0, g_cart_grid);
   MPI_Bcast(&untminusonet.c02, 1, MPI_DOUBLE_COMPLEX, 0, g_cart_grid);
   MPI_Bcast(&untminusonet.c10, 1, MPI_DOUBLE_COMPLEX, 0, g_cart_grid);
   MPI_Bcast(&untminusonet.c11, 1, MPI_DOUBLE_COMPLEX, 0, g_cart_grid);
   MPI_Bcast(&untminusonet.c12, 1, MPI_DOUBLE_COMPLEX, 0, g_cart_grid);
   MPI_Bcast(&untminusonet.c20, 1, MPI_DOUBLE_COMPLEX, 0, g_cart_grid);
   MPI_Bcast(&untminusonet.c21, 1, MPI_DOUBLE_COMPLEX, 0, g_cart_grid);
   MPI_Bcast(&untminusonet.c22, 1, MPI_DOUBLE_COMPLEX, 0, g_cart_grid);
#endif


   colortrace=(_Complex double *)malloc(sizeof(_Complex double) *8);
   spacetrace=(_Complex double *)malloc(sizeof(_Complex double) *8*T_global);
   spinortrace=(_Complex double *)malloc(sizeof(_Complex double)*2*T_global);
   flavortrace=(_Complex double *)malloc(sizeof(_Complex double)*T_global);

   if ( (colortrace == NULL) || (spacetrace == NULL) || (spinortrace == NULL) || (flavortrace == NULL) )
   {
     printf("Error in mem allocation in density_density_1234\n");
     exit(1);
   }
   *results=(_Complex double *)malloc(sizeof(_Complex double)*T_global);
   if (*results == NULL){
     printf("Error in vector current density\n");
     exit(1);
   }
   spinorstart=0;
   spinorend=4;


//Doing the neccessary communication
#if defined TM_USE_MPI
   for (s1=spinorstart; s1<spinorend; ++s1)
      for (c1=0; c1<3; ++c1)
         for (f1=0; f1<2; ++f1){
            count=0;
            generic_exchange_direction_nonblocking( propfields_source_ntmone[12*s1 + 4*c1 + 2*f1 + 0], sizeof(bispinor), TDOWN, request, &count );
            MPI_Waitall( count, request, statuses);
            count=0;
            generic_exchange_direction_nonblocking( propfields_source_zero[12*s1 + 4*c1 + 2*f1 + 0]  , sizeof(bispinor), TDOWN, request, &count );
            MPI_Waitall( count, request, statuses);
            count=0;
            generic_exchange_direction_nonblocking( propfields_source_ntmone[12*s1 + 4*c1 + 2*f1 + 1], sizeof(bispinor), TDOWN, request, &count );
            MPI_Waitall( count, request, statuses);
            count=0;
            generic_exchange_direction_nonblocking( propfields_source_zero[12*s1 + 4*c1 + 2*f1 + 1],   sizeof(bispinor), TDOWN, request, &count );
            MPI_Waitall( count, request, statuses);
         }
   free(request);
#endif
//Trace over up and down flavors
   for (i=0; i<T_global; ++i)
     flavortrace[i]=0.;
   for (f1=0; f1<2; ++f1){
//Trace over the spinor indices you have to trace only over those two spinor 
//component that appear in the final spinor
     for (i=0; i<2*T_global; ++i)
       spinortrace[i]=0.;
     for (s1= spinorstart; s1<spinorend; ++s1){
//Trace over the spatial indices
       for (i=0; i<8*T_global; ++i)
         spacetrace[i]=0.;
       for (ix = 0; ix< VOLUME; ++ix){
//Trace over the color indices for each sites
         for (i=0; i<8; ++i)
           colortrace[i]=0.;
         for (c1=0; c1<3; ++c1){
           _bispinor_null(running);
           if ( type_1234 == TYPE_1 ){
             bispinor_mult_su3matrix( &running, &propfields_source_ntmone[12*s1+4*c1+2*f1][ix], &g_gauge_field[g_idn[ix][TUP]][TUP], NO_DAGG);
           }
           else if ( type_1234 == TYPE_2 ){
             bispinor_mult_su3matrix( &running, &propfields_source_ntmone[12*s1+4*c1+2*f1][g_idn[ix][TUP]], &g_gauge_field[g_idn[ix][TUP]][TUP], DAGGER);
           }
           else if ( type_1234 == TYPE_3 ){
             bispinor_mult_su3matrix( &running, &propfields_source_zero[12*s1+4*c1+2*f1][ix],   &g_gauge_field[g_idn[ix][TUP]][TUP], NO_DAGG);
           }
           else if ( type_1234 == TYPE_4 ){
             bispinor_mult_su3matrix( &running, &propfields_source_zero[12*s1+4*c1+2*f1][g_idn[ix][TUP]],  &g_gauge_field[g_idn[ix][TUP]][TUP], DAGGER);
           }
           else {
              if (g_cart_id == 0){
                printf("Wrong type index in current current correlator\n");
                exit(1);
              }
           }
           if (vectororaxial == 1){
            bispinor_timesgamma5(&running);
           }
//Multiplication with gamma0
           bispinor_timesgamma0(&running);

//Multiplication with tau_i input parameter for the current
           bispinor_taui(&running, taucurrent);

//Backward propagator multiplication
           if ( type_1234 == TYPE_1 ){
              multiply_backward_propagator(&running, propfields_source_zero, &running, ix, TDOWN );
              bispinor_mult_su3matrix( &running, &running, &untminusonet, NO_DAGG);
           }
           else if ( type_1234 == TYPE_2 ){
              multiply_backward_propagator(&running, propfields_source_zero, &running, ix, NODIR );
              bispinor_mult_su3matrix( &running, &running, &untminusonet, NO_DAGG);
           }
           else if ( type_1234 == TYPE_3 ){
              multiply_backward_propagator(&running, propfields_source_ntmone, &running, ix, TDOWN );
              bispinor_mult_su3matrix( &running, &running, &untminusonet, DAGGER);
           }
           else if ( type_1234 == TYPE_4 ){
              multiply_backward_propagator(&running, propfields_source_ntmone, &running, ix, NODIR );
              bispinor_mult_su3matrix( &running, &running, &untminusonet, DAGGER);
           }

           if (vectororaxial == 1){
            bispinor_timesgamma5(&running);
           }
//Multiplication with gamma0
           bispinor_timesgamma0(&running);

//Multiplication with tau_i input parameter for the current
           bispinor_taui(&running, taucurrent);

           trace_in_color(colortrace,&running,c1);
         }  //End of trace color
         trace_in_space(spacetrace,colortrace,ix);
       } //End of trace space
//Gather the results from all nodes to complete the trace in space
#if defined TM_USE_MPI
       for (i=0; i<8*T_global; ++i){
         _Complex double tmp;
        MPI_Allreduce(&spacetrace[i], &tmp, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, g_cart_grid);
        spacetrace[i]= tmp;
       }
#endif
           // delta (spinor components of spacetrace, s1) for all time slices and flavor indices
       trace_in_spinor(spinortrace, spacetrace, s1);
     }//End of trace in spinor space
     //delta(flavor component in spinortrace, f1) for all time slices
     trace_in_flavor( flavortrace, spinortrace, f1 );
   } //End of trace in flavor space
   type = type_1234 == TYPE_1 ? 1 : type_1234 == TYPE_2 ? 2 : type_1234 == TYPE_3 ? 3 : 4 ;
   if (g_cart_id == 0){printf("%s current current correlator type (%s) for tau matrix current %d results\n", vectororaxial==1 ? "Axial" : "Vector", type_1234 == TYPE_1 ? "1" : type_1234 == TYPE_2 ? "2" : type_1234 == TYPE_3 ? "3" : "4", taucurrent);}
   for (i=0; i<T_global; ++i){
     if (g_cart_id == 0){
       printf("%sCURRENT%dCURRENT%d %d %.3d %10.10e %10.10e\n", vectororaxial==1 ? "AXIAL" : "VECTOR", taucurrent,taucurrent, type, i, creal(flavortrace[i])/4.,cimag(flavortrace[i])/4.);
       fflush(stdout);
     }
     (*results)[i]=flavortrace[i]/4.;
   }
   free(flavortrace);
   free(spacetrace);
   free(spinortrace);
   free(colortrace);
}
#endif
