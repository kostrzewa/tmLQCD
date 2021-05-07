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
#include "su3spinor.h"
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


void wilsonterm_current_density_312ab( bispinor ** propfields, int type_12, int type_ab, _Complex double **results ){
   int ix,i;
   int f1,c1,s1,tauindex;
   int spinorstart=0, spinorend=4;
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

   colortrace= (_Complex double *)malloc(sizeof(_Complex double) *8);
   spacetrace= (_Complex double *)malloc(sizeof(_Complex double) *8*T_global);
   spinortrace=(_Complex double *)malloc(sizeof(_Complex double)*2*T_global);
   flavortrace=(_Complex double *)malloc(sizeof(_Complex double)*T_global);
   paulitrace= (_Complex double *)malloc(sizeof(_Complex double)*T_global);

   if ( (colortrace == NULL) || (spacetrace == NULL) || (spinortrace == NULL) || (flavortrace == NULL) || (paulitrace == NULL) )
   {
     printf("Error in mem allocation in wilsonterm_current_density_312ab\n");
     exit(1);
   }
   *results=(_Complex double *)malloc(sizeof(_Complex double)*4*T_global);
   if (*results == NULL){
      printf("Not enough memory for results in current density three\n");
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
      if (g_cart_id == 0){fprintf(stdout,"Wrong argument for type_1234, it can only be TYPE_1, TYPE_2,  TYPE_3 or TYPE_4 \n"); exit(1);}
   }


// Doing the neccessary communication
#if defined TM_USE_MPI
   for (s1=spinorstart; s1<spinorend; ++s1)
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
//Trace over the Pauli matrices
   for (i=0; i<T_global; ++i){
      paulitrace[i]=0.;
   }
   for (tauindex= 0; tauindex <3; ++tauindex){
//Trace over flavour degrees of freedom
      for (i=0; i<T_global; ++i)
         flavortrace[i]=0.;

      for (f1=0; f1<2; ++f1){

//Trace over spinor indices
         for (i=0; i<2*T_global; ++i){
            spinortrace[i]=0.;
         }

         for (s1=spinorstart; s1<spinorend; ++s1){

//Trace over spatial indices
            for (i=0; i<8*T_global; ++i){
               spacetrace[i]=0.;
            }
            for (ix=0; ix<VOLUME; ++ix){

//Trace over the color indices for each sites
               for (i=0; i<8; ++i)
                  colortrace[i]=0.;
               for (c1=0; c1<3; ++c1){
/*   
       TYPE III.1.a OR  III.1.b     U0(x-0)*U0(x)* (1+gamma5)/2 *  S(x+0,ytilde)
       TYPE III.2.a OR  III.2.b                    (1+gamma5)/2 *  S(x-0,ytilde)
*/
                  _bispinor_null(running);
                  
                  if ( type_12 == TYPE_1){
                    bispinor_spinup_mult_su3matrix( &running, &propfields[12*s1 + 4*c1 + 2*f1][g_iup[ix][TUP]], &g_gauge_field[ix][TUP], NO_DAGG);

                    bispinor_spinup_mult_su3matrix( &running, &running, &g_gauge_field[g_idn[ix][TUP]][TUP], NO_DAGG);

                  }
                  else if ( type_12 == TYPE_2){

                    _vector_assign( running.sp_up.s0, propfields[12*s1 + 4*c1 + 2*f1][g_idn[ix][TUP]].sp_up.s0 );
                    _vector_assign( running.sp_up.s1, propfields[12*s1 + 4*c1 + 2*f1][g_idn[ix][TUP]].sp_up.s1 );
                    _vector_assign( running.sp_dn.s0, propfields[12*s1 + 4*c1 + 2*f1][g_idn[ix][TUP]].sp_dn.s0 );
                    _vector_assign( running.sp_dn.s1, propfields[12*s1 + 4*c1 + 2*f1][g_idn[ix][TUP]].sp_dn.s1 );
                  }
/*   
       TYPE III.1.a OR  III.1.b     tau_i*phi(x)*U0(x-0)*U0(x)* (1+gamma5)/2 *  S(x+0,ytilde)
       TYPE III.2.a OR  III.2.b     tau_i*phi(x)*               (1+gamma5)/2 *  S(x-0,ytilde)
*/
                  taui_scalarfield_spinor( &running, &running, GAMMA_UP, tauindex, ix, NODIR, NO_DAGG);

/*   
       TYPE III.1.a OR  III.1.b     S(ytilde, x-0)*tau_i*phi(x)*U0(x-0)*U0(x)* (1+gamma5)/2 *  S(x+0,ytilde)
       TYPE III.2.a OR  III.2.b     S(ytilde, x-0)*tau_i*phi(x)*               (1+gamma5)/2 *  S(x-0,ytilde)
*/
                  multiply_backward_propagator(&running, propfields, &running, ix, TDOWN);

                  //delta( color component of bispinor running, c1) for all spinor and flavor indices                  
                  trace_in_color(colortrace, &running, c1 );
               } //End of trace color
               //sum over all lattice sites the result of the color trace
               trace_in_space( spacetrace, colortrace, ix);
            }  //End of trace in space

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

         } //End of trace in spinor space
         
/*   
       TYPE III.1.a                 tau_i*phi(ytilde)*       (1+gamma5)/2* S(ytilde, x-0)*tau_i*phi(x)*U0(x-0)*U0(x)* (1+gamma5)/2 *  S(x+0,ytilde)
       TYPE III.1.b                 phi^dagger(ytilde)*tau_i*(1-gamma5)/2* S(ytilde, x-0)*tau_i*phi(x)*U0(x-0)*U0(x)* (1+gamma5)/2 *  S(x+0,ytilde)

       TYPE III.2.a                 tau_i*phi(ytilde)*       (1+gamma5)/2* S(ytilde, x-0)*tau_i*phi(x)*               (1+gamma5)/2 *  S(x-0,ytilde)
       TYPE III.2.b                 phi^dagger(ytilde)*tau_i*(1-gamma5)/2* S(ytilde, x-0)*tau_i*phi(x)*               (1+gamma5)/2 *  S(x-0,ytilde)

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
      for (int ii=0; ii<T_global; ++ii){
        (*results)[ii+tauindex*T_global]=flavortrace[ii]/4.;
      }
      for (i=0;i<T_global; ++i)
         paulitrace[i]+=flavortrace[i];
   } //End of trace for Pauli matrices

 
   if (g_cart_id == 0){printf("Wilson term Dirac Current Density correlator typeIII results= %s %s\n", type_12 == TYPE_1 ? "1" : "2",type_ab == TYPE_A ? "a" :"b");}
   for (i=0; i<T_global; ++i){
      if (g_cart_id == 0){
        printf("WCDPR1 %d %d %.3d %10.10e %10.10e\n", type_12, type_ab, i, creal(paulitrace[i])/4., cimag(paulitrace[i])/4.);
      }
      for (int ii=0; ii<T_global; ++ii){
        (*results)[ii+3*T_global]=paulitrace[ii]/4.;
      }
   }
   free(flavortrace);
   free(paulitrace);
   free(spacetrace);
   free(spinortrace);
   free(colortrace);
}

void wilsonterm_current_density_412ab( bispinor ** propfields, int type_12, int type_ab, _Complex double **results ){
   int ix,i;
   int f1,c1,s1,tauindex;
   int spinorstart=0, spinorend=4;
   bispinor **starting2d;
   bispinor running;
   su3 * restrict upm;
#if defined TM_USE_MPI
   int count;
   MPI_Status  statuses[8];
   MPI_Request *request;
#endif
   _Complex double *colortrace;
   _Complex double *spacetrace;
   _Complex double *spinortrace;
   _Complex double *flavortrace;
   _Complex double *paulitrace;

   colortrace= (_Complex double *)malloc(sizeof(_Complex double)*8);
   spacetrace= (_Complex double *)malloc(sizeof(_Complex double)*8*T_global);
   spinortrace=(_Complex double *)malloc(sizeof(_Complex double)*2*T_global);
   flavortrace=(_Complex double *)malloc(sizeof(_Complex double)*T_global);
   paulitrace= (_Complex double *)malloc(sizeof(_Complex double)*T_global);

   if ( (colortrace == NULL) || (spacetrace == NULL) || (spinortrace == NULL) || (flavortrace == NULL) || (paulitrace == NULL) )
   {
     printf("Error in mem allocation\n");
     exit(1);
   }
   *results=(_Complex double *)malloc(sizeof(_Complex double)*4*T_global);
   if ( *results == NULL ){
     printf("Not enough memory in wilson current density 4\n");
     exit(1);
   }
#if defined TM_USE_MPI
   request=( MPI_Request *) malloc(sizeof(MPI_Request)*8);
#endif
   if ( type_ab == TYPE_A ) {
        spinorstart=0;
        spinorend  =2;
   }
   else if ( type_ab == TYPE_B ){
        spinorstart=2;
        spinorend  =4;
   }
   else{
       if (g_cart_id == 0) fprintf(stdout,"Wrong argument for type_1234, it can only be TYPE_1, TYPE_2,  TYPE_3 or TYPE_4 \n");
       exit(1);
   }

   if (type_12 == TYPE_1){
     for (i=0; i<T_global; ++i)
       paulitrace[i]=0.;
     for (tauindex=0; tauindex<3; ++tauindex){
       for (i=0; i<T_global; ++i)
         flavortrace[i]=0.;
       for (f1=0; f1<2; ++f1){
         for (i=0; i<2*T_global; ++i)
           spinortrace[i]=0.;
         for (s1= spinorstart; s1<spinorend; ++s1){
           for (i=0; i<8*T_global; ++i)
             spacetrace[i]=0.;
           for (ix = 0; ix< VOLUME; ++ix){
             for (i=0; i<8; ++i)
               colortrace[i]=0.;
             for (c1=0; c1<3; ++c1){
               _vector_null( running.sp_up.s2 );
               _vector_null( running.sp_up.s3 );
               _vector_assign( running.sp_up.s0, propfields[12*s1+4*c1+2*f1][ix].sp_up.s0 );
               _vector_assign( running.sp_up.s1, propfields[12*s1+4*c1+2*f1][ix].sp_up.s1 );
               _vector_null( running.sp_dn.s2 );
               _vector_null( running.sp_dn.s3 );
               _vector_assign( running.sp_dn.s0, propfields[12*s1+4*c1+2*f1][ix].sp_dn.s0 );
               _vector_assign( running.sp_dn.s1, propfields[12*s1+4*c1+2*f1][ix].sp_dn.s1 );

               taui_scalarfield_spinor( &running, &running, GAMMA_UP, tauindex, ix, TDOWN, NO_DAGG );

               multiply_backward_propagator(&running, propfields, &running, ix, NODIR );

               trace_in_color(colortrace,&running,c1);
             }  //End of trace color
              //sum over all lattice sites the result of the color trace
             trace_in_space(spacetrace,colortrace,ix);
           } //End of trace space
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
         if ( type_ab == TYPE_A ){
           taui_scalarfield_flavoronly( spinortrace, tauindex, NO_DAGG, LEFT);
         }
         else if ( type_ab == TYPE_B ){
           taui_scalarfield_flavoronly( spinortrace, tauindex, DAGGER, LEFT  );
         }
         trace_in_flavor( flavortrace, spinortrace, f1 );
       } //End of trace in flavor space
      //sum for all Pauli matrices
       for (i=0;i<T_global; ++i){
         paulitrace[i]+=flavortrace[i];
         (*results)[i+tauindex*T_global] = flavortrace[i]/4.;
       }
     } //End of trace for Pauli matrices
     if  (g_cart_id == 0){printf("Wilson term Dirac Current Density correlator typeIV results= %s %s\n", "1",type_ab == TYPE_A ? "a" :"b");}
     for (i=0; i<T_global; ++i){
       if (g_cart_id == 0){
        printf("WCDPR2 %d %d %.3d %10.10e %10.10e\n", type_12, type_ab, i, creal(paulitrace[i])/4., cimag(paulitrace[i])/4.);
       }
       (*results)[i+3*T_global]=paulitrace[i]/4.;
     }
   }
   if (type_12 == TYPE_2 ){
      starting2d=(bispinor **)malloc(sizeof(bispinor *)*3);
      if (starting2d == NULL){
        if (g_cart_id == 0){
          printf("Memory allocation failure in extended current density contractions IV\n");
          exit(1);
        }
      }
      for (i=0; i<3; ++i){
        starting2d[i] =(bispinor *)malloc(sizeof(bispinor)*VOLUMEPLUSRAND);
        if (starting2d[i] == NULL){
          if (g_cart_id == 0) printf("Memory allocation error starting2d VI\n");
          exit(1);
        }
        for (ix=0; ix<VOLUME; ++ix){
          _bispinor_null(starting2d[i][ix]);
        }
      }
//Doing the neccesary communication
#if defined TM_USE_MPI
      for (s1=spinorstart; s1<spinorend; ++s1)
        for (c1=0; c1<3; ++c1)
           for (f1=0; f1<2; ++f1){
             count=0;
             generic_exchange_direction_nonblocking( propfields[12*s1 + 4*c1 + 2*f1 + 0], sizeof(bispinor), TDOWN   , request, &count );
             MPI_Waitall( count, request, statuses);
             count=0;
             generic_exchange_direction_nonblocking( propfields[12*s1 + 4*c1 + 2*f1 + 1], sizeof(bispinor), TUP     , request, &count );
             MPI_Waitall( count, request, statuses);
          }
#endif
      for (i=0; i<T_global; ++i)
        paulitrace[i]=0.;
      for (tauindex=0; tauindex<3; ++tauindex){
        for (i=0; i<T_global; ++i)
          flavortrace[i]=0.;
        for (f1=0; f1<2; ++f1){
          for (i=0; i<2*T_global; ++i)
            spinortrace[i]=0.;
          for (s1= spinorstart; s1<spinorend; ++s1){
            for (i=0; i<8*T_global; ++i)
              spacetrace[i]=0.;
            for (ix = 0; ix< VOLUME; ++ix){
              for (i=0; i<8; ++i)
                colortrace[i]=0.;
              for (c1=0; c1<3; ++c1){

                _bispinor_null(starting2d[c1][ix]);

                bispinor_spinup_mult_su3matrix( &starting2d[c1][ix], &propfields[12*s1 + 4*c1 + 2*f1][g_idn[ix][TUP]], &g_gauge_field[g_idn[ix][TUP]][TUP], DAGGER );

                bispinor_spinup_mult_su3matrix( &starting2d[c1][ix], &starting2d[c1][ix], &g_gauge_field[ix][TUP], DAGGER );

                taui_scalarfield_spinor( &starting2d[c1][ix], &starting2d[c1][ix], GAMMA_UP, tauindex, ix, NODIR, NO_DAGG );
                
                multiply_backward_propagator(&starting2d[c1][ix], propfields, &starting2d[c1][ix], ix, TUP);
              }
            }
#if defined TM_USE_MPI
            for (c1=0; c1<3; ++c1){
              count=0;
              generic_exchange_direction_nonblocking( starting2d[c1], sizeof(bispinor), TDOWN, request, &count );
              MPI_Waitall( count, request, statuses);
            }
#endif
            for (ix=0; ix<VOLUME; ++ix){
              for (i=0; i<8; ++i)
                colortrace[i]=0.;
              for (c1=0; c1<3; ++c1)
                trace_in_color(colortrace,&starting2d[c1][g_idn[ix][TUP]],c1);
              trace_in_space(spacetrace,colortrace,ix);
            }
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
          if ( type_ab == TYPE_A ){
            taui_scalarfield_flavoronly( spinortrace, tauindex, NO_DAGG, LEFT );
          }
          else if ( type_ab == TYPE_B ){
            taui_scalarfield_flavoronly( spinortrace, tauindex, DAGGER, LEFT  );
          }
          trace_in_flavor( flavortrace, spinortrace, f1 );
        } //End of trace in flavor space
      //sum for all Pauli matrices
        for (i=0;i<T_global; ++i){
          paulitrace[i]+=flavortrace[i];
          (*results)[i+tauindex*T_global]=flavortrace[i]/4.;
        }
      } //End of trace for Pauli matrices
      if  (g_cart_id == 0){printf("Wilson term Dirac Current Density correlator typeIV results= %s %s\n", type_12 == TYPE_1 ? "1" : "2",type_ab == TYPE_A ? "a" :"b");}
      for (i=0; i<T_global; ++i){
        if (g_cart_id == 0){
          printf("WCDPR2 %d %d %.3d %10.10e %10.10e\n", type_12, type_ab, i, creal(paulitrace[i])/4., cimag(paulitrace[i])/4.);
        }          
        (*results)[i+3*T_global]=paulitrace[i]/4.;
      }
      for (i=0; i<3; ++i){
        free(starting2d[i]);
      }
      free(starting2d);
   }
   free(flavortrace);
   free(paulitrace);
   free(spacetrace);
   free(spinortrace);
   free(colortrace);
#if defined TM_USE_MPI
   if (type_12 == TYPE_2)
     free(request);
#endif
}

void wilsonterm_current_density_512ab( bispinor ** propfields, int type_12, int type_ab, _Complex  double **results ){
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

   colortrace= (_Complex double *)malloc(sizeof(_Complex double)*8);
   spacetrace= (_Complex double *)malloc(sizeof(_Complex double)*8*T_global);
   spinortrace=(_Complex double *)malloc(sizeof(_Complex double)*2*T_global);
   flavortrace=(_Complex double *)malloc(sizeof(_Complex double)*T_global);
   paulitrace= (_Complex double *)malloc(sizeof(_Complex double)*T_global);

   if ( (colortrace == NULL) || (spacetrace == NULL) || (spinortrace == NULL) || (flavortrace == NULL) || (paulitrace == NULL) )
   {
     printf("Error in mem allocation\n");
     exit(1);
   }
   *results=(_Complex double *)malloc(sizeof(_Complex double)*4*T_global);
   if (*results == NULL){
     printf("not enough memory in current density five\n");
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
     if (g_cart_id == 0) fprintf(stdout,"Wrong argument for type_1234, it can only be TYPE_1, TYPE_2,  TYPE_3 or TYPE_4 \n");                                                                      
     exit(1);                                                                                                                                                                  
   }
#if defined TM_USE_MPI
//Doing the neccesary communication
   for (s1=spinorstart; s1<spinorend; ++s1)
     for (c1=0; c1<3; ++c1)
       for (f1=0; f1<2; ++f1){
           count=0;
           generic_exchange_direction_nonblocking( propfields[12*s1 + 4*c1 + 2*f1 + 0], sizeof(bispinor), TDOWN , request, &count );
           MPI_Waitall( count, request, statuses);
           count=0;
           generic_exchange_direction_nonblocking( propfields[12*s1 + 4*c1 + 2*f1 + 1], sizeof(bispinor), TUP   , request, &count );
           MPI_Waitall( count, request, statuses);
           count=0;
           generic_exchange_direction_nonblocking( propfields[12*s1 + 4*c1 + 2*f1 + 1], sizeof(bispinor), TDOWN , request, &count);
           MPI_Waitall( count, request, statuses);
       }
   free(request);
#endif
   for (i=0; i<T_global; ++i)
       paulitrace[i]=0.;
// Trace over the Pauli matrices
   for (tauindex=0; tauindex<3; ++tauindex){

//Trace over flavour degrees of freedom
      for (i=0; i<T_global; ++i)
         flavortrace[i]=0.;

      for (f1=0; f1<2; ++f1){

//Trace over spinor indices
         for (i=0; i<2*T_global; ++i){
            spinortrace[i]=0.;
         }

         for (s1=spinorstart; s1<spinorend; ++s1){

//Trace over spatial indices
            for (i=0; i<8*T_global; ++i){
               spacetrace[i]=0.;
            }
            for (ix=0; ix<VOLUME; ++ix){

//Trace over the color indices for each sites
               for (i=0; i<8; ++i)
                  colortrace[i]=0.;
               for (c1=0; c1<3; ++c1){
/*   
       TYPE V.1.a OR  V.1.b     U0^dagger(x)*U0^dagger(x-0)* (1-gamma5)/2 *  S(x-0,ytilde)
       TYPE V.2.a OR  V.2.b                                  (1-gamma5)/2 *  S(x-0,ytilde)
*/
                  _bispinor_null(running);
                  
                  if ( type_12 == TYPE_1){

                    bispinor_spinup_mult_su3matrix( &running, &propfields[12*s1 + 4*c1 + 2*f1][g_idn[ix][TUP]], &g_gauge_field[g_idn[ix][TUP]][TUP], DAGGER );

                    bispinor_spinup_mult_su3matrix( &running, &running, &g_gauge_field[ix][TUP], DAGGER );

                  }
                  else if ( type_12 == TYPE_2){
                    _vector_assign( running.sp_up.s2, propfields[12*s1 + 4*c1 + 2*f1][g_idn[ix][TUP]].sp_up.s2 );
                    _vector_assign( running.sp_up.s3, propfields[12*s1 + 4*c1 + 2*f1][g_idn[ix][TUP]].sp_up.s3 );

                    _vector_assign( running.sp_dn.s2, propfields[12*s1 + 4*c1 + 2*f1][g_idn[ix][TUP]].sp_dn.s2 );
                    _vector_assign( running.sp_dn.s3, propfields[12*s1 + 4*c1 + 2*f1][g_idn[ix][TUP]].sp_dn.s3 );
                  }
/*   
       TYPE V.1.a OR  V.1.b     phi^dagger(x)*tau_i*     U0^dagger(x)*U0^dagger(x-0)* (1-gamma5)/2 *  S(x-0,ytilde)
       TYPE V.2.a OR  V.2.b     phi^dagger(x)*tau_i                                        (1-gamma5)/2 *  S(x-0,ytilde)
*/
                  taui_scalarfield_spinor( &running, &running, GAMMA_DN, tauindex, ix, NODIR, DAGGER);

/*   
       TYPE V.1.a OR  V.1.b     S(ytilde,x+0)*phi^dagger(x)*tau_i*     U0^dagger(x)*U0^dagger(x-0)* (1-gamma5)/2 *  S(x-0,ytilde)
       TYPE V.2.a OR  V.2.b     S(ytilde,x-0)*phi^dagger(x)*tau_i                                   (1-gamma5)/2 *  S(x-0,ytilde)
*/

                  if (type_12 == TYPE_1){
                    multiply_backward_propagator(&running, propfields, &running, ix, TUP);
                  }
                  else if (type_12 == TYPE_2){
                    multiply_backward_propagator(&running, propfields, &running, ix, TDOWN);
                  }
/*   
       TYPE V.1.a                 tau_i*phi(ytilde)*       (1+gamma5)/2* S(ytilde, x+0)*tau_i*phi(x)*U0(x-0)*U0(x)* (1+gamma5)/2 *  S(x+0,ytilde)
       TYPE V.1.b                 phi^dagger(ytilde)*tau_i*(1-gamma5)/2* S(ytilde, x+0)*tau_i*phi(x)*U0(x-0)*U0(x)* (1+gamma5)/2 *  S(x+0,ytilde)

       TYPE V.2.a                 tau_i*phi(ytilde)*       (1+gamma5)/2* S(ytilde, x-0)*tau_i*phi(x)*               (1+gamma5)/2 *  S(x-0,ytilde)
       TYPE V.2.b                 phi^dagger(ytilde)*tau_i*(1-gamma5)/2* S(ytilde, x-0)*tau_i*phi(x)*               (1+gamma5)/2 *  S(x-0,ytilde)

*/
                  //delta( color component of bispinor running, c1) for all spinor and flavor indices
                  trace_in_color(colortrace, &running, c1 );
               } //End of trace color
               //sum over all lattice sites the result of the color trace
               trace_in_space( spacetrace, colortrace, ix);
            }  //End of trace in space
#if defined TM_USE_MPI
//Gather the results from all nodes to complete the trace in space
            for (i=0; i<8*T_global; ++i){
               _Complex double tmp;
               MPI_Allreduce(&spacetrace[i], &tmp, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
               spacetrace[i]= tmp;
            }
#endif
            // delta (spinor components of spacetrace, s1) for all time slices and flavor indices
            trace_in_spinor(spinortrace, spacetrace, s1);

         } //End of trace in spinor space

         if ( type_ab == TYPE_A ){
           taui_scalarfield_flavoronly( spinortrace, tauindex, NO_DAGG, LEFT );
         }
         else if ( type_ab == TYPE_B){
           taui_scalarfield_flavoronly( spinortrace, tauindex, DAGGER, LEFT  );
         }
         //delta(flavor component in spinortrace, f1) for all time slices
         trace_in_flavor( flavortrace, spinortrace, f1 );

      } //End of trace in flavor space

      for (i=0;i<T_global; ++i){
         paulitrace[i]+=flavortrace[i];
         (*results)[i+tauindex*T_global]= flavortrace[i]/4.;
      }
   } //End of trace for Pauli matrices


   if (g_cart_id == 0){printf("Wilson term Dirac Current Density correlator typeV results= %s %s\n", type_12 == TYPE_1 ? "1" : "2",type_ab == TYPE_A ? "a" :"b");}
   for (i=0; i<T_global; ++i){
      if (g_cart_id == 0){
        printf("WCDPL1 %d %d %.3d %10.10e %10.10e\n", type_12, type_ab, i, creal(paulitrace[i])/4., cimag(paulitrace[i])/4.);
      }
      (*results)[i+3*T_global] = paulitrace[i]/4.;
   }

   free(flavortrace);
   free(paulitrace);
   free(spacetrace);
   free(spinortrace);
   free(colortrace);
   
}
void wilsonterm_current_density_612ab( bispinor ** propfields, int type_12, int type_ab, _Complex double **results ){
   int ix,i;
   int f1,c1,s1,tauindex;
   int spinorstart=0, spinorend=4;
   bispinor **starting2d;
   bispinor running;
#if defined TM_USE_MPI
   int count;
   MPI_Status  statuses[8];
   MPI_Request *request;
#endif
   _Complex double *colortrace;
   _Complex double *spacetrace;
   _Complex double *spinortrace;
   _Complex double *flavortrace;
   _Complex double *paulitrace;

   colortrace= (_Complex double *)malloc(sizeof(_Complex double)*8);
   spacetrace= (_Complex double *)malloc(sizeof(_Complex double)*8*T_global);
   spinortrace=(_Complex double *)malloc(sizeof(_Complex double)*2*T_global);
   flavortrace=(_Complex double *)malloc(sizeof(_Complex double)*T_global);
   paulitrace= (_Complex double *)malloc(sizeof(_Complex double)*T_global);

   if ( (colortrace == NULL) || (spacetrace == NULL) || (spinortrace == NULL) || (flavortrace == NULL) || (paulitrace == NULL) )
   {
     printf("Error in mem allocation\n");
     exit(1);
   }
   *results=(_Complex double *)malloc(sizeof(_Complex double)*4*T_global);
   if (*results == NULL){
     printf("Not enough memory in current density six \n");
     exit(1);
   }

#if defined TM_USE_MPI
   request=( MPI_Request *) malloc(sizeof(MPI_Request)*8);
#endif
   if ( type_ab == TYPE_A ) {
        spinorstart=0;
        spinorend  =2;
   }
   else if ( type_ab == TYPE_B ){
        spinorstart=2;
        spinorend  =4;
   }
   else{
       if (g_cart_id == 0) fprintf(stdout,"Wrong argument for type_1234, it can only be TYPE_1, TYPE_2,  TYPE_3 or TYPE_4 \n");                                                                      
       exit(1);                                                                                                                                                                  
   }
   if (type_12 == TYPE_1){
     for (i=0; i<T_global; ++i)
       paulitrace[i]=0.;
     for (tauindex=0; tauindex<3; ++tauindex){
       for (i=0; i<T_global; ++i)
         flavortrace[i]=0.;
       for (f1=0; f1<2; ++f1){
         for (i=0; i<2*T_global; ++i)
           spinortrace[i]=0.;
         for (s1= spinorstart; s1<spinorend; ++s1){
           for (i=0; i<8*T_global; ++i)
             spacetrace[i]=0.;
           for (ix = 0; ix< VOLUME; ++ix){
             for (i=0; i<8; ++i)
               colortrace[i]=0.;
             for (c1=0; c1<3; ++c1){
               _bispinor_null(running);
               _vector_assign( running.sp_up.s2, propfields[12*s1+4*c1+2*f1][ix].sp_up.s2 );
               _vector_assign( running.sp_up.s3, propfields[12*s1+4*c1+2*f1][ix].sp_up.s3 );
               _vector_assign( running.sp_dn.s2, propfields[12*s1+4*c1+2*f1][ix].sp_dn.s2 );
               _vector_assign( running.sp_dn.s3, propfields[12*s1+4*c1+2*f1][ix].sp_dn.s3 );
             
               taui_scalarfield_spinor( &running, &running, GAMMA_DN, tauindex, ix, TDOWN, DAGGER );
                 
               multiply_backward_propagator(&running, propfields, &running, ix, NODIR );
  
               trace_in_color(colortrace,&running,c1);
             }  //End of trace color
              //sum over all lattice sites the result of the color trace
             trace_in_space(spacetrace,colortrace,ix);
           } //End of trace space
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
         if ( type_ab == TYPE_A ){
           taui_scalarfield_flavoronly( spinortrace, tauindex, NO_DAGG, LEFT );
         }
         else if ( type_ab == TYPE_B ){
           taui_scalarfield_flavoronly( spinortrace, tauindex, DAGGER, LEFT  );
         }
         trace_in_flavor( flavortrace, spinortrace, f1 );
       } //End of trace in flavor space
      //sum for all Pauli matrices
       for (i=0;i<T_global; ++i){
         paulitrace[i]+=flavortrace[i];
         (*results)[i+tauindex*T_global]=flavortrace[i]/4.;
       }
     } //End of trace for Pauli matrices
     if  (g_cart_id == 0){printf("Wilson term Dirac Current Density correlator typeVI results= %s %s\n", type_12 == TYPE_1 ? "1" : "2",type_ab == TYPE_A ? "a" :"b");}
     for (i=0; i<T_global; ++i){
       if (g_cart_id == 0){
        printf("WCDPL2 %d %d %.3d %10.10e %10.10e\n", type_12, type_ab, i, creal(paulitrace[i])/4., cimag(paulitrace[i])/4.);
       }
       (*results)[i+3*T_global]=paulitrace[i]/4.;
     }
   }
   if (type_12 == TYPE_2 ){
      starting2d=(bispinor **)malloc(sizeof(bispinor *)*3);
      if (starting2d == NULL){
        if (g_cart_id ==0){
          printf("Error in allocating temporary fields for bispinor starting2d type Vi\n");
          exit(1);
        }
      }
      for (i=0; i<3; ++i){
        starting2d[i] =(bispinor *)malloc(sizeof(bispinor)*VOLUMEPLUSRAND);
        if (starting2d[i] == NULL){
          if (g_cart_id == 0) printf("Memory allocation error starting2d VI\n");
          exit(1);
        }
        for (ix=0; ix<VOLUME; ++ix){
          _bispinor_null(starting2d[i][ix]);
        }
      } 
//Doing the neccesary communication
#if defined TM_USE_MPI
      for (s1=spinorstart; s1<spinorend; ++s1)
        for (c1=0; c1<3; ++c1)
           for (f1=0; f1<2; ++f1){
             count=0;
             generic_exchange_direction_nonblocking( propfields[12*s1 + 4*c1 + 2*f1 + 0], sizeof(bispinor), TUP   , request, &count );
             MPI_Waitall( count, request, statuses);
             count=0;
             generic_exchange_direction_nonblocking( propfields[12*s1 + 4*c1 + 2*f1 + 1], sizeof(bispinor), TDOWN   , request, &count );
             MPI_Waitall( count, request, statuses);
          }
#endif
      for (i=0; i<T_global; ++i)
        paulitrace[i]=0.;
      for (tauindex=0; tauindex<3; ++tauindex){
        for (i=0; i<T_global; ++i)
          flavortrace[i]=0.;
        for (f1=0; f1<2; ++f1){
          for (i=0; i<2*T_global; ++i)
            spinortrace[i]=0.;
          for (s1= spinorstart; s1<spinorend; ++s1){
            for (i=0; i<8*T_global; ++i)
              spacetrace[i]=0.;
            for (ix = 0; ix< VOLUME; ++ix){
              for (i=0; i<8; ++i)
                colortrace[i]=0.;
              for (c1=0; c1<3; ++c1){
                _bispinor_null( starting2d[c1][ix] );

                bispinor_spindown_mult_su3matrix( &starting2d[c1][ix], &propfields[12*s1 + 4*c1 + 2*f1][g_iup[ix][TUP]], &g_gauge_field[ix][TUP], NO_DAGG );

                bispinor_spindown_mult_su3matrix( &starting2d[c1][ix], &starting2d[c1][ix], &g_gauge_field[g_idn[ix][TUP]][TUP], NO_DAGG );

                taui_scalarfield_spinor( &starting2d[c1][ix], &starting2d[c1][ix], GAMMA_DN, tauindex, ix, NODIR, DAGGER );

                multiply_backward_propagator(&starting2d[c1][ix], propfields, &starting2d[c1][ix], ix, TDOWN);

              }
            }
#if defined TM_USE_MPI
            for (c1=0; c1<3; ++c1){
              count=0;
              generic_exchange_direction_nonblocking( starting2d[c1], sizeof(bispinor), TDOWN, request, &count );
              MPI_Waitall( count, request, statuses);
            }
#endif
            for (ix=0; ix<VOLUME; ++ix){
              for (i=0; i<8; ++i)
                colortrace[i]=0.;
              for (c1=0; c1<3; ++c1)
                trace_in_color(colortrace,&starting2d[c1][g_idn[ix][TUP]],c1);
              trace_in_space(spacetrace,colortrace,ix);
            } 
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
          if ( type_ab == TYPE_A ){
            taui_scalarfield_flavoronly( spinortrace, tauindex, NO_DAGG, LEFT );
          }
          else if ( type_ab == TYPE_B ){
            taui_scalarfield_flavoronly( spinortrace, tauindex, DAGGER, LEFT  );
          }
          trace_in_flavor( flavortrace, spinortrace, f1 );
        } //End of trace in flavor space
      //sum for all Pauli matrices
        for (i=0;i<T_global; ++i){
          paulitrace[i]+=flavortrace[i];
          (*results)[i+tauindex*T_global]=flavortrace[i]/4.;
        }
      } //End of trace for Pauli matrices
      if  (g_cart_id == 0){printf("Wilson term Dirac Current Density correlator typeVI results= %s %s\n", type_12 == TYPE_1 ? "1" : "2",type_ab == TYPE_A ? "a" :"b");}
       for (i=0; i<T_global; ++i){
         if (g_cart_id == 0){
          printf("WCDPL2 %d %d %.3d %10.10e %10.10e\n", type_12, type_ab, i, creal(paulitrace[i])/4., cimag(paulitrace[i])/4.);
         }
         (*results)[i+3*T_global]= paulitrace[i]/4.;
       }
       for (i=0; i<3; ++i){
         free(starting2d[i]);
       } 
       free(starting2d);
   }
   free(flavortrace);
   free(paulitrace);
   free(spacetrace);
   free(spinortrace);
   free(colortrace); 
#if defined TM_USE_MPI
   if (type_12 == TYPE_2)
     free(request);
#endif
}
#endif
