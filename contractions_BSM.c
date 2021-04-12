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
#include "git_hash.h"
#include "getopt.h"
#include "default_input_values.h"
#include "read_input.h"
#include "su3.h"
#include "operator/tm_operators.h"
#include "linalg_eo.h"
#include "geometry_eo.h"
#include "linalg/assign.h"
#include "operator/D_psi.h"
#ifdef TM_USE_BSM
#include "operator/D_psi_BSM3.h"
#include "operator/D_psi_BSM.h"
#include "operator/D_psi_BSM2b.h"
#include "operator/D_psi_BSM2f.h"
#include "operator/D_psi_BSM2m.h"
#endif
#include "operator/Dov_psi.h"
#include "operator/tm_operators_nd.h"
#include "operator/Hopping_Matrix.h"
#include "invert_eo.h"
#include "invert_doublet_eo.h"
#include "invert_overlap.h"
#include "invert_clover_eo.h"
#ifdef TM_USE_BSM
#include "init/init_scalar_field.h"
#include "init/init_bsm_2hop_lookup.h"
#endif
#include "boundary.h"
#include "start.h"
#include "solver/solver.h"
#include "xchange/xchange_gauge.h"
#include "prepare_source.h"
#include <io/params.h>
#include <io/gauge.h>
#include <io/spinor.h>
#include <io/utils.h>
#ifdef TM_USE_BSM
#include "io/scalar.h"
#endif
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
#include "contractions/contractions_checks.h"
#include "contractions/contractions_FP.h"
#include "solver/solver_field.h"
#include "source_generation.h"
#include "ranlxd.h"

int DAGGER;
int NO_DAGG;

int GAMMA_UP;
int GAMMA_DN;
int NO_GAMMA;

int WITH_SCALAR;
int NO_SCALAR;

int TYPE_A;
int TYPE_B;

int TYPE_1;
int TYPE_2;
int TYPE_3;
int TYPE_4;

int TYPE_I;
int TYPE_II;

int RIGHT;
int LEFT;


static void usage()
{
  fprintf(stdout, "Options: [-f input-filename]\n");
  exit(0);
}
static void process_args(int argc, char *argv[], char ** input_filename, char ** filename) {
  int c;
  while ((c = getopt(argc, argv, "h?vVf:o:")) != -1) {
    switch (c) {
      case 'f':
        *input_filename = calloc(200, sizeof(char));
        strncpy(*input_filename, optarg, 200);
        break;
      case 'o':
        *filename = calloc(200, sizeof(char));
        strncpy(*filename, optarg, 200);
        break;
      case 'v':
        verbose = 1;
        break;
      case 'V':
        if(g_proc_id == 0) {
//          fprintf(stdout,"%s %s\n",PACKAGE_STRING,git_hash);
        }
        exit(0);
        break;
      case 'h':
      case '?':
      default:
        if( g_proc_id == 0 ) {
          usage();
        }
        break;
    }
  }
}

static void set_default_filenames(char ** input_filename, char ** filename) {
  if( *input_filename == NULL ) {
    *input_filename = calloc(13, sizeof(char));
    strcpy(*input_filename,"invert.input");
  }

  if( *filename == NULL ) {
    *filename = calloc(7, sizeof(char));
    strcpy(*filename,"output");
  }
}
extern int nstore;
int check_geometry();
static void set_default_filenames(char ** input_filename, char ** filename);
static void process_args(int argc, char *argv[], char ** input_filename, char ** filename);
#ifndef TM_USE_BSM
int main(int argc, char *argv[]){
  printf("Works only with BSM operators switched on \n");
}
#else
int main(int argc, char *argv[]){
  FILE *parameterfile = NULL;
  FILE *out=NULL;
  char datafilename[206];
  char parameterfilename[206];
  char conf_filename[50];
  char scalar_filename[50];
  char * input_filename = NULL;
  char * filename = NULL;
  double plaquette_energy;
  int i,j,isample=0,op_id=0;
  char prop_fname[200];
  char contractions_fname[200];
  int src_idx, pos;
//  int count;
  int status_geo;
  int ix;
  _Complex double *current,*pseudoscalar,*scalar,*temp;
  _Complex double *current1,*current2,*current3;
  _Complex double *pscalar1,*pscalar2,*pscalar3;
  _Complex double *scalar1, *scalar2, *scalar3 ;
#if defined TM_USE_MPI

  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &g_proc_id);
#endif
  process_args(argc, argv, &input_filename,&filename);
  set_default_filenames(&input_filename, &filename);
//Setting default constants

  DAGGER=1;
  NO_DAGG=0; 

  GAMMA_UP=1;
  GAMMA_DN=-1;
  NO_GAMMA=0;

  WITH_SCALAR=1;
  NO_SCALAR=0;

  TYPE_A=1;
  TYPE_B=0;

  TYPE_1=1;
  TYPE_2=0;
  TYPE_3=2;
  TYPE_4=3;

  TYPE_I=1;
  TYPE_II=0;
  
  RIGHT=1;
  LEFT=0;

  /* Read the input file */
  if ( (i = read_input(input_filename)) != 0)
  {
      fprintf(stderr, "Could not find input file: %s\nAborting...\n", input_filename);
      exit(-1);
  }

  if(g_proc_id==0)
  {
      fprintf(stdout, "#parameter  rho_BSM set to %f\n",  rho_BSM);
      fprintf(stdout, "#parameter  eta_BSM set to %f\n",  eta_BSM);
      fprintf(stdout, "#parameter   m0_BSM set to %f\n",   m0_BSM);
      fprintf(stdout, "#parameter mu03_BSM set to %f\n", mu03_BSM);
      fprintf(stdout, "#parameter mu01_BSM set to %f\n", mu01_BSM);
  }

#ifdef TM_USE_OMP
  init_openmp();
#endif
  tmlqcd_mpi_init(argc, argv);

  if(g_proc_id == 0)
  {
      fprintf(stdout,"# The number of processes is %d \n",g_nproc);
      fprintf(stdout,"# The lattice size is %d x %d x %d x %d\n",
         (int)(T*g_nproc_t), (int)(LX*g_nproc_x), (int)(LY*g_nproc_y), (int)(g_nproc_z*LZ));
      fprintf(stdout,"# The local lattice size is %d x %d x %d x %d\n",
        (int)(T), (int)(LX), (int)(LY),(int) LZ);
      fflush(stdout);
  }


  g_dbw2rand = 0;

  /* starts the single and double precision random number */
  /* generator                                            */
  start_ranlux(rlxd_level, random_seed);


#ifdef _GAUGE_COPY
  j = init_gauge_field(VOLUMEPLUSRAND, 1);
#else
  j = init_gauge_field(VOLUMEPLUSRAND, 0);
#endif

  if (j != 0)
  {
      fprintf(stderr, "Not enough memory for gauge_fields! Aborting...\n");
      exit(-1);
  }

  init_geometry_indices(VOLUMEPLUSRAND);

/* Iniiialising the spinor fields */
#if (defined SSE || defined SSE2 || SSE3)
  signal(SIGILL, &catch_ill_inst);
#endif

  DUM_DERI = 8;
  DUM_MATRIX = DUM_DERI + 5;
#if ((defined BGL && defined XLC) || defined _USE_TSPLITPAR)
  NO_OF_SPINORFIELDS = DUM_MATRIX + 3;
#else
  NO_OF_SPINORFIELDS = DUM_MATRIX + 3;
#endif
  for(j = 0; j < no_operators; j++) if(!operator_list[j].even_odd_flag) even_odd_flag = 0;

#ifndef TM_USE_MPI
  g_dbw2rand = 0;
#endif

  if (even_odd_flag)
  {
      j = init_spinor_field(VOLUMEPLUSRAND / 2, NO_OF_SPINORFIELDS);
  }
  else
  {
      j = init_spinor_field(VOLUMEPLUSRAND, NO_OF_SPINORFIELDS);
  }
  if (j != 0)
  {
      fprintf(stderr, "Not enough memory for spinor fields! Aborting...\n");
      exit(-1);
  }
  j = init_bispinor_field(VOLUMEPLUSRAND, 4);
  if ( j!= 0)
  {
      fprintf(stderr, "Not enough memory for bispinor fields! Aborting...\n");
      exit(0);
  }

  int numbScalarFields = 4;
  j = init_scalar_field(VOLUMEPLUSRAND, numbScalarFields);
  if ( j!= 0)
  {
      fprintf(stderr, "Not enough memory for scalar fields! Aborting...\n");
      exit(0);
  }

  spinor ** temp_field = NULL;
  init_solver_field(&temp_field, VOLUMEPLUSRAND, 2);


  /* define the geometry */

  geometry();
  g_kappa=-1;
  if ((g_cart_id == 0) && (g_kappa != -1))
  {
      fprintf(stdout, "#error anti-periodic boundary condition is implemented via g_kappa %e\n",g_kappa);
      exit(1);
  }
  boundary(g_kappa);

  status_geo = check_geometry();
  if (status_geo != 0)
  {
      fprintf(stderr, "Checking of geometry failed. Unable to proceed.\nAborting....\n");
      exit(1);
  }


  if (Nsave == 0) {
    Nsave = 1;
  }

  g_mu = g_mu1;

  if (g_cart_id == 0)
  {
    /*construct the filenames for the observables and the parameters*/
      strncpy(datafilename, filename, 200);
      strcat(datafilename, ".data");
      strncpy(parameterfilename, filename, 200);
      strcat(parameterfilename, ".para");

      parameterfile = fopen(parameterfilename, "w");
      write_first_messages(parameterfile, "invert", git_hash);
      fclose(parameterfile);
  }

  init_operators();


  for (j = 0; j < Nmeas; j++)
  {
      sprintf(conf_filename, "%s.%.4d", gauge_input_filename, nstore);
      if (g_cart_id == 0)
      {
          printf("#\n# Trying to read gauge field from file %s in %s precision.\n",
            conf_filename, (gauge_precision_read_flag == 32 ? "single" : "double"));
          fflush(stdout);
      }
      if ( (i = read_gauge_field(conf_filename,g_gauge_field) ) !=0)
      {
          fprintf(stderr, "Error %d while reading gauge field from %s\n Aborting...\n", i, conf_filename);
          exit(-2);
      }

      snprintf(conf_filename, 50, "%s_smeared.%.4d", gauge_input_filename, nstore);
      if (g_cart_id == 0) {
        printf("#\n# Trying to read smeared gauge field from file %s in %s precision.\n",
                conf_filename, (gauge_precision_read_flag == 32 ? "single" : "double"));
        fflush(stdout);
      }

      if( (i = read_gauge_field(conf_filename,g_smeared_gauge_field)) !=0) {
          fprintf(stderr, "Error %d while reading gauge field from %s\n Aborting...\n", i, conf_filename);
          exit(-2);
      }


      if (g_cart_id == 0) {
          printf("# Finished reading gauge field.\n");
          fflush(stdout);
      }

#ifdef TM_USE_MPI
      xchange_gauge(g_gauge_field);
      xchange_gauge(g_smeared_gauge_field);

#endif
    /*compute the energy of the gauge field*/
      plaquette_energy = measure_plaquette( (const su3**) g_gauge_field);


      if (g_cart_id == 0) {
          printf("# The computed plaquette value is %e.\n", plaquette_energy / (6.*VOLUME*g_nproc));
          fflush(stdout);
      }

      plaquette_energy = measure_plaquette( (const su3**) g_smeared_gauge_field);


      if (g_cart_id == 0) {
          printf("# The computed plaquette value for smeared gauge field is %e.\n", plaquette_energy / (6.*VOLUME*g_nproc));
          fflush(stdout);
      }

      if(SourceInfo.type == 1) {
          index_start = 0;
          index_end = 1;
      }


      if (g_cart_id == 0) {
          fprintf(stdout, "#\n"); /*Indicate starting of the operator part*/
      }
      for (op_id =0; op_id < no_operators; op_id++){
          if ( (operator_list[op_id].type== BSM2f) || (operator_list[op_id].type == BSM3) ){
              if (operator_list[op_id].type== BSM2f){
                init_D_psi_BSM2f();
              }
              else {
                init_D_psi_BSM3();
                init_sw_fields(VOLUME);

                //Note here the factor of 1/2. has been applied since
                //the routine assign_mul_one_sw_pm_imu_site_lexic computes
                //1+i *csw*\sum_{\mu,nu} \sigma_mu,nuF_mu,nu/2.
                sw_term( (const su3**) g_smeared_gauge_field, 1.,  csw_BSM/2.);
              }
              operator_list[op_id].prop_zero=(bispinor  **)malloc(sizeof(bispinor*)*48);
              if (operator_list[op_id].prop_zero == NULL){
                printf("Error in memory allocation for storing the propagators\n");
                exit(1);
              }
              for (int ii=0; ii<48; ++ii){
                operator_list[op_id].prop_zero[ii]=(bispinor *)malloc(sizeof(bispinor)*VOLUMEPLUSRAND);
                if ( operator_list[op_id].prop_zero[ii] == NULL ){
                  printf("Error in allocating memory for propagators\n");
                  exit(1);
                }
              }
              if ( ( vectorcurrentcurrent_BSM == 1 ) || ( axialcurrentcurrent_BSM == 1 )){
                operator_list[op_id].prop_ntmone=(bispinor  **)malloc(sizeof(bispinor*)*48);
                if (operator_list[op_id].prop_ntmone == NULL){
                  printf("Error in memory allocation for storing the propagators\n");
                  exit(1);
                }
                for (int ii=0; ii<48; ++ii){
                  operator_list[op_id].prop_ntmone[ii]=(bispinor *)malloc(sizeof(bispinor)*VOLUMEPLUSRAND);
                  if ( operator_list[op_id].prop_ntmone[ii] == NULL ){
                    printf("Error in allocating memory for propagators\n");
                    exit(1);
                  }
                }
              }
          }
          boundary( operator_list[op_id].kappa);
          g_kappa = operator_list[op_id].kappa;
          if (g_cart_id ==0) {fprintf(stdout, "#kappa value=%e\n", g_kappa);}
          g_mu = 0.;
          if (g_cart_id == 0) printf("# npergauge=%d\n", operator_list[op_id].npergauge);

          if (g_cart_id == 0) printf("# Starting scalar counter is %d for gauge field %d \n", nscalar, nstore );
          /* support multiple inversions for the BSM operator, one for each scalar field */

          for(int i_pergauge = 0; i_pergauge < operator_list[op_id].npergauge; ++i_pergauge){
             /* set scalar field counter to InitialScalarCounter */
             int iscalar = nscalar+j*operator_list[op_id].nscalarstep*operator_list[op_id].npergauge+i_pergauge*operator_list[op_id].nscalarstep;
             operator_list[op_id].n = iscalar;
             // read scalar field
             if( strcmp(scalar_input_filename, "create_random_scalarfield") == 0 )
             {
                for( int s = 0; s < 4; s++) { ranlxd(g_scalar_field[s], VOLUME); }
             }
             else if ( strcmp(scalar_input_filename, "create_unit_scalarfield") == 0 ){             
                unit_scalar_field(g_scalar_field);
             }
             else
             {
                snprintf(scalar_filename, 50, "%s.%.8d", scalar_input_filename, iscalar);
                if (g_cart_id == 0)
                {
                    printf("#\n# Trying to read scalar field from file %s in %s precision.\n",
                       scalar_filename, (scalar_precision_read_flag == 32 ? "single" : "double"));
                    fflush(stdout);
                }
                int i;
                double read_end, read_begin=gettime();

                if( (i = read_scalar_field_parallel(scalar_filename,g_scalar_field)) !=0)
                {
                    fprintf(stderr, "Error %d while reading scalar field from %s\n Aborting...\n", i, scalar_filename);
                    exit(-2);
                }
                read_end=gettime();

                if (g_cart_id == 0) {
                   printf("# Finished reading scalar field in %.4e seconds.\n",read_end-read_begin); 
                   fflush(stdout);
                }

             }//End of reading scalar field

#if defined TM_USE_MPI
             for( int s=0; s<4; s++ )
               generic_exchange_nogauge(g_scalar_field[s], sizeof(scalar));
#endif
             for( isample = 0; isample < no_samples; isample++)
             {
               if (propagatorsonthefly_BSM == 1){

                 if ((g_cart_id == 0 ) && ( (index_start != 0) || (index_end!= 12) ))
                 {
                    fprintf(stderr, "Contraction can be computed only with full set of point propagators\n");
                    exit(1);
                 }

                 for(ix = index_start; ix < index_end; ix++) {
                    if (g_cart_id == 0) {
                      fprintf(stdout, "#\n"); /*Indicate starting of new index*/
                    }

                    /* we use g_spinor_field[0-7] for sources and props for the moment */
                    /* 0-3 in case of 1 flavour  */
                    /* 0-7 in case of 2 flavours */

                    prepare_source(nstore, isample, ix, op_id, read_source_flag, source_location, 0);

//                  if (g_cart_id == 0) printf("Source has been prepared\n\n\n");
                    //randmize initial guess for eigcg if needed-----experimental
                    if( (operator_list[op_id].solver == INCREIGCG) && (operator_list[op_id].solver_params.eigcg_rand_guess_opt) )
                    { //randomize the initial guess
                        gaussian_volume_source( operator_list[op_id].prop0, operator_list[op_id].prop1,isample,ix,0); //need to check this
                    } 
		    
		    operator_list[op_id].inverter(op_id, index_start, 1);

                 }//end of loop for spinor and color source degrees of freedom

                 if ( ( vectorcurrentcurrent_BSM == 1 ) || ( axialcurrentcurrent_BSM == 1 )){
                    int tindex=source_location/(LX*g_nproc_x*LY*g_nproc_y*LZ*g_nproc_z);
                    int tnewindex=(tindex-1+T_global)%T_global;
                    int spatialindex=source_location % (LX*g_nproc_x*LY*g_nproc_y*LZ*g_nproc_z);
                    int backsource=tnewindex*(LX*g_nproc_x*LY*g_nproc_y*LZ*g_nproc_z)+spatialindex;

                    for(ix = index_start; ix < index_end; ix++) {
                      if (g_cart_id == 0) {
                        fprintf(stdout, "#\n"); /*Indicate starting of new index*/
                      }

                      /* we use g_spinor_field[0-7] for sources and props for the moment */
                      /* 0-3 in case of 1 flavour  */
                      /* 0-7 in case of 2 flavours */

                      prepare_source(nstore, isample, ix, op_id, read_source_flag, backsource, 0);

                      //randmize initial guess for eigcg if needed-----experimental
                      if( (operator_list[op_id].solver == INCREIGCG) && (operator_list[op_id].solver_params.eigcg_rand_guess_opt) )
                      { //randomize the initial guess
                        gaussian_volume_source( operator_list[op_id].prop0, operator_list[op_id].prop1,isample,ix,0); //need to check this
                      }

                      operator_list[op_id].inverter(op_id, index_start, 1);
                    }//end of loop for spinor and color source degrees of freedom

                 }//end of vectorcurrentcurrent_BSM == 1

               }
               else{
                 for(src_idx = 0; src_idx < 12; src_idx++ )
                 {
                    snprintf(prop_fname,200,"bsm2prop.%.4d.%.2d.%02d.%.8d.inverted",nstore, isample, src_idx, iscalar);

                    for(pos = 0; pos < 8; ){
                       printf("READCHECK: Propagator in pos %02d from file %s\n", pos/2,prop_fname);

//read the propagator from source d to sink d 
                       read_spinor(g_spinor_field[0], g_spinor_field[1], prop_fname, pos);
                       convert_eo_to_lexic(temp_field[0], g_spinor_field[0], g_spinor_field[1]);
                       pos+=1;

//read the propagator from source d to sink u
                       read_spinor(g_spinor_field[0], g_spinor_field[1], prop_fname, pos);
                       convert_eo_to_lexic(temp_field[1], g_spinor_field[0], g_spinor_field[1]);
                       pos+=1;

//create a bispinor first insert sink u then sink d
//Store them in such a way that the u-ones should come first
                       compact(operator_list[op_id].prop_zero[pos > 4 ? src_idx*4+pos/2-3 : src_idx*4+pos/2+ 1], temp_field[1], temp_field[0]);
                    }

                 }//end of loop for spinor and color source degrees of freedom

                 if ( (vectorcurrentcurrent_BSM == 1 ) || ( axialcurrentcurrent_BSM == 1 )){
                   for(src_idx = 0; src_idx < 12; src_idx++ )
                   {
                      snprintf(prop_fname,200,"bsm2prop.%.4d.%.2d.%02d.%.8d.inverted",nstore, T_global-1, src_idx, iscalar);
                      for(pos = 0; pos < 8; ){
                        printf("READCHECK: Propagator in pos %02d from file %s\n", pos/2,prop_fname);
//read the propagator from source d to sink d 
                        read_spinor(g_spinor_field[0], g_spinor_field[1], prop_fname, pos);
                        convert_eo_to_lexic(temp_field[0], g_spinor_field[0], g_spinor_field[1]);
                        pos+=1;

//read the propagator from source d to sink u
                        read_spinor(g_spinor_field[0], g_spinor_field[1], prop_fname, pos);
                        convert_eo_to_lexic(temp_field[1], g_spinor_field[0], g_spinor_field[1]);
                        pos+=1;
//create a bispinor first insert sink u then sink d
//Store them in such a way that the u-ones should come first
                        compact(operator_list[op_id].prop_ntmone[pos > 4 ? src_idx*4+pos/2-3 : src_idx*4+pos/2+ 1], temp_field[1], temp_field[0]);
                     }

                   }//end of loop for spinor and color source degrees of freedom
                 }
               }
               if (g_cart_id == 0){
                    snprintf(contractions_fname,200,"bsmcontractions.%.4d.%d.%.8d",nstore, isample, iscalar);
               }

               if (smearedcorrelator_BSM == 1){
                smear_scalar_fields_correlator(g_smeared_scalar_field, g_scalar_field, timesmearcorrelator_BSM );
                if (g_cart_id == 0) printf("Smeared : %e\t Non smeared %e\n", g_smeared_scalar_field[0][0],g_scalar_field[0][0]);
                 for ( int s=0; s<4; s++ )
                   generic_exchange_nogauge(g_smeared_scalar_field[s], sizeof(scalar));
               }
               if (g_cart_id == 0) {
                 printf("Following measurements will be done\n");
                 if (vectorcurrentcurrent_BSM == 1) printf("#Vectorcurrentcurrent3 correlation function\n");
                 if (axialcurrentcurrent_BSM == 1) printf("#Axialcurrentcurrent1 correlation function trivial scalar\n");
                 if (densitydensity_BSM == 1) printf("#Density Density correlation function\n");
                 if (densitydensity_s0s0_BSM == 1) printf("#Density Density s0s0-p0p0 using trivial scalar field\n");
                 if (densitydensity_sxsx_BSM == 1) printf("#Density Density sxsx-pxpx using trivial scalar field\n");
                 if (diraccurrentdensity_BSM == 1) printf("#Dirac current density correlation function\n");
                 if (wilsoncurrentdensitypr1_BSM == 1) printf("#Wilson  current density PR1 correlation function\n");
                 if (wilsoncurrentdensitypr2_BSM == 1) printf("#Wilson  current density PR2 correlation function\n");
                 if (wilsoncurrentdensitypl1_BSM == 1) printf("#Wilson  current density PL1 correlation function\n");
                 if (wilsoncurrentdensitypl2_BSM == 1) printf("#Wilson  current density PL2 correlation function\n");
                 if (vectorcurrentdensity_BSM == 1) printf("#JtildeV3 D3, JtildeV1 P2, JtildeV2 P1 to be (JtildeV1 P2 nad JtildeV2 P1 with trivial scalar calculated\n");
                 if (axialcurrentdensity_BSM == 1) printf("#JtildeA1 P1, JtildeA2 P2 to be calculated\n");
                 if (pdensityvectordensity_BSM == 1) printf("#P density times vector density (nontrivial scalar) to be calculated\n");

               }
               scalar=(_Complex double *)malloc(sizeof(_Complex double)*T_global);
               pseudoscalar=(_Complex double *)malloc(sizeof(_Complex double)*T_global);
               current=(_Complex double *)malloc(sizeof(_Complex double)*T_global);
               if ( scalar == NULL || pseudoscalar == NULL || current == NULL){
                 printf("Error in memory allocation for scalar pseudoscalar and current\n");
                 exit(1);
               }
               pscalar1=(_Complex double *)malloc(sizeof(_Complex double)*T_global);
               pscalar2=(_Complex double *)malloc(sizeof(_Complex double)*T_global);
               pscalar3=(_Complex double *)malloc(sizeof(_Complex double)*T_global);

               if (pscalar1 == NULL || pscalar2 == NULL || pscalar3 == NULL){
                 printf("Error in allocating memory for storing pseudoscalar results\n");
                 exit(1);
               }

               scalar1=(_Complex double *)malloc(sizeof(_Complex double)*T_global);
               scalar2=(_Complex double *)malloc(sizeof(_Complex double)*T_global);
               scalar3=(_Complex double *)malloc(sizeof(_Complex double)*T_global);

               if (scalar1 == NULL || scalar2 == NULL || scalar3 == NULL){
                 printf("Error in allocating memory for storing scalar results\n");
                 exit(1);
               }

               current1=(_Complex double *)malloc(sizeof(_Complex double)*T_global);
               current2=(_Complex double *)malloc(sizeof(_Complex double)*T_global);
               current3=(_Complex double *)malloc(sizeof(_Complex double)*T_global);

               if (current1 == NULL || current2 == NULL || current3 == NULL){
                 printf("Error in allocating memory for storing current results\n");
                 exit(1);
               }



               for (int ii=0;ii<T_global; ++ii){
                 scalar[ii]=0.0;
                 pseudoscalar[ii]=0.0;
                 current[ii]=0.0;
                 pscalar1[ii]=0.0;
                 pscalar2[ii]=0.0;
                 pscalar3[ii]=0.0;
                 scalar1[ii]=0.0;
                 scalar2[ii]=0.0;
                 scalar3[ii]=0.0;
                 current1[ii]=0.0;
                 current2[ii]=0.0;
                 current3[ii]=0.0;
               }
               if (g_cart_id == 0){
                  out=fopen(contractions_fname,"a");
                  if (out == NULL){
                    printf("Error in opening file for storing contractions %s\n",filename);
                    exit(1);
                  }
               }
               if ( vectorcurrentcurrent_BSM == 1){
                 vector_axial_current_current_1234(operator_list[op_id].prop_zero, operator_list[op_id].prop_ntmone, TYPE_1, 2, 0, &temp );
                 for (int ii=0; ii<T_global; ++ii){
                   current[ii]+=(-1.)*temp[ii];
                 }
                 free(temp);
                 vector_axial_current_current_1234(operator_list[op_id].prop_zero, operator_list[op_id].prop_ntmone, TYPE_2, 2, 0, &temp );
                 for (int ii=0; ii<T_global; ++ii){
                   current[ii]+=(-1.)*temp[ii];
                 }
                 free(temp);
                 vector_axial_current_current_1234(operator_list[op_id].prop_zero, operator_list[op_id].prop_ntmone, TYPE_3, 2, 0, &temp );
                 for (int ii=0; ii<T_global; ++ii){
                   current[ii]+=(-1.)*temp[ii];
                 }
                 free(temp);
                 vector_axial_current_current_1234(operator_list[op_id].prop_zero, operator_list[op_id].prop_ntmone, TYPE_4, 2, 0, &temp );
                 for (int ii=0; ii<T_global; ++ii){
                   current[ii]+=(-1.)*temp[ii];
                 }
                 free(temp);

                 if (g_cart_id == 0){
                   for (int ii=0; ii<T_global; ++ii){
                     fprintf(out,"JTILDEV3JTILDEV3\t%d\t%10.10e\t%10.10e\n", ii, creal(current[ii]), cimag(current[ii]));
                   }
                 }
                 for (int ii=0;ii<T_global; ++ii){
                   scalar[ii]=0.0;
                   pseudoscalar[ii]=0.0;
                   current[ii]=0.0;
                   pscalar1[ii]=0.0;
                   pscalar2[ii]=0.0;
                   pscalar3[ii]=0.0;
                   scalar1[ii]=0.0;
                   scalar2[ii]=0.0;
                   scalar3[ii]=0.0;
                   current1[ii]=0.0;
                   current2[ii]=0.0;
                   current3[ii]=0.0;
                 }
               }
               if ( axialcurrentcurrent_BSM == 1 ){

                 unit_scalar_field(g_scalar_field);
                 for( int s=0; s<4; s++ )
                   generic_exchange_nogauge(g_scalar_field[s], sizeof(scalar));
                 if (smearedcorrelator_BSM == 1){
                   smear_scalar_fields_correlator(g_smeared_scalar_field, g_scalar_field, timesmearcorrelator_BSM );
                   for ( int s=0; s<4; s++ )
                    generic_exchange_nogauge(g_smeared_scalar_field[s], sizeof(scalar));
                 }


                 vector_axial_current_current_1234(operator_list[op_id].prop_zero, operator_list[op_id].prop_ntmone, TYPE_1, 0, 1, &temp );
                 for (int ii=0; ii<T_global; ++ii){
                   current[ii]+=(-1.)*temp[ii];
                 }
                 free(temp);
                 vector_axial_current_current_1234(operator_list[op_id].prop_zero, operator_list[op_id].prop_ntmone, TYPE_2, 0, 1, &temp );
                 for (int ii=0; ii<T_global; ++ii){
                   current[ii]+=(-1.)*temp[ii];
                 }
                 free(temp);
                 vector_axial_current_current_1234(operator_list[op_id].prop_zero, operator_list[op_id].prop_ntmone, TYPE_3, 0, 1, &temp );
                 for (int ii=0; ii<T_global; ++ii){
                   current[ii]+=(-1.)*temp[ii];
                 }
                 free(temp);
                 vector_axial_current_current_1234(operator_list[op_id].prop_zero, operator_list[op_id].prop_ntmone, TYPE_4, 0, 1, &temp );
                 for (int ii=0; ii<T_global; ++ii){
                   current[ii]+=(-1.)*temp[ii];
                 }
                 free(temp);

                 if (g_cart_id == 0){
                   for (int ii=0; ii<T_global; ++ii){
                     fprintf(out,"JTILDEA1JTILDEA1\t%d\t%10.10e\t%10.10e\n", ii, creal(current[ii]), cimag(current[ii]));
                   }
                 }
                 for (int ii=0;ii<T_global; ++ii){
                   scalar[ii]=0.0;
                   pseudoscalar[ii]=0.0;
                   current[ii]=0.0;
                   pscalar1[ii]=0.0;
                   pscalar2[ii]=0.0;
                   pscalar3[ii]=0.0;
                   scalar1[ii]=0.0;
                   scalar2[ii]=0.0;
                   scalar3[ii]=0.0;
                   current1[ii]=0.0;
                   current2[ii]=0.0;
                   current3[ii]=0.0;
                 }

                 double read_end, read_begin=gettime();
                 if( (i = read_scalar_field_parallel(scalar_filename,g_scalar_field)) !=0 )
                 {
                    fprintf(stderr, "Error %d while reading scalar field from %s\n Aborting...\n", i, scalar_filename);
                    exit(-2);
                 }
                 read_end=gettime();
                 if (g_cart_id == 0) {
                   printf("# Finished reading scalar field in %.4e seconds.\n",read_end-read_begin);
                   fflush(stdout);
                 }
                 for( int s=0; s<4; s++ )
                   generic_exchange_nogauge(g_scalar_field[s], sizeof(scalar));
                 if (smearedcorrelator_BSM == 1){
                   smear_scalar_fields_correlator(g_smeared_scalar_field, g_scalar_field, timesmearcorrelator_BSM );
                   for ( int s=0; s<4; s++ )
                    generic_exchange_nogauge(g_smeared_scalar_field[s], sizeof(scalar));
                 }
                 
               }
               if (giancarlo_BSM == 1){
                 giancarlodensity( operator_list[op_id].prop_zero, 0, &temp );
                 for (int ii=0; ii<T_global; ++ii){
                   current[ii]+=(-1.)*temp[ii];
                 }
                 free(temp);
                 if (g_cart_id == 0){
                   for (int ii=0; ii<T_global; ++ii){
                     fprintf(out,"GIANCARLOUNITYNONTRIVIAL\t%d\t%10.10e\t%10.10e\n", ii, creal(current[ii]), cimag(current[ii]));
                   }
                 }
               }
               if (vectorcurrentdensity_BSM == 1){
                 for (int ii=0;ii<T_global; ++ii){
                   scalar[ii]=0.0;
                   pseudoscalar[ii]=0.0;
                   current[ii]=0.0;
                   pscalar1[ii]=0.0;
                   pscalar2[ii]=0.0;
                   pscalar3[ii]=0.0;
                   scalar1[ii]=0.0;
                   scalar2[ii]=0.0;
                   scalar3[ii]=0.0;
                   current1[ii]=0.0;
                   current2[ii]=0.0;
                   current3[ii]=0.0;
                 }
                 giancarlodensity( operator_list[op_id].prop_zero, 1, &temp );
                 for (int ii=0; ii<T_global; ++ii){
                   current[ii]+=(-1.)*temp[ii];
                 }
                 free(temp);
                 if (g_cart_id == 0){
                   for (int ii=0; ii<T_global; ++ii){
                     fprintf(out,"GIANCARLOTAU3NONTRIVIAL\t%d\t%10.10e\t%10.10e\n", ii, creal(current[ii]), cimag(current[ii]));
                   }
                 }
                 for (int ii=0;ii<T_global; ++ii){
                   scalar[ii]=0.0;
                   pseudoscalar[ii]=0.0;
                   current[ii]=0.0;
                   pscalar1[ii]=0.0;
                   pscalar2[ii]=0.0;
                   pscalar3[ii]=0.0;
                   scalar1[ii]=0.0;
                   scalar2[ii]=0.0;
                   scalar3[ii]=0.0;
                   current1[ii]=0.0;
                   current2[ii]=0.0;
                   current3[ii]=0.0;
                 }

                 vector_axial_current_density_1234(operator_list[op_id].prop_zero, TYPE_1,2, 2, 0, 0, &temp );
                 for (int ii=0; ii<T_global; ++ii){
                   current[ii]+=(-1.)*temp[ii];
                 }
                 free(temp);
                 vector_axial_current_density_1234(operator_list[op_id].prop_zero, TYPE_2,2, 2, 0, 0, &temp );
                 for (int ii=0; ii<T_global; ++ii){
                   current[ii]+=(+1.)*temp[ii];
                 }
                 free(temp);
                 vector_axial_current_density_1234(operator_list[op_id].prop_zero, TYPE_3,2, 2, 0, 0, &temp );
                 for (int ii=0; ii<T_global; ++ii){
                   current[ii]+=(-1.)*temp[ii];
                 }
                 free(temp);
                 vector_axial_current_density_1234(operator_list[op_id].prop_zero, TYPE_4,2, 2, 0, 0, &temp );
                 for (int ii=0; ii<T_global; ++ii){
                   current[ii]+=(+1.)*temp[ii];
                 }
                 free(temp);
                 temp= NULL;
                 if (g_cart_id == 0){
//                 fprintf(out,"S1S1nontrivialscalar:\n");
                   for (int ii=0; ii<T_global; ++ii){
                     fprintf(out,"JTILDEV3DS3NONTRIVIAL\t%d\t%10.10e\t%10.10e\n", ii, creal(current[ii]), cimag(current[ii]));
                   }
                 }
                 for (int ii=0;ii<T_global; ++ii){
                   scalar[ii]=0.0;
                   pseudoscalar[ii]=0.0;
                   current[ii]=0.0;
                   pscalar1[ii]=0.0;
                   pscalar2[ii]=0.0;
                   pscalar3[ii]=0.0;
                   scalar1[ii]=0.0;
                   scalar2[ii]=0.0;
                   scalar3[ii]=0.0;
                   current1[ii]=0.0;
                   current2[ii]=0.0;
                   current3[ii]=0.0;
                 }


                 unit_scalar_field(g_scalar_field);
                 for( int s=0; s<4; s++ )
                   generic_exchange_nogauge(g_scalar_field[s], sizeof(scalar));
                 if (smearedcorrelator_BSM == 1){
                   smear_scalar_fields_correlator(g_smeared_scalar_field, g_scalar_field, timesmearcorrelator_BSM );
                   for ( int s=0; s<4; s++ )
                    generic_exchange_nogauge(g_smeared_scalar_field[s], sizeof(scalar));
                 }

                 vector_axial_current_density_1234(operator_list[op_id].prop_zero, TYPE_1,0, 1, 0, 1, &temp );
                 for (int ii=0; ii<T_global; ++ii){
                   current[ii]+=(-1.)*temp[ii];
                 }
                 free(temp);
                 vector_axial_current_density_1234(operator_list[op_id].prop_zero, TYPE_2,0, 1, 0, 1, &temp );
                 for (int ii=0; ii<T_global; ++ii){
                   current[ii]+=(+1.)*temp[ii];
                 }
                 free(temp);
                 vector_axial_current_density_1234(operator_list[op_id].prop_zero, TYPE_3,0, 1, 0, 1, &temp );
                 for (int ii=0; ii<T_global; ++ii){
                   current[ii]+=(-1.)*temp[ii];
                 }
                 free(temp);
                 vector_axial_current_density_1234(operator_list[op_id].prop_zero, TYPE_4,0, 1, 0, 1, &temp );
                 for (int ii=0; ii<T_global; ++ii){
                   current[ii]+=(+1.)*temp[ii];
                 }
                 free(temp);
                 if (g_cart_id == 0){
//                 fprintf(out,"S1S1nontrivialscalar:\n");
                   for (int ii=0; ii<T_global; ++ii){
                     fprintf(out,"JTILDEV2P1TRIVIAL\t%d\t%10.10e\t%10.10e\n", ii, creal(current[ii]), cimag(current[ii]));
                   }
                 }
                 for (int ii=0;ii<T_global; ++ii){
                   scalar[ii]=0.0;
                   pseudoscalar[ii]=0.0;
                   current[ii]=0.0;
                   pscalar1[ii]=0.0;
                   pscalar2[ii]=0.0;
                   pscalar3[ii]=0.0;
                   scalar1[ii]=0.0;
                   scalar2[ii]=0.0;
                   scalar3[ii]=0.0;
                   current1[ii]=0.0;
                   current2[ii]=0.0;
                   current3[ii]=0.0;
                 }

                 vector_axial_current_density_1234(operator_list[op_id].prop_zero, TYPE_1,1, 0, 0, 1, &temp );
                 for (int ii=0; ii<T_global; ++ii){
                   current[ii]+=(-1.)*temp[ii];
                 }
                 free(temp);
                 vector_axial_current_density_1234(operator_list[op_id].prop_zero, TYPE_2,1, 0, 0, 1, &temp );
                 for (int ii=0; ii<T_global; ++ii){
                   current[ii]+=(+1.)*temp[ii];
                 }
                 free(temp);
                 vector_axial_current_density_1234(operator_list[op_id].prop_zero, TYPE_3,1, 0, 0, 1, &temp );
                 for (int ii=0; ii<T_global; ++ii){
                   current[ii]+=(-1.)*temp[ii];
                 }
                 free(temp);
                 vector_axial_current_density_1234(operator_list[op_id].prop_zero, TYPE_4,1, 0, 0, 1, &temp );
                 for (int ii=0; ii<T_global; ++ii){
                   current[ii]+=(+1.)*temp[ii];
                 }
                 free(temp);
                 if (g_cart_id == 0){         
                   for (int ii=0; ii<T_global; ++ii){
                     fprintf(out,"JTILDEV1P2TRIVIAL\t%d\t%10.10e\t%10.10e\n", ii, creal(current[ii]), cimag(current[ii]));
                   }
                 } 


                 for (int ii=0;ii<T_global; ++ii){
                   scalar[ii]=0.0;
                   pseudoscalar[ii]=0.0;
                   current[ii]=0.0;
                   pscalar1[ii]=0.0;
                   pscalar2[ii]=0.0;
                   pscalar3[ii]=0.0;
                   scalar1[ii]=0.0;
                   scalar2[ii]=0.0;
                   scalar3[ii]=0.0;
                   current1[ii]=0.0;
                   current2[ii]=0.0;
                   current3[ii]=0.0;
                 }

                 double read_end, read_begin=gettime();
                 if( (i = read_scalar_field_parallel(scalar_filename,g_scalar_field)) !=0 )
                 {
                    fprintf(stderr, "Error %d while reading scalar field from %s\n Aborting...\n", i, scalar_filename);
                    exit(-2);
                 }
                 read_end=gettime();
                 if (g_cart_id == 0) {
                   printf("# Finished reading scalar field in %.4e seconds.\n",read_end-read_begin);
                   fflush(stdout);
                 }
                 for( int s=0; s<4; s++ )
                   generic_exchange_nogauge(g_scalar_field[s], sizeof(scalar));
                 if (smearedcorrelator_BSM == 1){
                   smear_scalar_fields_correlator(g_smeared_scalar_field, g_scalar_field, timesmearcorrelator_BSM );
                   for ( int s=0; s<4; s++ )
                    generic_exchange_nogauge(g_smeared_scalar_field[s], sizeof(scalar));
                 }
                  
               }
               if (vectordensitydensity_BSM == 1){

                 for (int ii=0;ii<T_global; ++ii){
                   scalar[ii]=0.0;
                   pseudoscalar[ii]=0.0;
                   current[ii]=0.0;
                   pscalar1[ii]=0.0;
                   pscalar2[ii]=0.0;
                   pscalar3[ii]=0.0;
                   scalar1[ii]=0.0;
                   scalar2[ii]=0.0;
                   scalar3[ii]=0.0;
                   current1[ii]=0.0;
                   current2[ii]=0.0;
                   current3[ii]=0.0;
                 }

                 vector_density_density_1234(operator_list[op_id].prop_zero, TYPE_1,2, &temp );
                 for (int ii=0; ii<T_global; ++ii){
                   scalar[ii]+=(+1.)*temp[ii];
                 }
                 free(temp);
                 vector_density_density_1234(operator_list[op_id].prop_zero, TYPE_2,2, &temp );
                 for (int ii=0; ii<T_global; ++ii){
                   scalar[ii]+=(-1.)*temp[ii];
                 }
                 free(temp);
                 vector_density_density_1234(operator_list[op_id].prop_zero, TYPE_3,2, &temp );
                 for (int ii=0; ii<T_global; ++ii){
                   scalar[ii]+=(-1.)*temp[ii];
                 }
                 free(temp);
                 vector_density_density_1234(operator_list[op_id].prop_zero, TYPE_4,2, &temp );
                 for (int ii=0; ii<T_global; ++ii){
                   scalar[ii]+=(+1.)*temp[ii];
                 }
                 free(temp);
                 temp= NULL;
                 if (g_cart_id == 0){
//                 fprintf(out,"S1S1nontrivialscalar:\n");
                   for (int ii=0; ii<T_global; ++ii){
                     fprintf(out,"VECTORDENSITY3DENSITY3NONTRIVIAL\t%d\t%10.10e\t%10.10e\n", ii, creal(scalar[ii]), cimag(scalar[ii]));
                   }
                 }
                 for (int ii=0;ii<T_global; ++ii){
                   scalar[ii]=0.0;
                   pseudoscalar[ii]=0.0;
                   current[ii]=0.0;
                   pscalar1[ii]=0.0;
                   pscalar2[ii]=0.0;
                   pscalar3[ii]=0.0;
                   scalar1[ii]=0.0;
                   scalar2[ii]=0.0;
                   scalar3[ii]=0.0;
                   current1[ii]=0.0;
                   current2[ii]=0.0;
                   current3[ii]=0.0;
                 }

               }
               if (axialcurrentdensity_BSM == 1){
                 unit_scalar_field(g_scalar_field);
                 for( int s=0; s<4; s++ )
                   generic_exchange_nogauge(g_scalar_field[s], sizeof(scalar));
                 if (smearedcorrelator_BSM == 1){
                   smear_scalar_fields_correlator(g_smeared_scalar_field, g_scalar_field, timesmearcorrelator_BSM );
                   for ( int s=0; s<4; s++ )
                    generic_exchange_nogauge(g_smeared_scalar_field[s], sizeof(scalar));
                 }
                 vector_axial_current_density_1234(operator_list[op_id].prop_zero, TYPE_1, 0, 0, 1, 1, &temp );
                 for (int ii=0; ii<T_global; ++ii){
                   current[ii]+=(-1.)*temp[ii];
                 }
                 free(temp);
                 vector_axial_current_density_1234(operator_list[op_id].prop_zero, TYPE_2, 0, 0, 1, 1, &temp );
                 for (int ii=0; ii<T_global; ++ii){
                   current[ii]+=(+1.)*temp[ii];
                 }
                 free(temp);
                 vector_axial_current_density_1234(operator_list[op_id].prop_zero, TYPE_3, 0, 0, 1, 1, &temp );
                 for (int ii=0; ii<T_global; ++ii){
                   current[ii]+=(-1.)*temp[ii];
                 }
                 free(temp);
                 vector_axial_current_density_1234(operator_list[op_id].prop_zero, TYPE_4, 0, 0, 1, 1, &temp );
                 for (int ii=0; ii<T_global; ++ii){
                   current[ii]+=(+1.)*temp[ii];
                 }
                 free(temp);
                 if (g_cart_id == 0){
//                 fprintf(out,"S1S1nontrivialscalar:\n");
                   for (int ii=0; ii<T_global; ++ii){
                     fprintf(out,"JTILDEA1P1TRIVIAL\t%d\t%10.10e\t%10.10e\n", ii, creal(current[ii]), cimag(current[ii]));
                   }
                 }

                 for (int ii=0;ii<T_global; ++ii){
                   scalar[ii]=0.0;
                   pseudoscalar[ii]=0.0;
                   current[ii]=0.0;
                   pscalar1[ii]=0.0;
                   pscalar2[ii]=0.0;
                   pscalar3[ii]=0.0;
                   scalar1[ii]=0.0;
                   scalar2[ii]=0.0;
                   scalar3[ii]=0.0;
                   current1[ii]=0.0;
                   current2[ii]=0.0;
                   current3[ii]=0.0;
                 }

               }



               if (densitydensity_BSM == 1){
                 density_density_1234(operator_list[op_id].prop_zero, TYPE_1, &temp);
                 for (int ii=0; ii<T_global; ++ii){
                   pscalar1[ii]+=(-1.)*temp[ii           ];
                   pscalar2[ii]+=(-1.)*temp[ii+1*T_global];
                   pscalar3[ii]+=(-1.)*temp[ii+2*T_global];
                   pseudoscalar[ii] +=(-1.)*temp[ii+3*T_global];

                   scalar1[ii]+=(-1.)*temp[ii           ];
                   scalar2[ii]+=(-1.)*temp[ii+1*T_global];
                   scalar3[ii]+=(-1.)*temp[ii+2*T_global];
                   scalar[ii] +=(-1.)*temp[ii+3*T_global];
                 }
                 free(temp);
//                 density_density_1234_petros(operator_list[op_id].prop);
                 density_density_1234(operator_list[op_id].prop_zero, TYPE_2, &temp);
                 for (int ii=0; ii<T_global; ++ii){
                   pscalar1[ii]+=temp[ii           ];
                   pscalar2[ii]+=temp[ii+1*T_global];
                   pscalar3[ii]+=temp[ii+2*T_global];
                   pseudoscalar[ii] +=temp[ii+3*T_global];

                   scalar1[ii]+=(-1.)*temp[ii           ];
                   scalar2[ii]+=(-1.)*temp[ii+1*T_global];
                   scalar3[ii]+=(-1.)*temp[ii+2*T_global];
                   scalar[ii] +=(-1.)*temp[ii+3*T_global];
                 }
                 free(temp);

                 density_density_1234(operator_list[op_id].prop_zero, TYPE_3, &temp);
                 for (int ii=0; ii<T_global; ++ii){
                   pscalar1[ii]+=temp[ii           ];
                   pscalar2[ii]+=temp[ii+1*T_global];
                   pscalar3[ii]+=temp[ii+2*T_global];
                   pseudoscalar[ii] +=temp[ii+3*T_global];

                   scalar1[ii]+=(-1.)*temp[ii           ];
                   scalar2[ii]+=(-1.)*temp[ii+1*T_global];
                   scalar3[ii]+=(-1.)*temp[ii+2*T_global];
                   scalar[ii] +=(-1.)*temp[ii+3*T_global];
                 }
                 free(temp);

                 density_density_1234(operator_list[op_id].prop_zero, TYPE_4, &temp);
                 for (int ii=0; ii<T_global; ++ii){      
                   pscalar1[ii]+=(-1.)*temp[ii           ];
                   pscalar2[ii]+=(-1.)*temp[ii+1*T_global];
                   pscalar3[ii]+=(-1.)*temp[ii+2*T_global];
                   pseudoscalar[ii] +=(-1.)*temp[ii+3*T_global];

                   scalar1[ii]+=(-1.)*temp[ii           ];
                   scalar2[ii]+=(-1.)*temp[ii+1*T_global];
                   scalar3[ii]+=(-1.)*temp[ii+2*T_global];
                   scalar[ii] +=(-1.)*temp[ii+3*T_global];
                 }
                 free(temp);
                 if (g_cart_id == 0){
//                 fprintf(out,"S1S1nontrivialscalar:\n");
                   for (int ii=0; ii<T_global; ++ii){
                     fprintf(out,"S1S1NONTRIVIAL\t%d\t%10.10e\t%10.10e\n", ii, creal(scalar1[ii]), cimag(scalar1[ii])); 
                   }
//                 fprintf(out,"S2S2nontrivialscalar:\n");
                   for (int ii=0; ii<T_global; ++ii){
                     fprintf(out,"S2S2NONTRIVIAL\t%d\t%10.10e\t%10.10e\n", ii, creal(scalar2[ii]), cimag(scalar2[ii]));
                   }
//                 fprintf(out,"S3S3nontrivialscalar:\n");
                   for (int ii=0; ii<T_global; ++ii){
                     fprintf(out,"S3S3NONTRIVIAL\t%d\t%10.10e\t%10.10e\n", ii, creal(scalar3[ii]), cimag(scalar3[ii]));
                   }
//                 fprintf(out,"SSnontrivialscalar:\n");
                   for (int ii=0; ii<T_global; ++ii){
                     fprintf(out,"SSNONTRIVIAL\t%d\t%10.10e\t%10.10e\n", ii, creal(scalar[ii]), cimag(scalar[ii]));
                   }
//                 fprintf(out,"S1S1nontrivialscalar:\n");
                   for (int ii=0; ii<T_global; ++ii){
                     fprintf(out,"P1P1NONTRIVIAL\t%d\t%10.10e\t%10.10e\n", ii, creal(pscalar1[ii]), cimag(pscalar1[ii]));
                   }
//                 fprintf(out,"S2S2nontrivialscalar:\n");
                   for (int ii=0; ii<T_global; ++ii){
                     fprintf(out,"P2P2NONTRIVIAL\t%d\t%10.10e\t%10.10e\n", ii, creal(pscalar2[ii]), cimag(pscalar2[ii]));
                   }
//                 fprintf(out,"PS3PS3nontrivialscalar:\n");
                   for (int ii=0; ii<T_global; ++ii){
                     fprintf(out,"P3P3NONTRIVIAL\t%d\t%10.10e\t%10.10e\n", ii, creal(pscalar3[ii]), cimag(pscalar3[ii]));
                   }
//                 fprintf(out,"SSnontrivialscalar:\n");
                   for (int ii=0; ii<T_global; ++ii){
                     fprintf(out,"PPNONTRIVIAL\t%d\t%10.10e\t%10.10e\n", ii, creal(pseudoscalar[ii]), cimag(pseudoscalar[ii]));
                   }

                 }
                 for (int ii=0;ii<T_global; ++ii){
                   scalar[ii]=0.0;
                   pseudoscalar[ii]=0.0;
                   current[ii]=0.0;
                   pscalar1[ii]=0.0;
                   pscalar2[ii]=0.0;
                   pscalar3[ii]=0.0;
                   scalar1[ii]=0.0;
                   scalar2[ii]=0.0;
                   scalar3[ii]=0.0;
                   current1[ii]=0.0;
                   current2[ii]=0.0;
                   current3[ii]=0.0;
                 }
               }

               if (pdensityvectordensity_BSM == 1){
                 for (int ii=0;ii<T_global; ++ii){
                   scalar[ii]=0.0;
                   pseudoscalar[ii]=0.0;
                   current[ii]=0.0;
                   pscalar1[ii]=0.0;
                   pscalar2[ii]=0.0;
                   pscalar3[ii]=0.0;
                   scalar1[ii]=0.0;
                   scalar2[ii]=0.0;
                   scalar3[ii]=0.0;
                   current1[ii]=0.0;
                   current2[ii]=0.0;
                   current3[ii]=0.0;
                 }
                 density_ptau_density_vector( operator_list[op_id].prop_zero, TYPE_1,&temp);
                 for (int ii=0; ii<T_global; ++ii){
                   pscalar1[ii]+=(-1.)*temp[ii           ];
                   pscalar2[ii]+=(-1.)*temp[ii+1*T_global];
                   pscalar3[ii]+=(-1.)*temp[ii+2*T_global];
                   pseudoscalar[ii] +=(-1.)*temp[ii+3*T_global];
                 }
                 free(temp);
                 density_ptau_density_vector( operator_list[op_id].prop_zero, TYPE_2,&temp);
                 for (int ii=0; ii<T_global; ++ii){
                   pscalar1[ii]+=(+1.)*temp[ii           ];
                   pscalar2[ii]+=(+1.)*temp[ii+1*T_global];
                   pscalar3[ii]+=(+1.)*temp[ii+2*T_global];
                   pseudoscalar[ii] +=(+1.)*temp[ii+3*T_global];
                 }
                 free(temp);
                 if (g_cart_id == 0){
//                 fprintf(out,"S1S1nontrivialscalar:\n");
                   for (int ii=0; ii<T_global; ++ii){
                     fprintf(out,"P1DP1NONTRIVIAL\t%d\t%10.10e\t%10.10e\n", ii, creal(pscalar1[ii]), cimag(pscalar1[ii]));
                   }
//                 fprintf(out,"S2S2nontrivialscalar:\n");
                   for (int ii=0; ii<T_global; ++ii){
                     fprintf(out,"P2DP2NONTRIVIAL\t%d\t%10.10e\t%10.10e\n", ii, creal(pscalar2[ii]), cimag(pscalar2[ii]));
                   }
//                 fprintf(out,"PS3PS3nontrivialscalar:\n");
                   for (int ii=0; ii<T_global; ++ii){
                     fprintf(out,"P3DP3NONTRIVIAL\t%d\t%10.10e\t%10.10e\n", ii, creal(pscalar3[ii]), cimag(pscalar3[ii]));
                   }
//                 fprintf(out,"SSnontrivialscalar:\n");
                   for (int ii=0; ii<T_global; ++ii){
                     fprintf(out,"PDPNONTRIVIAL\t%d\t%10.10e\t%10.10e\n", ii, creal(pseudoscalar[ii]), cimag(pseudoscalar[ii]));
                   }

                 }
                 for (int ii=0;ii<T_global; ++ii){
                   scalar[ii]=0.0;
                   pseudoscalar[ii]=0.0;
                   current[ii]=0.0;
                   pscalar1[ii]=0.0;
                   pscalar2[ii]=0.0;
                   pscalar3[ii]=0.0;
                   scalar1[ii]=0.0;
                   scalar2[ii]=0.0;
                   scalar3[ii]=0.0;
                   current1[ii]=0.0;
                   current2[ii]=0.0;
                   current3[ii]=0.0;
                 }
                 smearedcorrelator_BSM = 0;

                 for (int ii=0;ii<T_global; ++ii){
                   scalar[ii]=0.0;
                   pseudoscalar[ii]=0.0;
                   current[ii]=0.0;
                   pscalar1[ii]=0.0;
                   pscalar2[ii]=0.0;
                   pscalar3[ii]=0.0;
                   scalar1[ii]=0.0;
                   scalar2[ii]=0.0;
                   scalar3[ii]=0.0;
                   current1[ii]=0.0;
                   current2[ii]=0.0;
                   current3[ii]=0.0;
                 }
                 density_ptau_density_vector( operator_list[op_id].prop_zero, TYPE_1,&temp);
                 for (int ii=0; ii<T_global; ++ii){
                   pscalar1[ii]+=(-1.)*temp[ii           ];
                   pscalar2[ii]+=(-1.)*temp[ii+1*T_global];
                   pscalar3[ii]+=(-1.)*temp[ii+2*T_global];
                   pseudoscalar[ii] +=(-1.)*temp[ii+3*T_global];
                 }
                 free(temp);
                 density_ptau_density_vector( operator_list[op_id].prop_zero, TYPE_2,&temp);
                 for (int ii=0; ii<T_global; ++ii){
                   pscalar1[ii]+=(+1.)*temp[ii           ];
                   pscalar2[ii]+=(+1.)*temp[ii+1*T_global];
                   pscalar3[ii]+=(+1.)*temp[ii+2*T_global];
                   pseudoscalar[ii] +=(+1.)*temp[ii+3*T_global];
                 }
                 free(temp);
                 if (g_cart_id == 0){
//                 fprintf(out,"S1S1nontrivialscalar:\n");
                   for (int ii=0; ii<T_global; ++ii){
                     fprintf(out,"P1DP1NONSMEAREDNONTRIVIAL\t%d\t%10.10e\t%10.10e\n", ii, creal(pscalar1[ii]), cimag(pscalar1[ii]));
                   }
//                 fprintf(out,"S2S2nontrivialscalar:\n");
                   for (int ii=0; ii<T_global; ++ii){
                     fprintf(out,"P2DP2NONSMEAREDNONTRIVIAL\t%d\t%10.10e\t%10.10e\n", ii, creal(pscalar2[ii]), cimag(pscalar2[ii]));
                   }
//                 fprintf(out,"PS3PS3nontrivialscalar:\n");
                   for (int ii=0; ii<T_global; ++ii){
                     fprintf(out,"P3DP3NONSMEAREDNONTRIVIAL\t%d\t%10.10e\t%10.10e\n", ii, creal(pscalar3[ii]), cimag(pscalar3[ii]));
                   }
//                 fprintf(out,"SSnontrivialscalar:\n");
                   for (int ii=0; ii<T_global; ++ii){
                     fprintf(out,"PDPNONSMEAREDNONTRIVIAL\t%d\t%10.10e\t%10.10e\n", ii, creal(pseudoscalar[ii]), cimag(pseudoscalar[ii]));
                   }

                 }
                 
                 for (int ii=0;ii<T_global; ++ii){
                   scalar[ii]=0.0;
                   pseudoscalar[ii]=0.0;
                   current[ii]=0.0;
                   pscalar1[ii]=0.0;
                   pscalar2[ii]=0.0;
                   pscalar3[ii]=0.0;
                   scalar1[ii]=0.0;
                   scalar2[ii]=0.0;
                   scalar3[ii]=0.0;
                   current1[ii]=0.0;
                   current2[ii]=0.0;
                   current3[ii]=0.0;
                 }

                 smearedcorrelator_BSM = 1;

                 
               }


               if (densitydensity_s0s0_BSM == 1){

                 unit_scalar_field(g_scalar_field);
                 for( int s=0; s<4; s++ )
                   generic_exchange_nogauge(g_scalar_field[s], sizeof(scalar));
                 if (smearedcorrelator_BSM == 1){
                   smear_scalar_fields_correlator(g_smeared_scalar_field, g_scalar_field, timesmearcorrelator_BSM);
                   for ( int s=0; s<4; s++ )
                    generic_exchange_nogauge(g_smeared_scalar_field[s], sizeof(scalar));
                 }
                 density_density_1234_s0s0(operator_list[op_id].prop_zero, TYPE_1, &temp);
                 for (int ii=0; ii<T_global; ++ii){
                   pseudoscalar[ii] +=(-1.0)*temp[ii];
                   scalar[ii] +=(-1.)*temp[ii];
                 }
                 free(temp);                  
                 density_density_1234_s0s0(operator_list[op_id].prop_zero, TYPE_2, &temp);
                 for (int ii=0; ii<T_global; ++ii){
                   pseudoscalar[ii] +=temp[ii];
                   scalar[ii] +=(-1.)*temp[ii];
                 }
                 free(temp);
                 density_density_1234_s0s0(operator_list[op_id].prop_zero, TYPE_3, &temp);
                 for (int ii=0; ii<T_global; ++ii){
                   pseudoscalar[ii] +=temp[ii];
                   scalar[ii] +=(-1.)*temp[ii];
                 }
                 free(temp);
                 density_density_1234_s0s0(operator_list[op_id].prop_zero, TYPE_4, &temp);
                 for (int ii=0; ii<T_global; ++ii){
                   pseudoscalar[ii] +=(-1.)*temp[ii];
                   scalar[ii] +=(-1.)*temp[ii];
                 }
                 free(temp);
                 if (g_cart_id == 0){
//                 fprintf(out,"S0S0trivialscalar:\n");
                   for (int ii=0; ii<T_global; ++ii){
                     fprintf(out,"S0S0TRIVIAL\t%d\t%10.10e\t%10.10e\n", ii, creal(scalar[ii]), cimag(scalar[ii]));
                   }
//                 fprintf(out,"S2S2nontrivialscalar:\n");
                   for (int ii=0; ii<T_global; ++ii){
                     fprintf(out,"P0P0TRIVIAL\t%d\t%10.10e\t%10.10e\n", ii, creal(pseudoscalar[ii]), cimag(pseudoscalar[ii]));
                   }
                 }
                 for (int ii=0;ii<T_global; ++ii){
                   scalar[ii]=0.0;
                   pseudoscalar[ii]=0.0;
                   current[ii]=0.0;
                   pscalar1[ii]=0.0;
                   pscalar2[ii]=0.0;
                   pscalar3[ii]=0.0;
                   scalar1[ii]=0.0;
                   scalar2[ii]=0.0;
                   scalar3[ii]=0.0;
                   current1[ii]=0.0;
                   current2[ii]=0.0;
                   current3[ii]=0.0;
                 }
                 double read_end, read_begin=gettime();
                 if( (i = read_scalar_field_parallel(scalar_filename,g_scalar_field)) !=0 )
                 {
                    fprintf(stderr, "Error %d while reading scalar field from %s\n Aborting...\n", i, scalar_filename);
                    exit(-2);
                 }
                 read_end=gettime();
                 if (g_cart_id == 0) {
                   printf("# Finished reading scalar field in %.4e seconds.\n",read_end-read_begin);
                   fflush(stdout);
                 }
                 for( int s=0; s<4; s++ )
                   generic_exchange_nogauge(g_scalar_field[s], sizeof(scalar));
                 if (smearedcorrelator_BSM == 1){
                   smear_scalar_fields_correlator(g_smeared_scalar_field, g_scalar_field, timesmearcorrelator_BSM );
                   for ( int s=0; s<4; s++ )
                    generic_exchange_nogauge(g_smeared_scalar_field[s], sizeof(scalar));
                 }
               }
 
               if (densitydensity_sxsx_BSM ==1){
                 unit_scalar_field(g_scalar_field);
                 for( int s=0; s<4; s++ )
                   generic_exchange_nogauge(g_scalar_field[s], sizeof(scalar));
                 if (smearedcorrelator_BSM == 1){
                   smear_scalar_fields_correlator(g_smeared_scalar_field, g_scalar_field, timesmearcorrelator_BSM );
                   for ( int s=0; s<4; s++ )
                    generic_exchange_nogauge(g_smeared_scalar_field[s], sizeof(scalar));
                 }
                 density_density_1234_sxsx(operator_list[op_id].prop_zero, TYPE_1, &temp);
                 for (int ii=0; ii<T_global; ++ii){
                   pscalar1[ii]+=(-1.)*temp[ii           ];
                   pscalar2[ii]+=(-1.)*temp[ii+1*T_global];
                   pscalar3[ii]+=(-1.)*temp[ii+2*T_global];
                   pseudoscalar[ii] +=(-1.)*temp[ii+3*T_global];

                   scalar1[ii]+=(-1.)*temp[ii           ];
                   scalar2[ii]+=(-1.)*temp[ii+1*T_global];
                   scalar3[ii]+=(-1.)*temp[ii+2*T_global];
                   scalar[ii] +=(-1.)*temp[ii+3*T_global];
                 }
                 free(temp);

                 density_density_1234_sxsx(operator_list[op_id].prop_zero, TYPE_2, &temp);
                 for (int ii=0; ii<T_global; ++ii){
                   pscalar1[ii]+=temp[ii           ];
                   pscalar2[ii]+=temp[ii+1*T_global];
                   pscalar3[ii]+=temp[ii+2*T_global];
                   pseudoscalar[ii] +=temp[ii+3*T_global];

                   scalar1[ii]+=(-1.)*temp[ii           ];
                   scalar2[ii]+=(-1.)*temp[ii+1*T_global];
                   scalar3[ii]+=(-1.)*temp[ii+2*T_global];
                   scalar[ii] +=(-1.)*temp[ii+3*T_global];
                 }
                 free(temp);

                 density_density_1234_sxsx(operator_list[op_id].prop_zero, TYPE_3, &temp);
                 for (int ii=0; ii<T_global; ++ii){
                   pscalar1[ii]+=temp[ii           ];
                   pscalar2[ii]+=temp[ii+1*T_global];
                   pscalar3[ii]+=temp[ii+2*T_global];
                   pseudoscalar[ii] +=temp[ii+3*T_global];

                   scalar1[ii]+=(-1.)*temp[ii           ];
                   scalar2[ii]+=(-1.)*temp[ii+1*T_global];
                   scalar3[ii]+=(-1.)*temp[ii+2*T_global];
                   scalar[ii] +=(-1.)*temp[ii+3*T_global];
                 }
                 free(temp);

                 density_density_1234_sxsx(operator_list[op_id].prop_zero, TYPE_4, &temp);
                 for (int ii=0; ii<T_global; ++ii){
                   pscalar1[ii]+=(-1.)*temp[ii           ];
                   pscalar2[ii]+=(-1.)*temp[ii+1*T_global];
                   pscalar3[ii]+=(-1.)*temp[ii+2*T_global];
                   pseudoscalar[ii] +=(-1.)*temp[ii+3*T_global];

                   scalar1[ii]+=(-1.)*temp[ii           ];
                   scalar2[ii]+=(-1.)*temp[ii+1*T_global];
                   scalar3[ii]+=(-1.)*temp[ii+2*T_global];
                   scalar[ii] +=(-1.)*temp[ii+3*T_global];
                 }
                 free(temp);
                 if (g_cart_id == 0){
//                 fprintf(out,"S1S1nontrivialscalar:\n");
                   for (int ii=0; ii<T_global; ++ii){
                     fprintf(out,"S1S1TRIVIAL\t%d\t%10.10e\t%10.10e\n", ii, creal(scalar1[ii]), cimag(scalar1[ii]));
                   }
//                 fprintf(out,"S2S2nontrivialscalar:\n");
                   for (int ii=0; ii<T_global; ++ii){
                     fprintf(out,"S2S2TRIVIAL\t%d\t%10.10e\t%10.10e\n", ii, creal(scalar2[ii]), cimag(scalar2[ii]));
                   }
//                 fprintf(out,"S3S3nontrivialscalar:\n");
                   for (int ii=0; ii<T_global; ++ii){
                     fprintf(out,"S3S3TRIVIAL\t%d\t%10.10e\t%10.10e\n", ii, creal(scalar3[ii]), cimag(scalar3[ii]));
                   }
//                 fprintf(out,"SSnontrivialscalar:\n");
                   for (int ii=0; ii<T_global; ++ii){
                     fprintf(out,"SSTRIVIAL\t%d\t%10.10e\t%10.10e\n", ii, creal(scalar[ii]), cimag(scalar[ii]));
                   }
//                 fprintf(out,"S1S1nontrivialscalar:\n");
                   for (int ii=0; ii<T_global; ++ii){
                     fprintf(out,"P1P1TRIVIAL\t%d\t%10.10e\t%10.10e\n", ii, creal(pscalar1[ii]), cimag(pscalar1[ii]));
                   }
//                 fprintf(out,"S2S2nontrivialscalar:\n");
                   for (int ii=0; ii<T_global; ++ii){
                     fprintf(out,"P2P2TRIVIAL\t%d\t%10.10e\t%10.10e\n", ii, creal(pscalar2[ii]), cimag(pscalar2[ii]));
                   }
//                 fprintf(out,"PS3PS3nontrivialscalar:\n");
                   for (int ii=0; ii<T_global; ++ii){
                     fprintf(out,"P3P3TRIVIAL\t%d\t%10.10e\t%10.10e\n", ii, creal(pscalar3[ii]), cimag(pscalar3[ii]));
                   }
//                 fprintf(out,"SSnontrivialscalar:\n");
                   for (int ii=0; ii<T_global; ++ii){
                     fprintf(out,"PPTRIVIAL\t%d\t%10.10e\t%10.10e\n", ii, creal(pseudoscalar[ii]), cimag(pseudoscalar[ii]));
                   }

                 }
                 for (int ii=0;ii<T_global; ++ii){
                   scalar[ii]=0.0;
                   pseudoscalar[ii]=0.0;
                   current[ii]=0.0;
                   pscalar1[ii]=0.0;
                   pscalar2[ii]=0.0;
                   pscalar3[ii]=0.0;
                   scalar1[ii]=0.0;
                   scalar2[ii]=0.0;
                   scalar3[ii]=0.0;
                   current1[ii]=0.0;
                   current2[ii]=0.0;
                   current3[ii]=0.0;
                 }

                 double read_end, read_begin=gettime();
                 if( (i = read_scalar_field_parallel(scalar_filename,g_scalar_field)) !=0 )
                 {
                    fprintf(stderr, "Error %d while reading scalar field from %s\n Aborting...\n", i, scalar_filename);
                    exit(-2);
                 }
                 read_end=gettime();
                 if (g_cart_id == 0) {
                   printf("# Finished reading scalar field in %.4e seconds.\n",read_end-read_begin);
                   fflush(stdout);
                 }
                 for( int s=0; s<4; s++ )
                   generic_exchange_nogauge(g_scalar_field[s], sizeof(scalar));
                 if (smearedcorrelator_BSM == 1){
                   smear_scalar_fields_correlator(g_smeared_scalar_field, g_scalar_field, timesmearcorrelator_BSM );
                   for ( int s=0; s<4; s++ )
                    generic_exchange_nogauge(g_smeared_scalar_field[s], sizeof(scalar));
                 }
               }
               if (diraccurrentdensity_BSM == 1){
                 naivedirac_current_density_12ab( operator_list[op_id].prop_zero, TYPE_I , TYPE_A, &temp);
                 for (int ii=0; ii<T_global; ++ii){
                   current1[ii]+=(-1.)*temp[ii           ];
                   current2[ii]+=(-1.)*temp[ii+1*T_global];
                   current3[ii]+=(-1.)*temp[ii+2*T_global];
                   current[ii] +=(-1.)*temp[ii+3*T_global];
                 }
                 free(temp);

//                 diraccurrent1a_petros( operator_list[op_id].prop );
                 naivedirac_current_density_12ab( operator_list[op_id].prop_zero, TYPE_I , TYPE_B, &temp);
                 for (int ii=0; ii<T_global; ++ii){
                   current1[ii]+=(+1.)*temp[ii           ];
                   current2[ii]+=(+1.)*temp[ii+1*T_global];
                   current3[ii]+=(+1.)*temp[ii+2*T_global];
                   current[ii] +=(+1.)*temp[ii+3*T_global];
                 }
                 free(temp);

                 naivedirac_current_density_12ab( operator_list[op_id].prop_zero, TYPE_II, TYPE_A, &temp);
                 for (int ii=0; ii<T_global; ++ii){
                   current1[ii]+=(-1.)*temp[ii           ];
                   current2[ii]+=(-1.)*temp[ii+1*T_global];
                   current3[ii]+=(-1.)*temp[ii+2*T_global];
                   current[ii] +=(-1.)*temp[ii+3*T_global];
                 }
                 free(temp);

                 naivedirac_current_density_12ab( operator_list[op_id].prop_zero, TYPE_II, TYPE_B, &temp);
                 for (int ii=0; ii<T_global; ++ii){
                   current1[ii]+=(+1.)*temp[ii           ];
                   current2[ii]+=(+1.)*temp[ii+1*T_global];
                   current3[ii]+=(+1.)*temp[ii+2*T_global];
                   current[ii] +=(+1.)*temp[ii+3*T_global];
                 }
                 free(temp);
                 if (g_cart_id == 0){
//                   fprintf(out,"S1S1nontrivialscalar:\n");
                   for (int ii=0; ii<T_global; ++ii){
                     fprintf(out,"J1D1NONTRIVIAL\t%d\t%10.10e\t%10.10e\n", ii, creal(current1[ii]), cimag(current1[ii]));
                   }
//                 fprintf(out,"S2S2nontrivialscalar:\n");
                   for (int ii=0; ii<T_global; ++ii){
                     fprintf(out,"J2D2NONTRIVIAL\t%d\t%10.10e\t%10.10e\n", ii, creal(current2[ii]), cimag(current2[ii]));
                   }
//                 fprintf(out,"PS3PS3nontrivialscalar:\n");
                   for (int ii=0; ii<T_global; ++ii){
                     fprintf(out,"J3D3NONTRIVIAL\t%d\t%10.10e\t%10.10e\n", ii, creal(current3[ii]), cimag(current3[ii]));
                   }
//                 fprintf(out,"SSnontrivialscalar:\n");
                   for (int ii=0; ii<T_global; ++ii){
                     fprintf(out,"JDNONTRIVIAL\t%d\t%10.10e\t%10.10e\n", ii, creal(current[ii]), cimag(current[ii]));
                   }
                 }
                 for (int ii=0;ii<T_global; ++ii){
                   scalar[ii]=0.0;
                   pseudoscalar[ii]=0.0;
                   current[ii]=0.0;
                   pscalar1[ii]=0.0;
                   pscalar2[ii]=0.0;
                   pscalar3[ii]=0.0;
                   scalar1[ii]=0.0;
                   scalar2[ii]=0.0;
                   scalar3[ii]=0.0;
                   current1[ii]=0.0;
                   current2[ii]=0.0;
                   current3[ii]=0.0;
                 }
               }
               if (wilsoncurrentdensitypr1_BSM == 1){
                 wilsonterm_current_density_312ab( operator_list[op_id].prop_zero, TYPE_1, TYPE_A, &temp);
                 for (int ii=0; ii<T_global; ++ii){
                   current1[ii]+=(-1.)*temp[ii           ];
                   current2[ii]+=(-1.)*temp[ii+1*T_global];
                   current3[ii]+=(-1.)*temp[ii+2*T_global];
                   current[ii] +=(-1.)*temp[ii+3*T_global];
                 }
                 free(temp);
//                 wilsoncurrent_density_3_petros( operator_list[op_id].prop );
                 wilsonterm_current_density_312ab( operator_list[op_id].prop_zero, TYPE_1, TYPE_B, &temp);
                 for (int ii=0; ii<T_global; ++ii){
                   current1[ii]+=(+1.)*temp[ii           ];
                   current2[ii]+=(+1.)*temp[ii+1*T_global];
                   current3[ii]+=(+1.)*temp[ii+2*T_global];
                   current[ii] +=(+1.)*temp[ii+3*T_global];
                 }
                 free(temp);
                 wilsonterm_current_density_312ab( operator_list[op_id].prop_zero, TYPE_2, TYPE_A, &temp);
                 for (int ii=0; ii<T_global; ++ii){
                   current1[ii]+=(+1.)*temp[ii           ];
                   current2[ii]+=(+1.)*temp[ii+1*T_global];
                   current3[ii]+=(+1.)*temp[ii+2*T_global];
                   current[ii] +=(+1.)*temp[ii+3*T_global];
                 }
                 free(temp);


                 wilsonterm_current_density_312ab( operator_list[op_id].prop_zero, TYPE_2, TYPE_B, &temp);
                 for (int ii=0; ii<T_global; ++ii){
                   current1[ii]+=(-1.)*temp[ii           ];
                   current2[ii]+=(-1.)*temp[ii+1*T_global];
                   current3[ii]+=(-1.)*temp[ii+2*T_global];
                   current[ii] +=(-1.)*temp[ii+3*T_global];
                 }
                 free(temp);
                 if (g_cart_id == 0){
//                   fprintf(out,"S1S1nontrivialscalar:\n");
                   for (int ii=0; ii<T_global; ++ii){
                     fprintf(out,"JWPR11D1NONTRIVIAL\t%d\t%10.10e\t%10.10e\n", ii, creal(current1[ii]), cimag(current1[ii]));
                   }
//                 fprintf(out,"S2S2nontrivialscalar:\n");
                   for (int ii=0; ii<T_global; ++ii){
                     fprintf(out,"JWPR12D2NONTRIVIAL\t%d\t%10.10e\t%10.10e\n", ii, creal(current2[ii]), cimag(current2[ii]));
                   }
//                 fprintf(out,"PS3PS3nontrivialscalar:\n");
                   for (int ii=0; ii<T_global; ++ii){
                     fprintf(out,"JWPR13D3NONTRIVIAL\t%d\t%10.10e\t%10.10e\n", ii, creal(current3[ii]), cimag(current3[ii]));
                   }
//                 fprintf(out,"SSnontrivialscalar:\n");
                   for (int ii=0; ii<T_global; ++ii){
                     fprintf(out,"JWPR1DNONTRIVIAL\t%d\t%10.10e\t%10.10e\n", ii, creal(current[ii]), cimag(current[ii]));
                   }
                 }
                 for (int ii=0;ii<T_global; ++ii){
                   scalar[ii]=0.0;
                   pseudoscalar[ii]=0.0;
                   current[ii]=0.0;
                   pscalar1[ii]=0.0;
                   pscalar2[ii]=0.0;
                   pscalar3[ii]=0.0;
                   scalar1[ii]=0.0;
                   scalar2[ii]=0.0;
                   scalar3[ii]=0.0;
                   current1[ii]=0.0;
                   current2[ii]=0.0;
                   current3[ii]=0.0;
                 }
               }
               if (wilsoncurrentdensitypr2_BSM == 1){
                 wilsonterm_current_density_412ab( operator_list[op_id].prop_zero, TYPE_1, TYPE_A, &temp);
                 for (int ii=0; ii<T_global; ++ii){
                   current1[ii]+=(-1.)*temp[ii           ];
                   current2[ii]+=(-1.)*temp[ii+1*T_global];
                   current3[ii]+=(-1.)*temp[ii+2*T_global];
                   current[ii] +=(-1.)*temp[ii+3*T_global];
                 }
                 free(temp);
                 wilsonterm_current_density_412ab( operator_list[op_id].prop_zero, TYPE_1, TYPE_B, &temp);
                 for (int ii=0; ii<T_global; ++ii){
                   current1[ii]+=(+1.)*temp[ii           ];
                   current2[ii]+=(+1.)*temp[ii+1*T_global];
                   current3[ii]+=(+1.)*temp[ii+2*T_global];
                   current[ii] +=(+1.)*temp[ii+3*T_global];
                 }
                 free(temp);
                 wilsonterm_current_density_412ab( operator_list[op_id].prop_zero, TYPE_2, TYPE_A, &temp);
                 for (int ii=0; ii<T_global; ++ii){
                   current1[ii]+=(+1.)*temp[ii           ];
                   current2[ii]+=(+1.)*temp[ii+1*T_global];
                   current3[ii]+=(+1.)*temp[ii+2*T_global];
                   current[ii] +=(+1.)*temp[ii+3*T_global];
                 }
                 free(temp);
                 wilsonterm_current_density_412ab( operator_list[op_id].prop_zero, TYPE_2, TYPE_B, &temp);
                 for (int ii=0; ii<T_global; ++ii){
                   current1[ii]+=(-1.)*temp[ii           ];
                   current2[ii]+=(-1.)*temp[ii+1*T_global];
                   current3[ii]+=(-1.)*temp[ii+2*T_global];
                   current[ii] +=(-1.)*temp[ii+3*T_global];
                 }
                 free(temp);
                 if (g_cart_id == 0){
//                   fprintf(out,"S1S1nontrivialscalar:\n");
                   for (int ii=0; ii<T_global; ++ii){
                     fprintf(out,"JWPR21D1NONTRIVIAL\t%d\t%10.10e\t%10.10e\n", ii, creal(current1[ii]), cimag(current1[ii]));
                   }
//                 fprintf(out,"S2S2nontrivialscalar:\n");
                   for (int ii=0; ii<T_global; ++ii){
                     fprintf(out,"JWPR2D2NONTRIVIAL\t%d\t%10.10e\t%10.10e\n",  ii, creal(current2[ii]), cimag(current2[ii]));
                   }
//                 fprintf(out,"PS3PS3nontrivialscalar:\n");
                   for (int ii=0; ii<T_global; ++ii){
                     fprintf(out,"JWPR23D3NONTRIVIAL\t%d\t%10.10e\t%10.10e\n", ii, creal(current3[ii]), cimag(current3[ii]));
                   }
//                 fprintf(out,"SSnontrivialscalar:\n");
                   for (int ii=0; ii<T_global; ++ii){
                     fprintf(out,"JWPR2DNONTRIVIAL\t%d\t%10.10e\t%10.10e\n", ii, creal(current[ii]), cimag(current[ii]));
                   }
                 }
                 for (int ii=0;ii<T_global; ++ii){
                   scalar[ii]=0.0;
                   pseudoscalar[ii]=0.0;
                   current[ii]=0.0;
                   pscalar1[ii]=0.0;
                   pscalar2[ii]=0.0;
                   pscalar3[ii]=0.0;
                   scalar1[ii]=0.0;
                   scalar2[ii]=0.0;
                   scalar3[ii]=0.0;
                   current1[ii]=0.0;
                   current2[ii]=0.0;
                   current3[ii]=0.0;
                 }

               }
               if (wilsoncurrentdensitypl1_BSM == 1){
                 wilsonterm_current_density_512ab( operator_list[op_id].prop_zero, TYPE_1, TYPE_A, &temp);
                 for (int ii=0; ii<T_global; ++ii){
                   current1[ii]+=(-1.)*temp[ii           ];
                   current2[ii]+=(-1.)*temp[ii+1*T_global];
                   current3[ii]+=(-1.)*temp[ii+2*T_global];
                   current[ii] +=(-1.)*temp[ii+3*T_global];
                 }
                 free(temp);

                 wilsonterm_current_density_512ab( operator_list[op_id].prop_zero, TYPE_1, TYPE_B, &temp);
                 for (int ii=0; ii<T_global; ++ii){
                   current1[ii]+=(+1.)*temp[ii           ];
                   current2[ii]+=(+1.)*temp[ii+1*T_global];
                   current3[ii]+=(+1.)*temp[ii+2*T_global];
                   current[ii] +=(+1.)*temp[ii+3*T_global];
                 }
                 free(temp);

                 wilsonterm_current_density_512ab( operator_list[op_id].prop_zero, TYPE_2, TYPE_A, &temp);
                 for (int ii=0; ii<T_global; ++ii){
                   current1[ii]+=(+1.)*temp[ii           ];
                   current2[ii]+=(+1.)*temp[ii+1*T_global];
                   current3[ii]+=(+1.)*temp[ii+2*T_global];
                   current[ii] +=(+1.)*temp[ii+3*T_global];
                 }
                 free(temp);

                 wilsonterm_current_density_512ab( operator_list[op_id].prop_zero, TYPE_2, TYPE_B, &temp);
                 for (int ii=0; ii<T_global; ++ii){
                   current1[ii]+=(-1.)*temp[ii           ];
                   current2[ii]+=(-1.)*temp[ii+1*T_global];
                   current3[ii]+=(-1.)*temp[ii+2*T_global];
                   current[ii] +=(-1.)*temp[ii+3*T_global];
                 }
                 free(temp);
                 if (g_cart_id == 0){
//                   fprintf(out,"S1S1nontrivialscalar:\n");
                   for (int ii=0; ii<T_global; ++ii){
                     fprintf(out,"JWPL11D1NONTRIVIAL\t%d\t%10.10e\t%10.10e\n", ii, creal(current1[ii]), cimag(current1[ii]));
                   }
//                 fprintf(out,"S2S2nontrivialscalar:\n");
                   for (int ii=0; ii<T_global; ++ii){
                     fprintf(out,"JWPL12D2NONTRIVIAL\t%d\t%10.10e\t%10.10e\n", ii, creal(current2[ii]), cimag(current2[ii]));
                   }
//                 fprintf(out,"PS3PS3nontrivialscalar:\n");
                   for (int ii=0; ii<T_global; ++ii){
                     fprintf(out,"JWPL13D3NONTRIVIAL\t%d\t%10.10e\t%10.10e\n", ii, creal(current3[ii]), cimag(current3[ii]));
                   }
//                 fprintf(out,"SSnontrivialscalar:\n");
                   for (int ii=0; ii<T_global; ++ii){
                     fprintf(out,"JWPL1DNONTRIVIAL\t%d\t%10.10e\t%10.10e\n", ii, creal(current[ii]), cimag(current[ii]));
                   }
                 }
                 for (int ii=0;ii<T_global; ++ii){
                   scalar[ii]=0.0;
                   pseudoscalar[ii]=0.0;
                   current[ii]=0.0;
                   pscalar1[ii]=0.0;
                   pscalar2[ii]=0.0;
                   pscalar3[ii]=0.0;
                   scalar1[ii]=0.0;
                   scalar2[ii]=0.0;
                   scalar3[ii]=0.0;
                   current1[ii]=0.0;
                   current2[ii]=0.0;
                   current3[ii]=0.0;
                 }
               }
               if (wilsoncurrentdensitypl2_BSM == 1){
               //wilsoncurrent61a_petros( operator_list[op_id].prop );
                 wilsonterm_current_density_612ab( operator_list[op_id].prop_zero, TYPE_1, TYPE_A, &temp);
                 for (int ii=0; ii<T_global; ++ii){
                   current1[ii]+=(-1.)*temp[ii           ];
                   current2[ii]+=(-1.)*temp[ii+1*T_global];
                   current3[ii]+=(-1.)*temp[ii+2*T_global];
                   current[ii] +=(-1.)*temp[ii+3*T_global];
                 }
                 free(temp);

                 wilsonterm_current_density_612ab( operator_list[op_id].prop_zero, TYPE_1, TYPE_B, &temp);
                 for (int ii=0; ii<T_global; ++ii){
                   current1[ii]+=(+1.)*temp[ii           ];
                   current2[ii]+=(+1.)*temp[ii+1*T_global];
                   current3[ii]+=(+1.)*temp[ii+2*T_global];
                   current[ii] +=(+1.)*temp[ii+3*T_global];
                 }
                 free(temp);
         

               //wilsoncurrent62a_petros( operator_list[op_id].prop );
                 wilsonterm_current_density_612ab( operator_list[op_id].prop_zero, TYPE_2, TYPE_A, &temp);
                 for (int ii=0; ii<T_global; ++ii){
                   current1[ii]+=(+1.)*temp[ii           ];
                   current2[ii]+=(+1.)*temp[ii+1*T_global];
                   current3[ii]+=(+1.)*temp[ii+2*T_global];
                   current[ii] +=(+1.)*temp[ii+3*T_global];
                 }
                 free(temp);

                 wilsonterm_current_density_612ab( operator_list[op_id].prop_zero, TYPE_2, TYPE_B, &temp);
                 for (int ii=0; ii<T_global; ++ii){
                   current1[ii]+=(-1.)*temp[ii           ];
                   current2[ii]+=(-1.)*temp[ii+1*T_global];
                   current3[ii]+=(-1.)*temp[ii+2*T_global];
                   current[ii] +=(-1.)*temp[ii+3*T_global];
                 }
                 free(temp);
                 if (g_cart_id == 0){
//                   fprintf(out,"S1S1nontrivialscalar:\n");
                   for (int ii=0; ii<T_global; ++ii){
                     fprintf(out,"JWPL21D1NONTRIVIAL\t%d\t%10.10e\t%10.10e\n", ii, creal(current1[ii]), cimag(current1[ii]));
                   }
//                 fprintf(out,"S2S2nontrivialscalar:\n");
                   for (int ii=0; ii<T_global; ++ii){
                     fprintf(out,"JWPL22D2NONTRIVIAL\t%d\t%10.10e\t%10.10e\n", ii, creal(current2[ii]), cimag(current2[ii]));
                   }
//                 fprintf(out,"PS3PS3nontrivialscalar:\n");
                   for (int ii=0; ii<T_global; ++ii){
                     fprintf(out,"JWPL23D3NONTRIVIAL\t%d\t%10.10e\t%10.10e\n", ii, creal(current3[ii]), cimag(current3[ii]));
                   }
//                 fprintf(out,"SSnontrivialscalar:\n");
                   for (int ii=0; ii<T_global; ++ii){
                     fprintf(out,"JWPL2DNONTRIVIAL\t%d\t%10.10e\t%10.10e\n", ii, creal(current[ii]), cimag(current[ii]));
                   }
                 }
                 for (int ii=0;ii<T_global; ++ii){
                   scalar[ii]=0.0;
                   pseudoscalar[ii]=0.0;
                   current[ii]=0.0;
                   pscalar1[ii]=0.0;
                   pscalar2[ii]=0.0;
                   pscalar3[ii]=0.0;
                   scalar1[ii]=0.0;
                   scalar2[ii]=0.0;
                   scalar3[ii]=0.0;
                   current1[ii]=0.0;
                   current2[ii]=0.0;
                   current3[ii]=0.0;
                 }

               }
//               density_density_1234_petros(operator_list[op_id].prop);
               if (g_cart_id == 0){
                 fclose(out);
               }
               free(scalar);
               free(pseudoscalar);
               free(current);
               free(pscalar1);
               free(pscalar2);
               free(pscalar3);
               free(scalar1);
               free(scalar2);
               free(scalar3);
               free(current1);
               free(current2);
               free(current3);

             } //End of loop over samples

          } //End loop over scalar fields
          if ( ( operator_list[op_id].type == BSM2f ) || ( operator_list[op_id].type == BSM3 )){
             if ( operator_list[op_id].type == BSM2f ){
               free_D_psi_BSM2f();
             }
             else {
               free_D_psi_BSM3();
             }
             for (int ii=0; ii<48; ++ii)
               free(operator_list[op_id].prop_zero[ii]);
             free(operator_list[op_id].prop_zero);
             if ( ( vectorcurrentcurrent_BSM == 1 ) || ( axialcurrentcurrent_BSM == 1 ) ){
               for (int ii=0; ii<48; ++ii)
                 free(operator_list[op_id].prop_ntmone[ii]);
               free(operator_list[op_id].prop_ntmone);
             }
          }

      }//End loop over operators

      nstore+=Nsave;
  }//End of loop over gauges

  finalize_solver(temp_field,2);
  free_gauge_field();
  free_geometry_indices();
  free_bispinor_field();
  free_scalar_field();
  free_spinor_field();
#if defined TM_USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
#endif
}
#endif
