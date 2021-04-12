/* Copyright (C) 2002,2003,2004,2005,2006,2007,2008 Carsten Urbach,
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
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with tmLQCD.	If not, see <http://www.gnu.org/licenses/>.
 *
 *******************************************************************************/

/*******************************************************************************
*
* test program for Frezzotti-Rossi BSM toy model Dslash (D_psi_BSM)
* set variable TEST_INVERSION to 1 for testing the inversion,
* otherwise a simple application of Dslash on a spinor will be tested.
*
*******************************************************************************/
#define TEST_INVERSION 0


#ifdef HAVE_CONFIG_H
# include<tmlqcd_config.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>
#ifdef TM_USE_MPI
# include <mpi.h>
# ifdef HAVE_LIBLEMON
#	include <io/params.h>
#	include <io/gauge.h>
# endif
#endif
#ifdef TM_USE_OMP
# include <omp.h>
# include "init/init_openmp.h"
#endif
#include "gettime.h"
#include "su3.h"
#include "linalg/scalar_prod.h"
#include "linalg/diff.h"
#include "su3adj.h"
#include "ranlxd.h"
#include "geometry_eo.h"
#include "read_input.h"
#include "start.h"
#include "boundary.h"
#include "io/gauge.h"
#ifdef TM_USE_BSM
#include "io/scalar.h"
#endif
#include "global.h"
#include "git_hash.h"
#include "getopt.h"
#include "xchange/xchange.h"
#include "init/init.h"
#include "linalg/assign.h"
#include "operator/D_psi.h"
#ifdef TM_USE_BSM
#include "init/init_scalar_field.h"
#include "init/init_bsm_2hop_lookup.h"
#endif
#include "test/check_geometry.h"
#ifdef TM_USE_BSM
#include "operator/D_psi_BSM2b.h"
#include "operator/D_psi_BSM3.h"
#include "operator/D_psi_BSM3_test.h"
#include "operator/D_psi_BSM2m.h"
#include "operator/M_psi.h"
#include "buffers/utils_nogauge.h"
#include "operator/clovertm_operators.h"
#include "operator/clover_leaf.h"
#endif
#include "mpi_init.h"
#include "measure_gauge_action.h"
#include "buffers/utils.h"
#include "linalg/square_norm.h"
#include "linalg/comp_decomp.h"
#include "linalg/assign_diff_mul.h"
#include "solver/fgmres4bispinors.h"
#include "solver/solver.h"

#ifdef PARALLELT
#	define SLICE (LX*LY*LZ/2)
#elif defined PARALLELXT
#	define SLICE ((LX*LY*LZ/2)+(T*LY*LZ/2))
#elif defined PARALLELXYT
#	define SLICE ((LX*LY*LZ/2)+(T*LY*LZ/2) + (T*LX*LZ/2))
#elif defined PARALLELXYZT
#	define SLICE ((LX*LY*LZ/2)+(T*LY*LZ/2) + (T*LX*LZ/2) + (T*LX*LY/2))
#elif defined PARALLELX
#	define SLICE ((LY*LZ*T/2))
#elif defined PARALLELXY
#	define SLICE ((LY*LZ*T/2) + (LX*LZ*T/2))
#elif defined PARALLELXYZ
#	define SLICE ((LY*LZ*T/2) + (LX*LZ*T/2) + (LX*LY*T/2))
#endif

int check_xchange();

static void usage();
static void process_args(int argc, char *argv[], char ** input_filename, char ** filename);
static void set_default_filenames(char ** input_filename, char ** filename);
#ifndef TM_USE_BSM
int main(int argc,char *argv[])
{
  printf("Works only with BSM operator switched on\n");
}
#else
int main(int argc,char *argv[])
{
  FILE *parameterfile = NULL;
  char datafilename[206];
  char parameterfilename[206];
  char conf_filename[50];
  char scalar_filename[50];
  char * input_filename = NULL;
  char * filename = NULL;
  double plaquette_energy;

#ifdef _USE_HALFSPINOR
#undef _USE_HALFSPINOR
  printf("# WARNING: USE_HALFSPINOR will be ignored (not supported here).\n");
#endif

  if(even_odd_flag)
  {
     even_odd_flag=0;
     printf("# WARNING: even_odd_flag will be ignored (not supported here).\n");
  }
  int j,j_max,k,k_max = 2;
  _Complex double * drvsc;

#ifdef HAVE_LIBLEMON
  paramsXlfInfo *xlfInfo;
#endif
  int status = 0;

  static double t1,t2,dt,sdt,dts,qdt,sqdt;
  double antioptaway=0.0;

  static double dt2;

  DUM_DERI = 6;
  DUM_MATRIX = DUM_DERI+5;
  NO_OF_SPINORFIELDS = DUM_MATRIX+2;


#ifdef TM_USE_MPI
#ifdef TM_USE_OMP
  int mpi_thread_provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_thread_provided);
#else
  MPI_Init(&argc, &argv);
#endif
  MPI_Comm_rank(MPI_COMM_WORLD, &g_proc_id);
#else 
  g_proc_id = 0;
#endif
  g_rgi_C1 = 1.;

  process_args(argc,argv,&input_filename,&filename);
  set_default_filenames(&input_filename, &filename);

  /* Read the input file */
  if( (j = read_input(input_filename)) != 0) {
    fprintf(stderr, "Could not find input file: %s\nAborting...\n", input_filename);
    exit(-1);
  }

  if(g_proc_id==0) {
    printf("parameter rho_BSM set to %f\n", rho_BSM);
    printf("parameter eta_BSM set to %f\n", eta_BSM);
    printf("parameter  m0_BSM set to %f\n",  m0_BSM);
    printf("parameter mu03_BSM set to %f\n", mu03_BSM);
    printf("parameter mu01_BSM set to %f\n", mu01_BSM);
  }

#ifdef TM_USE_OMP
  init_openmp();
#endif
     

  tmlqcd_mpi_init(argc, argv);


  if(g_proc_id==0) {
#ifdef SSE
     printf("# The code was compiled with SSE instructions\n");
#endif
#ifdef SSE2
     printf("# The code was compiled with SSE2 instructions\n");
#endif
#ifdef SSE3
     printf("# The code was compiled with SSE3 instructions\n");
#endif
#ifdef P4
     printf("# The code was compiled for Pentium4\n");
#endif
#ifdef OPTERON
     printf("# The code was compiled for AMD Opteron\n");
#endif
#ifdef _GAUGE_COPY
     printf("# The code was compiled with -D_GAUGE_COPY\n");
#endif
#ifdef BGL
     printf("# The code was compiled for Blue Gene/L\n");
#endif
#ifdef BGP
     printf("# The code was compiled for Blue Gene/P\n");
#endif
#ifdef _USE_HALFSPINOR
     printf("# The code was compiled with -D_USE_HALFSPINOR\n");
#endif
#ifdef _USE_SHMEM
     printf("# The code was compiled with -D_USE_SHMEM\n");
#ifdef _PERSISTENT
     printf("# The code was compiled for persistent MPI calls (halfspinor only)\n");
#endif
#ifdef TM_USE_BSM
     printf("# The code was compiled for persistent for BSM operators i.e. we are using two gauge fields\n");
#endif
#endif
#ifdef TM_USE_MPI
#ifdef _NON_BLOCKING
     printf("# The code was compiled for non-blocking MPI calls (spinor and gauge)\n");
#endif
#endif
     printf("\n");
     fflush(stdout);
  }

#ifdef TM_USE_BSM

#ifdef _GAUGE_COPY
  init_gauge_field(VOLUMEPLUSRAND + g_dbw2rand, 1);
#else
  init_gauge_field(VOLUMEPLUSRAND + g_dbw2rand, 0);
#endif
  init_geometry_indices(VOLUMEPLUSRAND + g_dbw2rand);


  j = init_bispinor_field(VOLUMEPLUSRAND, 12);
  if ( j!= 0) {
     fprintf(stderr, "Not enough memory for bispinor fields! Aborting...\n");
     exit(0);  
  }

  j = init_spinor_field(VOLUMEPLUSRAND, NO_OF_SPINORFIELDS+1);
  if ( j!= 0) {
    fprintf(stderr, "Not enough memory for spinor fields! Aborting...\n");
    exit(0);
  }

  int numbScalarFields = 4;
  j = init_scalar_field(VOLUMEPLUSRAND, numbScalarFields);
  if ( j!= 0) {
     fprintf(stderr, "Not enough memory for scalar fields! Aborting...\n");
		exit(0);
  }
 
  drvsc = malloc(18*VOLUMEPLUSRAND*sizeof(_Complex double));

  if(g_proc_id == 0) {
    fprintf(stdout,"# The number of processes is %d \n",g_nproc);
    fprintf(stdout,"# The lattice size is %d x %d x %d x %d\n",
		 (int)(T*g_nproc_t), (int)(LX*g_nproc_x), (int)(LY*g_nproc_y), (int)(g_nproc_z*LZ));
    fprintf(stdout,"# The local lattice size is %d x %d x %d x %d\n",
		 (int)(T), (int)(LX), (int)(LY),(int) LZ);

    fflush(stdout); 
  }

  /* define the geometry */
  geometry();

  /* define the boundary conditions for the fermion fields */
  /* for the actual inversion, this is done in invert.c as the operators are iterated through */
  // 
  // For the BSM operator we don't use kappa normalisation,
  // as a result, when twisted boundary conditions are applied this needs to be unity.
  // In addition, unlike in the Wilson case, the hopping term comes with a plus sign.
  // However, in boundary(), the minus sign for the Wilson case is implicitly included.
  // We therefore use -1.0 here.
  boundary(-1.0);

  status = check_geometry();
  if (status != 0) {
    fprintf(stderr, "Checking of geometry failed. Unable to proceed.\nAborting....\n");
    exit(1);
  }
  init_dirac_halfspinor();
  init_D_psi_BSM3();
//#if (defined MPI && !(defined _USE_SHMEM))
// fails, we're not using spinor fields
//	check_xchange();
//#endif

  start_ranlux(1, 123456);

  // read gauge field
  if( strcmp(gauge_input_filename, "create_random_gaugefield") == 0 ) {

    //Creating random field for the chitilde breaking part
    random_gauge_field(reproduce_randomnumber_flag, g_gauge_field);

    //Creating random field for the wilson part part
    random_gauge_field(reproduce_randomnumber_flag, g_smeared_gauge_field);

  }
  else {
    snprintf(conf_filename, 50, "%s.%.4d", gauge_input_filename, nstore);
    if (g_cart_id == 0) {
      printf("#\n# Trying to read gauge field from file %s in %s precision.\n",
		conf_filename, (gauge_precision_read_flag == 32 ? "single" : "double"));
      fflush(stdout);
    }

    int i;
    if( (i = read_gauge_field(conf_filename,g_gauge_field)) !=0) {
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
  }

  // read scalar field
  if( strcmp(scalar_input_filename, "create_random_scalarfield") == 0 ) {
    for( int s=0; s<numbScalarFields; s++ )
      ranlxd(g_scalar_field[s], VOLUME);
  }
  else if( strcmp(scalar_input_filename, "create_trivial_scalarfield") == 0 ) {
    for(int s=0;s<VOLUME;s++){
      g_scalar_field[0][s]=1.0;
      g_scalar_field[1][s]=0;
      g_scalar_field[2][s]=0;
      g_scalar_field[3][s]=0;
    }
  }
  else {
    sprintf(scalar_filename, "%s.%d", scalar_input_filename, nscalar);
    if (g_cart_id == 0) {
      printf("#\n# Trying to read scalar field from file %s in %s precision.\n",
	scalar_filename, (scalar_precision_read_flag == 32 ? "single" : "double"));
        fflush(stdout);
    }

    int i;
    if( (i = read_scalar_field_parallel(scalar_filename,g_scalar_field)) !=0) {
      fprintf(stderr, "Error %d while reading scalar field from %s\n Aborting...\n", i, scalar_filename);
      exit(-2);
    }

    if (g_cart_id == 0) {
      printf("# Finished reading scalar field.\n");
      fflush(stdout);
    }
  }

#ifdef TM_USE_MPI
  xchange_gauge(g_gauge_field);
  xchange_gauge(g_smeared_gauge_field);
#endif

  /*compute the energy of the gauge field*/
  plaquette_energy = measure_plaquette( (const su3**) g_gauge_field);

  if (g_cart_id == 0) {
    printf("# The computed plaquette for the unsmeared gauge field value is %e.\n", plaquette_energy / (6.*VOLUME*g_nproc));
    fflush(stdout);
  }

  plaquette_energy = measure_plaquette( (const su3**) g_smeared_gauge_field);

  if (g_cart_id == 0) {
    printf("# The computed plaquette for the  smeared gauge field value is %e.\n", plaquette_energy / (6.*VOLUME*g_nproc));
    fflush(stdout);
  }


#if defined TM_USE_MPI
  for( int s=0; s<numbScalarFields; s++ )
    generic_exchange_nogauge(g_scalar_field[s], sizeof(scalar));
#endif

  if (g_cart_id == 0) {
    printf("# Scalar send is done\n");
    fflush(stdout);
  }

  /*initialize the bispinor fields*/
  j_max=1;
  sdt=0.;
  // w
  random_spinor_field_lexic( (spinor*)(g_bispinor_field[4]), reproduce_randomnumber_flag, RN_GAUSS);
  random_spinor_field_lexic( (spinor*)(g_bispinor_field[4])+VOLUME, reproduce_randomnumber_flag, RN_GAUSS);
        
  if (g_cart_id == 0) {
   printf("# Random fields for fermion generated\n");
   fflush(stdout);
  }

  // for the D^\dagger test:
  random_spinor_field_lexic( (spinor*)(g_bispinor_field[5])+VOLUME, reproduce_randomnumber_flag, RN_GAUSS);
#if defined TM_USE_MPI
  generic_exchange(g_bispinor_field[4], sizeof(bispinor));
  generic_exchange(g_bispinor_field[5], sizeof(bispinor));
#endif

  double squarenorm_w;

  init_sw_fields(VOLUME);
/*************************************************************************************************************
 *
 *
 *           Testing for the correct implementation of dagger of the Dslash 
 *
 ************************************************************************************************************/

  printf("# [tmlqcd-BSM test] First we test the correct implementation of the dagger of D_psi_BSM3\n");      

  double t_FP;

#ifdef TM_USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  t_FP = 0.0;
  t1 = gettime();
  D_psi_BSM3(g_bispinor_field[2], g_bispinor_field[4]);
  t1 = gettime() - t1;
#ifdef TM_USE_MPI
  MPI_Allreduce (&t1, &t_FP, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#else
  t_FP = t1;
#endif

#ifdef TM_USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  t_FP = 0.0;
  t1 = gettime();
  D_psi_dagger_BSM3(g_bispinor_field[1], g_bispinor_field[5]);
  t1 = gettime() - t1;
#ifdef TM_USE_MPI
  MPI_Allreduce (&t1, &t_FP, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#else
  t_FP = t1;
#endif

  _Complex double prod2_FP_vw  = scalar_prod((spinor*)g_bispinor_field[5], (spinor*)g_bispinor_field[2], 2*VOLUME, 1);
  _Complex double prod2_FP_wdv = scalar_prod((spinor*)g_bispinor_field[4], (spinor*)g_bispinor_field[1], 2*VOLUME, 1);

  if( g_proc_id == 0 ){
          printf("# [tmlqcd-BSM test] < v, D_FP w > = %.16e + I*(%.16e)\n", creal(prod2_FP_vw), cimag(prod2_FP_vw));
          printf("# [tmlqcd-BSM test] < w, D_FP^dagger v > = %.16e + I*(%.16e)\n\n", creal(prod2_FP_wdv), cimag(prod2_FP_wdv));

  }

#if TEST_INVERSION
  if(g_proc_id==1)
    printf("Testing inversion\n");
  // Feri's operator
  assign_add_mul((spinor*)g_bispinor_field[10], (spinor*)g_bispinor_field[5], 1.0, 2*VOLUME);
  printf("Starting field %e\n",creal(g_bispinor_field[5][0].sp_up.s0.c0));
  printf("Starting field %e\n",creal(g_bispinor_field[10][0].sp_up.s0.c0));
  t1 = gettime();
        cg_her_bi(g_bispinor_field[2], g_bispinor_field[5],
           25000, 1.0e-14, 0, VOLUME, &Q2_psi_BSM3);
  t_FP = gettime() - t1;

  if(g_proc_id==0)
    printf("Operator inversion time: t_FP = %f sec\n\n", t_FP);
  

  Q2_psi_BSM3(g_bispinor_field[9], g_bispinor_field[2]);
  printf("Starting field %e\n",creal(g_bispinor_field[10][0].sp_up.s0.c0));
  assign_diff_mul((spinor*)g_bispinor_field[9], (spinor*)g_bispinor_field[10], 1.0, 2*VOLUME);
  
  double squarenorm_FP = square_norm((spinor*)g_bispinor_field[9], 2*VOLUME, 1);
  if(g_proc_id==0) {
     printf("# ||Q2_FP*(Q2_FP)^-1*(b)-b||^2 = %.16e\n\n", squarenorm_FP);
     fflush(stdout);
  }
#endif

  eta_BSM=0.5;
  rho_BSM=0.5;

  csw_BSM=0;  /* Also original BSM operator is without the clover term */
  r0_BSM=0;
  //sw_term( (const su3**) g_smeared_gauge_field, 1, csw_BSM);
  
  random_spinor_field_lexic( (spinor*)(g_bispinor_field[4]), reproduce_randomnumber_flag, RN_GAUSS);
  random_spinor_field_lexic( (spinor*)(g_bispinor_field[4])+VOLUME, reproduce_randomnumber_flag, RN_GAUSS);


#if defined TM_USE_MPI
  generic_exchange(g_bispinor_field[4], sizeof(bispinor));
#endif
/**************************************************************************************************************************
 *
 * Testing the new BSM operator with respect to the old one (r0_BSM=0)
 *
 * ************************************************************************************************************************/
  printf("# [tmlqcd-BSM test] Testing the new BSM operator with respect to the old one at r0_BSM=0\n");

 // print L2-norm of w source:
  squarenorm_w = square_norm((spinor*)g_bispinor_field[4], 2*VOLUME, 1);
  if(g_proc_id==0) {
   printf("# [tmlqcd-BSM test] input source vector for application D_BSM3: square norm of the source: ||w||^2 = %e\n", squarenorm_w);
   fflush(stdout);
  }

  // Feri's operator

  printf("# [tmlqcd-BSM test] Application D_psi_BSM3 operator\n");
#ifdef TM_USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  t_FP = 0.0;
  t1 = gettime();
  printf("r0_BSM = %e\n",r0_BSM);
  D_psi_BSM3_test(g_bispinor_field[2], g_bispinor_field[4]);
  t1 = gettime() - t1;
#ifdef TM_USE_MPI
  MPI_Allreduce (&t1, &t_FP, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#else
  t_FP = t1;
#endif
  squarenorm_w = square_norm((spinor*)g_bispinor_field[2], 2*VOLUME, 1);
  if(g_proc_id==0) {
   printf("# [tmlqcd-BSM test] square norm of the multiplication results with D_BSM3: ||w||^2 = %e\n", squarenorm_w);
   fflush(stdout);
  }

#ifdef TM_USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  squarenorm_w = square_norm((spinor*)g_bispinor_field[4], 2*VOLUME, 1);
  if(g_proc_id==0) {
   printf("# [tmlqcd-BSM test] square norm of the source for D_psi_BSM : ||w||^2 = %e\n", squarenorm_w);
   fflush(stdout);
  }

  printf("# [tmlqcd-BSM test] Application of D_psi_BSM  ( Carsten' operator )\n");
  t_FP = 0.0;
  t1 = gettime();
  D_psi_BSM(g_bispinor_field[5], g_bispinor_field[4]);
  t1 = gettime() - t1;
#ifdef TM_USE_MPI
  MPI_Allreduce (&t1, &t_FP, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#else
  t_FP = t1;
#endif
  squarenorm_w = square_norm((spinor*)g_bispinor_field[5], 2*VOLUME, 1);
  if(g_proc_id==0) {
   printf("# [tmlqcd-BSM test] square norm of the multiplication results with D_psi_BSM:  ||w||^2 = %e\n", squarenorm_w);
   fflush(stdout);
  }

  assign_diff_mul((spinor *)g_bispinor_field[5], (spinor *)g_bispinor_field[2],1.0,  2*VOLUME ); 

  squarenorm_w = square_norm((spinor*)g_bispinor_field[5], 2*VOLUME, 1);
  if(g_proc_id==0) {
   printf("# [tmlqcd-BSM test] square norm of the difference D_psi_BSM - D_psi_BSM3:  ||w||^2 = %e\n\n", squarenorm_w);
   fflush(stdout);
  }

/***********************************************************************************************************
 *
 * Testing D_psi_BSM3 Dirac operator gives the same results as the usual 
 * wilson operator at rho=eta=0,csw=0
 *
 *
 *********************************************************************************************************/
 // print L2-norm of w source:
  bispinor_assign(g_bispinor_field[5],g_bispinor_field[4], VOLUME);
  if(g_proc_id==0) {
   printf("# [tmlqcd-BSM test] Test the compatibility with the Wilson operator implemented, BSM parameters zero(mu,rho,eta), r0_BSM=1\n");
   fflush(stdout);
  }

  squarenorm_w = square_norm((spinor*)g_bispinor_field[4], 2*VOLUME, 1);
  if(g_proc_id==0) {
   printf("# [tmlqcd-BSM test] input source vector for application D_BSM3: square norm of the source: ||w||^2 = %e\n", squarenorm_w);
   fflush(stdout);
  }
  // Feri's operator
  csw_BSM=0;
  m0_BSM=0;
  mu03_BSM=0;
  eta_BSM=0.;
  rho_BSM=0;

  printf("# [tmlqcd-BSM test] csw_BSM %e m0_BSM %e mu03_BSM %e eta_BSM %e rho_BSM \n", csw_BSM, m0_BSM, mu03_BSM, eta_BSM, rho_BSM);
  g_mu=0;
  g_c_sw=0;
  g_kappa=1;

  printf("# [tmlqcd-BSM test] g_mu %e g_c_sw %e g_kappa %e\n", g_mu,g_c_sw,g_kappa);
  printf("# [tmlqcd-BSM test] application of D_psi_BSM3 operator\n");
#ifdef TM_USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  t_FP = 0.0;
  t1 = gettime();
  D_psi_BSM3(g_bispinor_field[2], g_bispinor_field[4]);
  t1 = gettime() - t1;
#ifdef TM_USE_MPI
  MPI_Allreduce (&t1, &t_FP, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#else
  t_FP = t1;
#endif
  squarenorm_w = square_norm((spinor*)g_bispinor_field[2], 2*VOLUME, 1);
  if(g_proc_id==0) {
   printf("# [tmlqcd-BSM test] square norm of the multiplication results with D_BSM3: ||w||^2 = %e\n", squarenorm_w);
   fflush(stdout);
  }

  squarenorm_w = square_norm((spinor*)g_bispinor_field[4], 2*VOLUME, 1);
  if(g_proc_id==0) {
   printf("# [tmlqcd-BSM test] input source vector for application D_psi: square norm of the source: ||w||^2 = %e\n", squarenorm_w);
   fflush(stdout);
  }


  printf("# [tmlqcd-BSM test] Application of D_psi ( Carsten' operator ) \n");
  D_psi_bispinor((bispinor *)(g_bispinor_field[5]), (bispinor *)(g_bispinor_field[4]));

  squarenorm_w = square_norm((spinor*)g_bispinor_field[5], 2*VOLUME, 1);
  if(g_proc_id==0) {
   printf("# [tmlqcd-BSM test] square norm of the multiplication results with D_psi_bispinor:  ||w||^2 = %e\n", squarenorm_w);
   fflush(stdout);
  }

  assign_diff_mul((spinor *)g_bispinor_field[5], (spinor *)g_bispinor_field[2],1.0,  2*VOLUME );

  squarenorm_w = square_norm((spinor*)g_bispinor_field[5], 2*VOLUME, 1);
  if(g_proc_id==0) {
   printf("# [tmlqcd-BSM test] square norm of the difference D_psi - D_psi_BSM3:  ||w||^2 = %e\n\n", squarenorm_w);
   fflush(stdout);
  }

/******************************************************************************************************************************
 *
 *
 *  Test the compatibility with the Clover Wilson operator implemented, BSM parameters zero(mu,rho,eta), r0_BSM=1,csw=1,kappa=1
 *
 *
 ******************************************************************************************************************************/

  if(g_proc_id==0) {
   printf("# [tmlqcd-BSM test] Test the compatibility with the Clover Wilson operator implemented, BSM parameters zero(mu,rho,eta), r0_BSM=1,csw=1,kappa=1\n");
   fflush(stdout);
  }

  csw_BSM=1;  /* Also the new BSM operator is with the clover term */
  sw_term( (const su3**) g_smeared_gauge_field, 1., csw_BSM/2.);
  bispinor_assign(g_bispinor_field[5],g_bispinor_field[4], VOLUME);
 
  printf("# [tmlqcd-BSM test] application of D_psi_BSM3 operator\n");
#ifdef TM_USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  t_FP = 0.0;
  t1 = gettime();
  D_psi_BSM3(g_bispinor_field[2], g_bispinor_field[4]);
  t1 = gettime() - t1;
#ifdef TM_USE_MPI
  MPI_Allreduce (&t1, &t_FP, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#else
  t_FP = t1;
#endif
  squarenorm_w = square_norm((spinor*)g_bispinor_field[2], 2*VOLUME, 1);
  if(g_proc_id==0) {
   printf("# [tmlqcd-BSM test] square norm of the multiplication results with D_BSM3: ||w||^2 = %e\n", squarenorm_w);
   fflush(stdout);
  }

  g_c_sw=1;
  g_kappa=1;
  sw_term( (const su3**) g_gauge_field, 1., g_c_sw/2);

  printf("# [tmlqcd-BSM test] Application of D_psi ( Carsten' operator ) \n");

  D_psi_bispinor((bispinor *)(g_bispinor_field[3]), (bispinor *)(g_bispinor_field[4]));

  squarenorm_w = square_norm((spinor*)g_bispinor_field[3], 2*VOLUME, 1);
  if(g_proc_id==0) {
   printf("# [tmlqcd-BSM test] square norm of the multiplication results with D_psi_bispinor:  ||w||^2 = %e\n", squarenorm_w);
   fflush(stdout);
  }

  assign_diff_mul((spinor *)g_bispinor_field[3], (spinor *)g_bispinor_field[2],1.0,  2*VOLUME );

  squarenorm_w = square_norm((spinor*)g_bispinor_field[3], 2*VOLUME, 1);
  if(g_proc_id==0) {
   printf("# [tmlqcd-BSM test] square norm of the difference D_psi - D_psi_BSM3:  ||w||^2 = %e\n", squarenorm_w);
   fflush(stdout);
  }

  free_D_psi_BSM3();
  free_bispinor_field();
  free_scalar_field();

  free_gauge_field();
  free_geometry_indices();

#endif //_USE_BSM
  

#ifdef TM_USE_OMP
  free_omp_accumulators();
#endif 
#ifdef TM_USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
#endif
  return(0);
}


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
#endif

