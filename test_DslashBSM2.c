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
# include<config.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>
#ifdef MPI
# include <mpi.h>
# ifdef HAVE_LIBLEMON
#	include <io/params.h>
#	include <io/gauge.h>
# endif
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
#include "io/scalar.h"
#include "global.h"
#include "getopt.h"
#include "xchange/xchange.h"
#include "init/init.h"
#include "init/init_scalar_field.h"
#include "init/init_bsm_2hop_lookup.h"
#include "test/check_geometry.h"
#include "operator/D_psi_BSM2b.h"
#include "operator/D_psi_BSM2m.h"
#include "operator/D_psi_BSM2f.h"
#include "solver/eigenvalues_krylov.h"
#include "operator/M_psi.h"
#include "mpi_init.h"
#include "measure_gauge_action.h"
#include "buffers/utils.h"
#include "linalg/square_norm.h"
#include "linalg/comp_decomp.h"
#include "linalg/assign_diff_mul.h"
#include "solver/fgmres4bispinors.h"
#include "solver/solver.h"
#include "solver/eigenvalues_bi.h"
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

//int check_xchange();

static void usage();
static void process_args(int argc, char *argv[], char ** input_filename, char ** filename);
static void set_default_filenames(char ** input_filename, char ** filename);

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
//	_Complex double * drvsc;

#ifdef HAVE_LIBLEMON
	paramsXlfInfo *xlfInfo;
#endif
	int status = 0;

	static double t1,t2,dt,sdt,dts,qdt,sqdt;
	double antioptaway=0.0;

#ifdef MPI
	static double dt2;

	DUM_DERI = 6;
	DUM_SOLVER = DUM_DERI+2;
	DUM_MATRIX = DUM_SOLVER+6;
	NO_OF_SPINORFIELDS = DUM_MATRIX+2;

	MPI_Init(&argc, &argv);

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
	}

#ifdef OMP
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
#endif
#ifdef MPI
	#ifdef _NON_BLOCKING
		printf("# The code was compiled for non-blocking MPI calls (spinor and gauge)\n");
	#endif
#endif
		printf("\n");
		fflush(stdout);
	}


#ifdef _GAUGE_COPY
	init_gauge_field(VOLUMEPLUSRAND + g_dbw2rand, 1);
#else
	init_gauge_field(VOLUMEPLUSRAND + g_dbw2rand, 0);
#endif
	init_geometry_indices(VOLUMEPLUSRAND + g_dbw2rand);


	j = init_bispinor_field(VOLUMEPLUSRAND, 28);
	if ( j!= 0) {
		fprintf(stderr, "Not enough memory for bispinor fields! Aborting...\n");
		exit(0);
	}

	j = init_spinor_field(VOLUMEPLUSRAND, 29);
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

	//drvsc = malloc(18*VOLUMEPLUSRAND*sizeof(_Complex double));

	if(g_proc_id == 0) {
		fprintf(stdout,"# The number of processes is %d \n",g_nproc);
		printf("# The lattice size is %d x %d x %d x %d\n",
		 (int)(T*g_nproc_t), (int)(LX*g_nproc_x), (int)(LY*g_nproc_y), (int)(g_nproc_z*LZ));
		printf("# The local lattice size is %d x %d x %d x %d\n",
		 (int)(T), (int)(LX), (int)(LY),(int) LZ);

		fflush(stdout);
	}

	/* define the geometry */
	geometry();

    //    j = init_bsm_2hop_lookup(VOLUME);
    //	if ( j!= 0) {
    // this should not be reached since the init function calls fatal_error anyway
    //		fprintf(stderr, "Not enough memory for BSM2b 2hop lookup table! Aborting...\n");
    //		exit(0);
    //	}

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
#if (defined MPI && !(defined _USE_SHMEM))
	// fails, we're not using spinor fields
//	check_xchange();
#endif

	start_ranlux(1, 123456);

	// read gauge field
	if( strcmp(gauge_input_filename, "create_random_gaugefield") == 0 ) {
		random_gauge_field(reproduce_randomnumber_flag, g_gauge_field);
	}
	else {
		sprintf(conf_filename, "%s.%.4d", gauge_input_filename, nstore);
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

#ifdef MPI
    xchange_gauge(g_gauge_field);
#endif

   
    /*compute the energy of the gauge field*/
    plaquette_energy = measure_plaquette( (const su3**) g_gauge_field);

    
    if (g_cart_id == 0) {
      printf("# The computed plaquette value is %e.\n", plaquette_energy / (6.*VOLUME*g_nproc));
      fflush(stdout);
    }
   

#ifdef MPI
        // printf("Starting generic exchange routines for the scalar field\n");
	for( int s=0; s<numbScalarFields; s++ )
            generic_exchange_nogauge(g_scalar_field[s], sizeof(scalar));

#endif

	/*initialize the bispinor fields*/
	j_max=1;
	sdt=0.;
  // w/
	unit_spinor_field_lexic( (spinor*)(g_bispinor_field[4]));//, 	  reproduce_randomnumber_flag, RN_GAUSS);
	unit_spinor_field_lexic( (spinor*)(g_bispinor_field[4])+VOLUME);//, reproduce_randomnumber_flag, RN_GAUSS);
	// for the D^\dagger test:
  // v
	unit_spinor_field_lexic( (spinor*)(g_bispinor_field[5]));//,	  reproduce_randomnumber_flag, RN_GAUSS);
	unit_spinor_field_lexic( (spinor*)(g_bispinor_field[5])+VOLUME);//, reproduce_randomnumber_flag, RN_GAUSS);
#if defined MPI
	generic_exchange_nogauge(g_bispinor_field[4], sizeof(bispinor));
//        if (g_proc_id == 0  ) printf("Generic exchange for the spinor fields are done\n");
#endif

	// print L2-norm of source:
	double squarenorm = square_norm((spinor*)g_bispinor_field[4], 2*VOLUME, 1);
	if (g_proc_id == 0 ) 
        {
		printf("\n# square norm of the source: ||w||^2 = %e\n\n", squarenorm);
		fflush(stdout);
	}

////  eigmax();
  double t_F;
	/* inversion needs to be done first because it uses loads of the g_bispinor_fields internally */
#if TEST_INVERSION
  if(g_proc_id==1)
    printf("Testing inversion\n");
  // Feri's operator
  t1 = gettime();
	cg_her_bi(g_bispinor_field[9], g_bispinor_field[4],
           25000, 1.0e-14, 0, VOLUME, &Q2_psi_BSM2f);
  t_F = gettime() - t1;

  if(g_proc_id==0)
    printf("Operator inversion time: t_F = %f sec \n\n", t_F); 
#endif

  /* now apply the operators to the same bispinor field and do various comparisons */

  // Feri's operator
#ifdef MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif
  t_F = 0.0;
  t1 = gettime();
  D_psi_BSM2f(g_bispinor_field[0], g_bispinor_field[4]);
  t1 = gettime() - t1;
#ifdef MPI
	MPI_Allreduce (&t1, &t_F, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#else
  t_F = t1;
#endif
  if (g_proc_id == 0 ) printf("time %10.10f\n", t_F);
  
  if(g_proc_id==0)
    printf("Operator application time: t_F = %f sec \n\n", t_F); 

  squarenorm = square_norm((spinor*)g_bispinor_field[0], 2*VOLUME, 1);
  if(g_proc_id==0) {
    printf("# || D_F w ||^2 = %.16e\n", squarenorm);
    fflush(stdout);
  }

	// < D w, v >
  printf("Check consistency of D and D^dagger\n");
  _Complex double prod1_F = scalar_prod( (spinor*)g_bispinor_field[0], (spinor*)g_bispinor_field[5], 2*VOLUME, 1 );
	if(g_proc_id==0)
    printf("< D_F w, v >        = %.16e + I*(%.16e)\n", creal(prod1_F), cimag(prod1_F));
	
	
  // < w, D^\dagger v >
  t_F = gettime();
	D_psi_dagger_BSM2f(g_bispinor_field[6], g_bispinor_field[5]);
  t_F = gettime()-t_F;

  if(g_proc_id==0)
    printf("Operator dagger application time: t_F = %f sec \t \n", t_F); 

	_Complex double prod2_F = scalar_prod((spinor*)g_bispinor_field[4], (spinor*)g_bispinor_field[6], 2*VOLUME, 1);
  if( g_proc_id == 0 ){
	  printf("< w, D_F^dagger v > = %.16e + I*(%.16e)\n", creal(prod2_F), cimag(prod2_F));
	  
          printf("\n| < D_F w, v > - < w, D_F^dagger v > | = %.16e\n",cabs(prod2_F-prod1_F));
  }
	
  free_gauge_field();
  free_geometry_indices();
  free_bispinor_field();
  free_scalar_field();
#ifdef MPI
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

