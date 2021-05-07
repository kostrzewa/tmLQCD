/***********************************************************************
 *
 * Copyright (C) 2002,2003,2004,2005,2006,2007,2008 Carsten Urbach
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
 *
 * invert for twisted mass QCD
 *
 * Author: Carsten Urbach
 *         urbach@physik.fu-berlin.de
 *
 *******************************************************************************/

#include"lime.h"
#ifdef HAVE_CONFIG_H
# include<config.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include <signal.h>
#ifdef TM_USE_MPI
#include <mpi.h>
#endif
#ifdef TM_USE_OMP
# include <omp.h>
#endif
#include "global.h"
#include "git_hash.h"
#include "getopt.h"
#include "linalg_eo.h"
#include "geometry_eo.h"
#include "start.h"
/*#include "eigenvalues.h"*/
#include "measure_gauge_action.h"
#ifdef TM_USE_MPI
#include "xchange/xchange.h"
#endif
#include <io/utils.h>
#include <io/scalar.h>
#include "read_input.h"
#include "mpi_init.h"
#include "sighandler.h"
#include "boundary.h"
#include "solver/solver.h"
#include "solver/solver_field.h"
#include "init/init.h"
#include "smearing/stout.h"
#include "invert_eo.h"
#include "monomial/monomial.h"
#include "ranlxd.h"
#include "phmc.h"
#include "operator/D_psi.h"
#include "little_D.h"
#include "reweighting_factor.h"
#include "linalg/convert_eo_to_lexic.h"
#include "block.h"
#include "operator.h"
#include "sighandler.h"
#include "solver/dfl_projector.h"
#include "solver/generate_dfl_subspace.h"
#include "prepare_source.h"
#include <io/params.h>
#include <io/gauge.h>
#include <io/spinor.h>
#include <io/utils.h>
#include "solver/dirac_operator_eigenvectors.h"
#include "operator/tm_operators.h"
#include "operator/Dov_psi.h"
#include "solver/spectral_proj.h"

extern int nstore;
int check_geometry();

static void usage();
static void process_args(int argc, char *argv[], char ** input_filename, char ** filename);
static void set_default_filenames(char ** input_filename, char ** filename);

int main(int argc, char *argv[])
{
  FILE *parameterfile = NULL;
  int j, i, ix = 0, isample = 0, op_id = 0;
  char datafilename[206];
  char parameterfilename[206];
  char conf_filename[50];
  char scalar_filename[50];
  char * input_filename = NULL;
  char * filename = NULL;
  double plaquette_energy;
  struct stout_parameters params_smear;
  spinor **s, *s_;

#ifdef _KOJAK_INST
#pragma pomp inst init
#pragma pomp inst begin(main)
#endif

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

  verbose = 0;
  g_use_clover_flag = 0;

#ifdef TM_USE_MPI

#  ifdef TM_USE_OMP
  int mpi_thread_provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_thread_provided);
#  else
  MPI_Init(&argc, &argv);
#  endif

  MPI_Comm_rank(MPI_COMM_WORLD, &g_proc_id);
#else
  g_proc_id = 0;
#endif

  process_args(argc,argv,&input_filename,&filename);
  set_default_filenames(&input_filename, &filename);

  /* Read the input file */
  if( (j = read_input(input_filename)) != 0) {
    fprintf(stderr, "Could not find input file: %s\nAborting...\n", input_filename);
    exit(-1);
  }

#ifdef TM_USE_OMP
  init_openmp();
#endif

  if (Nsave == 0) {
    Nsave = 1;
  }

  tmlqcd_mpi_init(argc, argv);

  if (even_odd_flag) {
    j = init_spinor_field(VOLUMEPLUSRAND / 2, NO_OF_SPINORFIELDS);
  }
  else {
    j = init_spinor_field(VOLUMEPLUSRAND, NO_OF_SPINORFIELDS);
  }
  if (j != 0) {
    fprintf(stderr, "Not enough memory for spinor fields! Aborting...\n");
    exit(-1);
  }
  j = init_geometry_indices(VOLUMEPLUSRAND);                                                                                                                                        
  if (j != 0) {
    fprintf(stderr, "Not enough memory for geometry indices! Aborting...\n");
    exit(-1);
  }
  /* define the geometry */
  geometry();

  spinor ** temp_field = NULL;
  init_solver_field(&temp_field, VOLUMEPLUSRAND, 1);

  int t[] = {0,4,20,23};
  int x[] = {0,8,3,13};
  int y[] = {0,12,6,11};
  int z[] = {0,2,4,8};
  
  char prop_fname[200];

  for( int src_idx = 0; src_idx < 12; src_idx++ ){
    snprintf(prop_fname,200,"bsm2prop.0400.00.%02d.000.inverted",src_idx);
    for(int pos = 0; pos < 8; pos++){
      printf("READCHECK: Propagator in pos %02d from file %s\n", pos, prop_fname);
      read_spinor(g_spinor_field[0], g_spinor_field[1], prop_fname, pos);
      convert_eo_to_lexic(temp_field[0], g_spinor_field[0], g_spinor_field[1]);
      for( int i = 0; i < 4; i++){
        int idx = g_ipt[ t[i] ][ x[i] ][ y[i] ][ z[i] ]; 
        _Complex double * sp = (_Complex double*)&(temp_field[0][idx]);
        printf("READCHECK: t:%02d, x:%02d, y:%02d, z:%02d\n",t[i],x[i],y[i],z[i]);
        for(int s = 0; s < 4; ++s){
          printf("READCHECK: s%d: ", s);
          for(int c = 0; c < 3; ++c){
            unsigned int offset = s*3 + c;
            printf("c%d: %+.7e + i*(%+.7e) \t", c, creal(*(sp+offset)), cimag(*(sp+offset)) );
          }
          printf("\n");
        }
      }
    printf("READCHECK \n\n\n");
    }
  }

  finalize_solver(temp_field,1);

#ifdef TM_USE_OMP
  free_omp_accumulators();
#endif
  free_blocks();
  free_dfl_subspace();
  free_gauge_field();
  free_scalar_field();
  free_geometry_indices();
  free_spinor_field();
  free_moment_field();
  free_chi_spinor_field();
  free(filename);
  free(input_filename);
#ifdef TM_USE_MPI
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
#endif
  return(0);
#ifdef _KOJAK_INST
#pragma pomp inst end(main)
#endif
}

static void usage()
{
  fprintf(stdout, "Inversion for EO preconditioned Wilson twisted mass QCD\n");
  fprintf(stdout, "Version %s \n\n", PACKAGE_VERSION);
  fprintf(stdout, "Please send bug reports to %s\n", PACKAGE_BUGREPORT);
  fprintf(stdout, "Usage:   invert [options]\n");
  fprintf(stdout, "Options: [-f input-filename]\n");
  fprintf(stdout, "         [-o output-filename]\n");
  fprintf(stdout, "         [-v] more verbosity\n");
  fprintf(stdout, "         [-h|-? this help]\n");
  fprintf(stdout, "         [-V] print version information and exit\n");
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
          fprintf(stdout,"%s %s\n",PACKAGE_STRING,git_hash);
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
    *input_filename = calloc(20, sizeof(char));
    strcpy(*input_filename,"prop_io_test.input");
  }
  
  if( *filename == NULL ) {
    *filename = calloc(7, sizeof(char));
    strcpy(*filename,"output");
  } 
}

