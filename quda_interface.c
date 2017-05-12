/***********************************************************************
 *
 * Copyright (C) 2015 Mario Schroeck
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
 *
 ***********************************************************************/
/***********************************************************************
*
* File quda_interface.h
*
* Author: Mario Schroeck <mario.schroeck@roma3.infn.it>
* 
* Last changes: 06/2015
*
*
* Interface to QUDA for multi-GPU inverters
*
* The externally accessible functions are
*
*   void _initQuda()
*     Initializes the QUDA library. Carries over the lattice size and the
*     MPI process grid and thus must be called after initializing MPI.
*     Currently it is called in init_operators() if optr->use_qudainverter
*     flag is set.
*     Memory for the QUDA gaugefield on the host is allocated but not filled
*     yet (the latter is done in _loadGaugeQuda(), see below).
*     Performance critical settings are done here and can be changed.
*
*   void _endQuda()
*     Finalizes the QUDA library. Call before MPI_Finalize().
*
*   void _loadGaugeQuda()
*     Copies and reorders the gaugefield on the host and copies it to the GPU.
*     Must be called between last changes on the gaugefield (smearing etc.)
*     and first call of the inverter. In particular, 'boundary(const double kappa)'
*     must be called before if nontrivial boundary conditions are to be used since
*     those will be applied directly to the gaugefield. Currently it is called just
*     before the inversion is done (might result in wasted loads...).
*
*   void _setMultigridParam()
*     borrowed from QUDA multigrid_invert_test
*
*   The functions
*
*     int invert_eo_quda(...);
*     int invert_doublet_eo_quda(...);
*     void M_full_quda(...);
*     void D_psi_quda(...);
*
*   mimic their tmLQCD counterparts in functionality as well as input and
*   output parameters. The invert functions will check the parameters
*   g_mu, g_c_sw do decide which QUDA operator to create.
*
*   To activate those, set "UseQudaInverter = yes" in the operator
*   declaration of the input file. For details see the documentation.
*
*   The function
*
*     int invert_quda_direct(...);
*
*   provides a direct interface to the QUDA solver and is not accessible through
*   the input file.
*
* Notes:
*
* Minimum QUDA version is 0.7.0 (see https://github.com/lattice/quda/issues/151 
* and https://github.com/lattice/quda/issues/157).
*
*
**************************************************************************/

#include "quda.h"
#include "quda_interface.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "boundary.h"
#include "linalg/convert_eo_to_lexic.h"
#include "linalg/mul_r.h"
#include "solver/solver.h"
#include "solver/solver_field.h"
#include "gettime.h"
#include "boundary.h"
#include "global.h"
#include "operator.h"

double X0, X1, X2, X3;

// define order of the spatial indices
// default is LX-LY-LZ-T, see below def. of local lattice size, this is related to
// the gamma basis transformation from tmLQCD -> UKQCD
// for details see https://github.com/lattice/quda/issues/157
#define USE_LZ_LY_LX_T 0

// TRIVIAL_BC are trivial (anti-)periodic boundary conditions,
// i.e. 1 or -1 on last timeslice
// tmLQCD uses twisted BC, i.e. phases on all timeslices.
// if using TRIVIAL_BC: can't compare inversion result to tmLQCD
// if not using TRIVIAL_BC: BC will be applied to gauge field,
// can't use 12 parameter reconstruction
#define TRIVIAL_BC 0

#define MAX(a,b) ((a)>(b)?(a):(b))

// gauge and invert paramameter structs; init. in _initQuda()
QudaGaugeParam  gauge_param;
QudaInvertParam inv_param;

QudaVerbosity get_verbosity_type(char* s)
{
  QudaVerbosity ret =  QUDA_INVALID_VERBOSITY;

  if (strcmp(s, "silent") == 0){
    ret = QUDA_SILENT;
  }else if (strcmp(s, "summarize") == 0){
    ret = QUDA_SUMMARIZE;
  }else if (strcmp(s, "verbose") == 0){
    ret = QUDA_VERBOSE;
  }else if (strcmp(s, "debug") == 0){
    ret = QUDA_DEBUG_VERBOSE;
  }else{
    fprintf(stderr, "Error: invalid verbosity type %s\n", s);
    exit(1);
  }
  return ret;
}

QudaInverterType get_solver_type(char* s)
{
  QudaInverterType ret =  QUDA_INVALID_INVERTER;

  if (strcmp(s, "cg") == 0){
    ret = QUDA_CG_INVERTER;
  } else if (strcmp(s, "bicgstab") == 0){
    ret = QUDA_BICGSTAB_INVERTER;
  } else if (strcmp(s, "gcr") == 0){
    ret = QUDA_GCR_INVERTER;
  } else if (strcmp(s, "pcg") == 0){
    ret = QUDA_PCG_INVERTER;
  } else if (strcmp(s, "mpcg") == 0){
    ret = QUDA_MPCG_INVERTER;
  } else if (strcmp(s, "mpbicgstab") == 0){
    ret = QUDA_MPBICGSTAB_INVERTER;
  } else if (strcmp(s, "mr") == 0){
    ret = QUDA_MR_INVERTER;
  } else if (strcmp(s, "sd") == 0){
    ret = QUDA_SD_INVERTER;
  } else if (strcmp(s, "eigcg") == 0){
    ret = QUDA_EIGCG_INVERTER;
  } else if (strcmp(s, "inc-eigcg") == 0){
    ret = QUDA_INC_EIGCG_INVERTER;
  } else if (strcmp(s, "gmresdr") == 0){
    ret = QUDA_GMRESDR_INVERTER;
  } else if (strcmp(s, "gmresdr-proj") == 0){
    ret = QUDA_GMRESDR_PROJ_INVERTER;
  } else if (strcmp(s, "gmresdr-sh") == 0){
    ret = QUDA_GMRESDR_SH_INVERTER;
  } else if (strcmp(s, "fgmresdr") == 0){
    ret = QUDA_FGMRESDR_INVERTER;
  } else if (strcmp(s, "mg") == 0){
    ret = QUDA_MG_INVERTER;
  } else if (strcmp(s, "bicgstab-l") == 0){
    ret = QUDA_BICGSTABL_INVERTER;
  } else {
    fprintf(stderr, "Error: invalid solver type\n");
    exit(1);
  }

  return ret;
}


// FIXME: params to pass to MG
QudaInvertParam inv_mg_param;
Quda_mg_params_t Quda_multigrid_input;
Quda_invert_params_t Quda_invert_input;
// pointer to the QUDA gaugefield
double *gauge_quda[4];

// pointer to a temp. spinor, used for reordering etc.
double *tempSpinor;
  
// function that maps coordinates in the communication grid to MPI ranks
int commsMap(const int *coords, void *fdata) {
#if USE_LZ_LY_LX_T
  int n[4] = {coords[3], coords[2], coords[1], coords[0]};
#else
  int n[4] = {coords[3], coords[0], coords[1], coords[2]};
#endif

  int rank = 0;
#ifdef TM_USE_MPI
  MPI_Cart_rank( g_cart_grid, n, &rank );
#endif

  return rank;
}

// variable to check if quda has been initialized
static int quda_initialized = 0;

void _setMultigridParam(QudaMultigridParam* mg_param);

void _initQuda() {
  if( quda_initialized )
    return;

  if( g_debug_level > 0 )
    if(g_proc_id == 0)
      printf("\n# QUDA: Detected QUDA version %d.%d.%d\n\n", QUDA_VERSION_MAJOR, QUDA_VERSION_MINOR, QUDA_VERSION_SUBMINOR);
  if( QUDA_VERSION_MAJOR == 0 && QUDA_VERSION_MINOR < 7) {
    fprintf(stderr, "Error: minimum QUDA version required is 0.7.0 (for support of chiral basis and removal of bug in mass normalization with preconditioning).\n");
    exit(-2);
  }

  gauge_param = newQudaGaugeParam();
  inv_param = newQudaInvertParam();
  inv_mg_param = newQudaInvertParam();

  // *** QUDA parameters begin here (sloppy prec. will be adjusted in invert)
  QudaPrecision cpu_prec  = QUDA_DOUBLE_PRECISION;
  QudaPrecision cuda_prec = QUDA_DOUBLE_PRECISION;
  QudaPrecision cuda_prec_sloppy = QUDA_SINGLE_PRECISION;
  QudaPrecision cuda_prec_precondition = QUDA_SINGLE_PRECISION;

  QudaTune tune = QUDA_TUNE_YES;

  // *** the remainder should not be changed for this application
  // local lattice size
#if USE_LZ_LY_LX_T
  gauge_param.X[0] = LZ;
  gauge_param.X[1] = LY;
  gauge_param.X[2] = LX;
  gauge_param.X[3] = T;
#else
  gauge_param.X[0] = LX;
  gauge_param.X[1] = LY;
  gauge_param.X[2] = LZ;
  gauge_param.X[3] = T;
#endif

  inv_param.Ls = 1;

  gauge_param.anisotropy = 1.0;
  gauge_param.type = QUDA_WILSON_LINKS;
  gauge_param.gauge_order = QUDA_QDP_GAUGE_ORDER;

  gauge_param.cpu_prec = cpu_prec;
  gauge_param.cuda_prec = cuda_prec;
  gauge_param.reconstruct = 18;
  gauge_param.cuda_prec_sloppy = cuda_prec_sloppy;
  gauge_param.reconstruct_sloppy = 18;
  gauge_param.cuda_prec_precondition = cuda_prec_precondition;
  gauge_param.reconstruct_precondition = 18;
  gauge_param.gauge_fix = QUDA_GAUGE_FIXED_NO;
  

  inv_param.dagger = QUDA_DAG_NO;
  inv_param.mass_normalization = QUDA_KAPPA_NORMALIZATION;
  inv_param.solver_normalization = QUDA_DEFAULT_NORMALIZATION;

  inv_param.pipeline = 0;
  inv_param.gcrNkrylov = 10;

  // require both L2 relative and heavy quark residual to determine convergence
//  inv_param.residual_type = (QudaResidualType)(QUDA_L2_RELATIVE_RESIDUAL | QUDA_HEAVY_QUARK_RESIDUAL);
  inv_param.tol_hq = 1.0;//1e-3; // specify a tolerance for the residual for heavy quark residual
  inv_param.reliable_delta = 1e-3; // ignored by multi-shift solver
  inv_param.use_sloppy_partial_accumulator = 0;

  // domain decomposition preconditioner parameters
  inv_param.inv_type_precondition = QUDA_CG_INVERTER;
  inv_param.schwarz_type = QUDA_ADDITIVE_SCHWARZ;
  inv_param.precondition_cycle = 1;
  inv_param.tol_precondition = 1e-1;
  inv_param.maxiter_precondition = 10;
  inv_param.verbosity_precondition = QUDA_SILENT;
  if( g_debug_level >= 5 )
    inv_param.verbosity_precondition = QUDA_VERBOSE;

  inv_param.cuda_prec_precondition = cuda_prec_precondition;
  inv_param.omega = 1.0;

  inv_param.cpu_prec = cpu_prec;
  inv_param.cuda_prec = cuda_prec;
  inv_param.cuda_prec_sloppy = cuda_prec_sloppy;

  inv_param.clover_cpu_prec = cpu_prec;
  inv_param.clover_cuda_prec = cuda_prec;
  inv_param.clover_cuda_prec_sloppy = cuda_prec_sloppy;
  inv_param.clover_cuda_prec_precondition = cuda_prec_precondition;

  inv_param.preserve_source = QUDA_PRESERVE_SOURCE_YES;
  inv_param.gamma_basis = QUDA_CHIRAL_GAMMA_BASIS; // CHIRAL -> UKQCD does not seem to be supported right now...
  inv_param.dirac_order = QUDA_DIRAC_ORDER;

  inv_param.input_location = QUDA_CPU_FIELD_LOCATION;
  inv_param.output_location = QUDA_CPU_FIELD_LOCATION;

  inv_param.tune = tune ? QUDA_TUNE_YES : QUDA_TUNE_NO;

  gauge_param.ga_pad = 0; // 24*24*24/2;
  inv_param.sp_pad = 0; // 24*24*24/2;
  inv_param.cl_pad = 0; // 24*24*24/2;

  // For multi-GPU, ga_pad must be large enough to store a time-slice
  int x_face_size = gauge_param.X[1]*gauge_param.X[2]*gauge_param.X[3]/2;
  int y_face_size = gauge_param.X[0]*gauge_param.X[2]*gauge_param.X[3]/2;
  int z_face_size = gauge_param.X[0]*gauge_param.X[1]*gauge_param.X[3]/2;
  int t_face_size = gauge_param.X[0]*gauge_param.X[1]*gauge_param.X[2]/2;
  int pad_size =MAX(x_face_size, y_face_size);
  pad_size = MAX(pad_size, z_face_size);
  pad_size = MAX(pad_size, t_face_size);
  gauge_param.ga_pad = pad_size;

  // solver verbosity
  if( g_debug_level == 0 )
    inv_param.verbosity = QUDA_SILENT;
  else if( g_debug_level >= 1 && g_debug_level < 3 )
    inv_param.verbosity = QUDA_SUMMARIZE;
  else if( g_debug_level >= 3 && g_debug_level < 5 )
    inv_param.verbosity = QUDA_VERBOSE;
  else if( g_debug_level >= 5 )
    inv_param.verbosity = QUDA_DEBUG_VERBOSE;

  // general verbosity
  setVerbosityQuda( QUDA_SUMMARIZE, "# QUDA: ", stdout);

  // declare the grid mapping used for communications in a multi-GPU grid
#if USE_LZ_LY_LX_T
  int grid[4] = {g_nproc_z, g_nproc_y, g_nproc_x, g_nproc_t};
#else
  int grid[4] = {g_nproc_x, g_nproc_y, g_nproc_z, g_nproc_t};
#endif

  initCommsGridQuda(4, grid, commsMap, NULL);

  // alloc gauge_quda
  size_t gSize = (gauge_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);

  for (int dir = 0; dir < 4; dir++) {
    gauge_quda[dir] = (double*) malloc(VOLUME*18*gSize);
    if(gauge_quda[dir] == NULL) {
      fprintf(stderr, "_initQuda: malloc for gauge_quda[dir] failed");
      exit(-2);
    }
  }

  // alloc space for a temp. spinor, used throughout this module
  tempSpinor  = (double*)malloc( 2*VOLUME*24*sizeof(double) ); /* factor 2 for doublet */
  if(tempSpinor == NULL) {
    fprintf(stderr, "_initQuda: malloc for tempSpinor failed");
    exit(-2);
  }

  // initialize the QUDA library
#ifdef TM_USE_MPI
  initQuda(-1); //sets device numbers automatically
#else
  // when running in 'subprocess' mode, the external program should have provided us with a unique
  // id in the range 0 to (N-1), where N is the number of NVIDIA devices available (see wrapper/lib_wrapper.c)
  if(subprocess_flag){
    initQuda(g_external_id);
  }else{
    initQuda(0);  //scalar build without subprocess: use device 0
  }
#endif
  quda_initialized = 1;
}

// finalize the QUDA library
void _endQuda() {
  if( quda_initialized ) {
    freeGaugeQuda();
    freeCloverQuda(); // this is safe even if there is no Clover field loaded, at least it was in QUDA v0.7.2
    free((void*)tempSpinor);
    endQuda();
  }
}


void _loadGaugeQuda( const int compression ) {
  if( inv_param.verbosity > QUDA_SILENT )
    if(g_proc_id == 0)
      printf("# QUDA: Called _loadGaugeQuda\n");

  _Complex double tmpcplx;

  size_t gSize = (gauge_param.cpu_prec == QUDA_DOUBLE_PRECISION) ? sizeof(double) : sizeof(float);
  
  // now copy and reorder
  for( int x0=0; x0<T; x0++ )
    for( int x1=0; x1<LX; x1++ )
      for( int x2=0; x2<LY; x2++ )
        for( int x3=0; x3<LZ; x3++ ) {
#if USE_LZ_LY_LX_T
          int j = x3 + LZ*x2 + LY*LZ*x1 + LX*LY*LZ*x0;
          int tm_idx = x1 + LX*x2 + LY*LX*x3 + LZ*LY*LX*x0;
#else
          int j = x1 + LX*x2 + LY*LX*x3 + LZ*LY*LX*x0;
          int tm_idx = x3 + LZ*x2 + LY*LZ*x1 + LX*LY*LZ*x0;
#endif
          int oddBit = (x0+x1+x2+x3) & 1;
          int quda_idx = 18*(oddBit*VOLUME/2+j/2);

#if USE_LZ_LY_LX_T
          memcpy( &(gauge_quda[0][quda_idx]), &(g_gauge_field[tm_idx][3]), 18*gSize);
          memcpy( &(gauge_quda[1][quda_idx]), &(g_gauge_field[tm_idx][2]), 18*gSize);
          memcpy( &(gauge_quda[2][quda_idx]), &(g_gauge_field[tm_idx][1]), 18*gSize);
          memcpy( &(gauge_quda[3][quda_idx]), &(g_gauge_field[tm_idx][0]), 18*gSize);
#else
          memcpy( &(gauge_quda[0][quda_idx]), &(g_gauge_field[tm_idx][1]), 18*gSize);
          memcpy( &(gauge_quda[1][quda_idx]), &(g_gauge_field[tm_idx][2]), 18*gSize);
          memcpy( &(gauge_quda[2][quda_idx]), &(g_gauge_field[tm_idx][3]), 18*gSize);
          memcpy( &(gauge_quda[3][quda_idx]), &(g_gauge_field[tm_idx][0]), 18*gSize);

        if( compression == 18 ) {
          // apply boundary conditions
          for( int i=0; i<9; i++ ) {
            tmpcplx = gauge_quda[0][quda_idx+2*i] + I*gauge_quda[0][quda_idx+2*i+1];
            tmpcplx *= -phase_1/g_kappa;
            gauge_quda[0][quda_idx+2*i]   = creal(tmpcplx);
            gauge_quda[0][quda_idx+2*i+1] = cimag(tmpcplx);

            tmpcplx = gauge_quda[1][quda_idx+2*i] + I*gauge_quda[1][quda_idx+2*i+1];
            tmpcplx *= -phase_2/g_kappa;
            gauge_quda[1][quda_idx+2*i]   = creal(tmpcplx);
            gauge_quda[1][quda_idx+2*i+1] = cimag(tmpcplx);

            tmpcplx = gauge_quda[2][quda_idx+2*i] + I*gauge_quda[2][quda_idx+2*i+1];
            tmpcplx *= -phase_3/g_kappa;
            gauge_quda[2][quda_idx+2*i]   = creal(tmpcplx);
            gauge_quda[2][quda_idx+2*i+1] = cimag(tmpcplx);

            tmpcplx = gauge_quda[3][quda_idx+2*i] + I*gauge_quda[3][quda_idx+2*i+1];
            tmpcplx *= -phase_0/g_kappa;
            gauge_quda[3][quda_idx+2*i]   = creal(tmpcplx);
            gauge_quda[3][quda_idx+2*i+1] = cimag(tmpcplx);
          }
        }

#endif
        }

  loadGaugeQuda((void*)gauge_quda, &gauge_param);
}


// reorder spinor to QUDA format
void reorder_spinor_toQuda( double* sp, QudaPrecision precision, int doublet, double* sp2 ) {
  double startTime = gettime();

  if( doublet ) {
    memcpy( tempSpinor,           sp,  VOLUME*24*sizeof(double) );
    memcpy( tempSpinor+VOLUME*24, sp2, VOLUME*24*sizeof(double) );
  }
  else {
    memcpy( tempSpinor, sp, VOLUME*24*sizeof(double) );
  }

  // now copy and reorder from tempSpinor to spinor
  for( int x0=0; x0<T; x0++ )
    for( int x1=0; x1<LX; x1++ )
      for( int x2=0; x2<LY; x2++ )
        for( int x3=0; x3<LZ; x3++ ) {
#if USE_LZ_LY_LX_T
          int j = x3 + LZ*x2 + LY*LZ*x1 + LX*LY*LZ*x0;
          int tm_idx = x1 + LX*x2 + LY*LX*x3 + LZ*LY*LX*x0;
#else
          int j = x1 + LX*x2 + LY*LX*x3 + LZ*LY*LX*x0;
          int tm_idx   = x3 + LZ*x2 + LY*LZ*x1 + LX*LY*LZ*x0;
#endif
          int oddBit = (x0+x1+x2+x3) & 1;

          if( doublet ) {
            memcpy( &(sp[24*(oddBit*VOLUME+j/2)]),          &(tempSpinor[24* tm_idx        ]), 24*sizeof(double));
            memcpy( &(sp2[24*(oddBit*VOLUME+j/2+VOLUME/2)]), &(tempSpinor[24*(tm_idx+VOLUME)]), 24*sizeof(double)); // FIXME BK: I don't think this is correct!
          }
          else {
            memcpy( &(sp[24*(oddBit*VOLUME/2+j/2)]), &(tempSpinor[24*tm_idx]), 24*sizeof(double));
          }

        }

  double endTime = gettime();
  double diffTime = endTime - startTime;
  if(g_proc_id == 0)
    printf("# QUDA: time spent in reorder_spinor_toQuda: %f secs\n", diffTime);
}

// reorder spinor from QUDA format
void reorder_spinor_fromQuda( double* sp, QudaPrecision precision, int doublet, double* sp2 ) {
  double startTime = gettime();

  if( doublet ) {
    memcpy( tempSpinor, sp, 2*VOLUME*24*sizeof(double) ); // FIXME BK: I think this is wrong (why is there sp2?)
  }
  else {
    memcpy( tempSpinor, sp, VOLUME*24*sizeof(double) );
  }

  // now copy and reorder from tempSpinor to spinor
  for( int x0=0; x0<T; x0++ )
    for( int x1=0; x1<LX; x1++ )
      for( int x2=0; x2<LY; x2++ )
        for( int x3=0; x3<LZ; x3++ ) {
#if USE_LZ_LY_LX_T
          int j = x3 + LZ*x2 + LY*LZ*x1 + LX*LY*LZ*x0;
          int tm_idx = x1 + LX*x2 + LY*LX*x3 + LZ*LY*LX*x0;
#else
          int j = x1 + LX*x2 + LY*LX*x3 + LZ*LY*LX*x0;
          int tm_idx   = x3 + LZ*x2 + LY*LZ*x1 + LX*LY*LZ*x0;
#endif
          int oddBit = (x0+x1+x2+x3) & 1;

          if( doublet ) {
            memcpy( &(sp[24* tm_idx]),  &(tempSpinor[24*(oddBit*VOLUME+j/2)         ]), 24*sizeof(double));
            memcpy( &(sp2[24*(tm_idx)]), &(tempSpinor[24*(oddBit*VOLUME+j/2+VOLUME/2)]), 24*sizeof(double));
          }
          else {
            memcpy( &(sp[24*tm_idx]), &(tempSpinor[24*(oddBit*VOLUME/2+j/2)]), 24*sizeof(double));
          }
        }

  double endTime = gettime();
  double diffTime = endTime - startTime;
  if(g_proc_id == 0)
    printf("# QUDA: time spent in reorder_spinor_fromQuda: %f secs\n", diffTime);
}

void set_boundary_conditions( CompressionType* compression ) {
  // we can't have compression and theta-BC
  if( fabs(X1)>0.0 || fabs(X2)>0.0 || fabs(X3)>0.0 || (fabs(X0)!=0.0 && fabs(X0)!=1.0) ) {
    if( *compression!=NO_COMPRESSION ) {
      if(g_proc_id == 0) {
          printf("\n# QUDA: WARNING you can't use compression %d with boundary conditions for fermion fields (t,x,y,z)*pi: (%f,%f,%f,%f) \n", *compression,X0,X1,X2,X3);
          printf("# QUDA: disabling compression.\n\n");
          *compression=NO_COMPRESSION;
      }
    }
  }

  QudaReconstructType link_recon;
  QudaReconstructType link_recon_sloppy;

  if( *compression==NO_COMPRESSION ) { // theta BC
    gauge_param.t_boundary = QUDA_PERIODIC_T; // BC will be applied to gaugefield
    link_recon = 18;
    link_recon_sloppy = 18;
  }
  else { // trivial BC
    gauge_param.t_boundary = ( fabs(X0)>0.0 ? QUDA_ANTI_PERIODIC_T : QUDA_PERIODIC_T );
    link_recon = 12;
    link_recon_sloppy = *compression;
    if( g_debug_level > 0 )
      if(g_proc_id == 0)
        printf("\n# QUDA: WARNING using %d compression with trivial (A)PBC instead of theta-BC ((t,x,y,z)*pi: (%f,%f,%f,%f))! This works fine but the residual check on the host (CPU) will fail.\n",*compression,X0,X1,X2,X3);
  }

  gauge_param.reconstruct = link_recon;
  gauge_param.reconstruct_sloppy = link_recon_sloppy;
  gauge_param.reconstruct_precondition = link_recon_sloppy;
}

void set_sloppy_prec( const SloppyPrecision sloppy_precision ) {

  // choose sloppy prec.
  QudaPrecision cuda_prec_sloppy;
  if( sloppy_precision==SLOPPY_DOUBLE ) {
    cuda_prec_sloppy = QUDA_DOUBLE_PRECISION;
    if(g_proc_id == 0) printf("# QUDA: Using double prec. as sloppy!\n");
  }
  else if( sloppy_precision==SLOPPY_HALF ) {
    cuda_prec_sloppy = QUDA_HALF_PRECISION;
    if(g_proc_id == 0) printf("# QUDA: Using half prec. as sloppy!\n");
  }
  else {
    cuda_prec_sloppy = QUDA_SINGLE_PRECISION;
    if(g_proc_id == 0) printf("# QUDA: Using single prec. as sloppy!\n");
  }
  gauge_param.cuda_prec_sloppy = cuda_prec_sloppy;
  inv_param.cuda_prec_sloppy = cuda_prec_sloppy;
  inv_param.clover_cuda_prec_sloppy = cuda_prec_sloppy;
}

int invert_quda_direct(double * const propagator, double * const source,
                const int op_id, const int gauge_persist) {

  double atime, atotaltime = gettime();
  void *spinorIn  = (void*)source; // source
  void *spinorOut = (void*)propagator; // solution
  static int loadGauge = 1;
  
  operator * optr = &operator_list[op_id];
  // g_kappa is necessary for the gauge field to be correctly translated from tmLQCD to QUDA
  g_kappa = optr->kappa;
  g_c_sw = optr->c_sw;
  g_mu = optr->mu;

  boundary(optr->kappa);
  
  if ( g_relative_precision_flag )
    inv_param.residual_type = QUDA_L2_RELATIVE_RESIDUAL;
  else
    inv_param.residual_type = QUDA_L2_ABSOLUTE_RESIDUAL;
  
  inv_param.kappa = optr->kappa;

  // figure out which BC to use (theta, trivial...)
  set_boundary_conditions(&optr->compression_type);

  // set the sloppy precision of the mixed prec solver
  set_sloppy_prec(optr->sloppy_precision);

  // choose dslash type
  if( optr->mu != 0.0 && optr->c_sw > 0.0 ) {
    inv_param.twist_flavor = QUDA_TWIST_SINGLET;
    inv_param.dslash_type = QUDA_TWISTED_CLOVER_DSLASH;
    inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN;
    inv_param.solution_type = QUDA_MAT_SOLUTION;
    inv_param.clover_order = QUDA_PACKED_CLOVER_ORDER;
    // IMPORTANT: use opposite TM flavor since gamma5 -> -gamma5 (until LXLYLZT prob. resolved)
    inv_param.mu = -optr->mu/2./optr->kappa;
    inv_param.clover_coeff = optr->c_sw*optr->kappa;
    inv_param.compute_clover_inverse = 1;
    inv_param.compute_clover = 1;
  }
  else if( optr->mu != 0.0 ) {
    inv_param.twist_flavor = QUDA_TWIST_SINGLET;
    inv_param.dslash_type = QUDA_TWISTED_MASS_DSLASH;
    inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN_ASYMMETRIC;
    inv_param.solution_type = QUDA_MAT_SOLUTION;
    // IMPORTANT: use opposite TM flavor since gamma5 -> -gamma5 (until LXLYLZT prob. resolved)
    inv_param.mu = optr->mu/2./optr->kappa;
  }
  else if( optr->c_sw > 0.0 ) {
    inv_param.twist_flavor = QUDA_TWIST_NO;
    inv_param.dslash_type = QUDA_CLOVER_WILSON_DSLASH;
    inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN;
    inv_param.solution_type = QUDA_MAT_SOLUTION;
    inv_param.clover_order = QUDA_PACKED_CLOVER_ORDER;
    inv_param.clover_coeff = optr->c_sw*optr->kappa;
    inv_param.compute_clover_inverse = 1;
    inv_param.compute_clover = 1;
  }
  else {
    inv_param.twist_flavor = QUDA_TWIST_NO;
    inv_param.dslash_type = QUDA_WILSON_DSLASH;
    inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN;
    inv_param.solution_type = QUDA_MAT_SOLUTION;
  }

  // choose solver
  if(optr->solver == BICGSTAB) {
    if(g_proc_id == 0) {printf("# QUDA: Using BiCGstab!\n"); fflush(stdout);}
    inv_param.inv_type = QUDA_BICGSTAB_INVERTER;
  }
  else {
    /* Here we invert the hermitean operator squared */
    inv_param.inv_type = QUDA_CG_INVERTER;
    if(g_proc_id == 0) {
      printf("# QUDA: Using mixed precision CG!\n");
      printf("# QUDA: mu = %.12f, kappa = %.12f\n", optr->mu/2./optr->kappa, optr->kappa);
      fflush(stdout);
    }
  }

  // direct or norm-op. solve
  if( inv_param.inv_type == QUDA_CG_INVERTER ) {
    if( optr->even_odd_flag ) {
      inv_param.solve_type = QUDA_NORMERR_PC_SOLVE;
      if(g_proc_id == 0) printf("# QUDA: Using preconditioning!\n");
    }
    else {
      inv_param.solve_type = QUDA_NORMERR_SOLVE;
      if(g_proc_id == 0) printf("# QUDA: Not using preconditioning!\n");
    }
  }
  else {
    if( optr->even_odd_flag ) {
      inv_param.solve_type = QUDA_DIRECT_PC_SOLVE;
      if(g_proc_id == 0) printf("# QUDA: Using preconditioning!\n");
    }
    else {
      inv_param.solve_type = QUDA_DIRECT_SOLVE;
      if(g_proc_id == 0) printf("# QUDA: Not using preconditioning!\n");
    }
  }

  inv_param.tol = sqrt(optr->eps_sq);
  inv_param.maxiter = optr->maxiter;

  inv_param.Ls = 1;
  
  // load gauge after setting precision
  if(loadGauge == 1){
    atime = gettime();
    _loadGaugeQuda(optr->compression_type);
    if(g_proc_id==0 && g_debug_level > 0 ) printf("# QUDA: Time for loadGaugeQuda: %.4e\n",gettime()-atime);
    // NULL pointers to force construction of the clover fields
    if( optr->c_sw > 0.0 ) {
      atime = gettime();
      loadCloverQuda(NULL, NULL, &inv_param);
      if(g_proc_id==0 && g_debug_level > 0 ) printf("# QUDA: Time for loadCloverQuda: %.4e\n",gettime()-atime);
    }
  }
  
  if(gauge_persist == 1)
    loadGauge = 0;
  
  // reorder spinor
  reorder_spinor_toQuda( (double*)spinorIn, inv_param.cpu_prec, 0, NULL );

  // perform the inversion
  invertQuda(spinorOut, spinorIn, &inv_param);

  if( inv_param.verbosity == QUDA_VERBOSE )
    if(g_proc_id == 0)
      printf("# QUDA: Device memory used:  Spinor: %f GiB,  Gauge: %f GiB, Clover: %f GiB\n",
             inv_param.spinorGiB, gauge_param.gaugeGiB, inv_param.cloverGiB);
  if( inv_param.verbosity > QUDA_SILENT )
    if(g_proc_id == 0)
      printf("# QUDA: Done: %i iter / %g secs = %g Gflops\n",
             inv_param.iter, inv_param.secs, inv_param.gflops/inv_param.secs);

  // number of CG iterations
  optr->iterations = inv_param.iter;

  // reorder spinor
  reorder_spinor_fromQuda( (double*)spinorIn,  inv_param.cpu_prec, 0, NULL );
  reorder_spinor_fromQuda( (double*)spinorOut, inv_param.cpu_prec, 0, NULL );
  // propagator in usual normalisation, this is only necessary in invert_quda_direct
  mul_r((spinor*)spinorOut, (2*optr->kappa), (spinor*)spinorOut, VOLUME );

  // when gauge_persist == 1, we do not free the gauge so that it's more efficient!
  // and we also don't load it on subsequent calls
  if(gauge_persist != 1) {
    freeGaugeQuda();
    freeCloverQuda(); // this is safe even if there is no Clover field loaded, at least it was in QUDA v0.7.2
  }

  if( g_proc_id==0 && g_debug_level > 0 )
    printf("# QUDA: Total time for invert_quda_direct: %.4e\n",gettime()-atotaltime); 

  if(optr->iterations >= optr->maxiter)
    return(-1);

  return(optr->iterations);
}

int invert_eo_quda(spinor * const Even_new, spinor * const Odd_new,
                   spinor * const Even, spinor * const Odd,
                   const double precision, const int max_iter,
                   const int solver_flag, const int rel_prec,
                   const int even_odd_flag, solver_params_t solver_params,
                   SloppyPrecision sloppy_precision,
                   CompressionType compression) {

  spinor ** solver_field = NULL;
  const int nr_sf = 2;
  init_solver_field(&solver_field, VOLUMEPLUSRAND, nr_sf);

  convert_eo_to_lexic(solver_field[0],  Even, Odd);

//  convert_eo_to_lexic(solver_field[1], Even_new, Odd_new);

  void *spinorIn  = (void*)solver_field[0]; // source
  void *spinorOut = (void*)solver_field[1]; // solution

  if ( rel_prec )
    inv_param.residual_type = QUDA_L2_RELATIVE_RESIDUAL;
  else
    inv_param.residual_type = QUDA_L2_ABSOLUTE_RESIDUAL;

  inv_param.kappa = g_kappa;
  //inv_param.mass = 1/(2*g_kappa)-4.0;

  // figure out which BC to use (theta, trivial...)
  set_boundary_conditions(&compression);

  // set the sloppy precision of the mixed prec solver
  set_sloppy_prec(sloppy_precision);

  // load gauge after setting precision
  _loadGaugeQuda(compression);

  // choose dslash type
  if( g_mu != 0.0 && g_c_sw > 0.0 ) {
    inv_param.twist_flavor = QUDA_TWIST_SINGLET;
    inv_param.dslash_type = QUDA_TWISTED_CLOVER_DSLASH;
    inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN;
    inv_param.solution_type = QUDA_MAT_SOLUTION;
    inv_param.clover_order = QUDA_PACKED_CLOVER_ORDER;
    // IMPORTANT: use opposite TM flavor since gamma5 -> -gamma5 (until LXLYLZT prob. resolved)
    inv_param.mu = -g_mu/2./g_kappa;
    inv_param.clover_coeff = g_c_sw*g_kappa;
    inv_param.compute_clover_inverse = 1;
    inv_param.compute_clover = 1;
  }
  else if( g_mu != 0.0 ) {
    inv_param.twist_flavor = QUDA_TWIST_SINGLET;
    inv_param.dslash_type = QUDA_TWISTED_MASS_DSLASH;
    inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN_ASYMMETRIC;
    inv_param.solution_type = QUDA_MAT_SOLUTION;
    // IMPORTANT: use opposite TM flavor since gamma5 -> -gamma5 (until LXLYLZT prob. resolved)
    inv_param.mu = -g_mu/2./g_kappa;
  }
  else if( g_c_sw > 0.0 ) {
    inv_param.twist_flavor = QUDA_TWIST_NO;
    inv_param.dslash_type = QUDA_CLOVER_WILSON_DSLASH;
    inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN;
    inv_param.solution_type = QUDA_MAT_SOLUTION;
    inv_param.clover_order = QUDA_PACKED_CLOVER_ORDER;
    inv_param.clover_coeff = g_c_sw*g_kappa;
    inv_param.compute_clover_inverse = 1;
    inv_param.compute_clover = 1;
  }
  else {
    inv_param.twist_flavor = QUDA_TWIST_NO;
    inv_param.dslash_type = QUDA_WILSON_DSLASH;
    inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN;
    inv_param.solution_type = QUDA_MAT_SOLUTION;
  }

  // choose solver
  if(solver_flag == BICGSTAB) {
    if(g_proc_id == 0) {printf("# QUDA: Using BiCGstab!\n"); fflush(stdout);}
    inv_param.inv_type = QUDA_BICGSTAB_INVERTER;
  }
  else {
    /* Here we invert the hermitean operator squared */
    inv_param.inv_type = QUDA_CG_INVERTER;
    if(g_proc_id == 0) {
      printf("# QUDA: Using mixed precision CG!\n");
      printf("# QUDA: mu = %.12f, kappa = %.12f\n", g_mu/2./g_kappa, g_kappa);
      fflush(stdout);
    }
  }

  // direct or norm-op. solve
  if( inv_param.inv_type == QUDA_CG_INVERTER ) {
    if( even_odd_flag ) {
      inv_param.solve_type = QUDA_NORMERR_PC_SOLVE;
      if(g_proc_id == 0) printf("# QUDA: Using preconditioning!\n");
    }
    else {
      inv_param.solve_type = QUDA_NORMERR_SOLVE;
      if(g_proc_id == 0) printf("# QUDA: Not using preconditioning!\n");
    }
  }
  else {
    if( even_odd_flag ) {
      inv_param.solve_type = QUDA_DIRECT_PC_SOLVE;
      if(g_proc_id == 0) printf("# QUDA: Using preconditioning!\n");
    }
    else {
      inv_param.solve_type = QUDA_DIRECT_SOLVE;
      if(g_proc_id == 0) printf("# QUDA: Not using preconditioning!\n");
    }
  }

  inv_param.tol = sqrt(precision);
  inv_param.maxiter = max_iter;

  inv_param.Ls = 1;

  // NULL pointers to host fields to force
  // construction instead of download of the clover field:
  if( g_c_sw > 0.0 )
    loadCloverQuda(NULL, NULL, &inv_param);

  // reorder spinor
  reorder_spinor_toQuda( (double*)spinorIn, inv_param.cpu_prec, 0, NULL );

  // FIXME should be set in input file and compatibility between various params should be checked
  int use_multigrid_quda = 1;
  void* mg_preconditioner = NULL;
  QudaMultigridParam mg_param = newQudaMultigridParam();
  if( use_multigrid_quda && inv_param.inv_type != QUDA_CG_INVERTER ){
    // FIXME explicitly select compatible params for inv_param
    // probably need two separate sets of params, one for the setup and one for the target solution
    // coarsening does not support QUDA_MATPC_EVEN_EVEN_ASYMMETRIC
    if( inv_param.matpc_type == QUDA_MATPC_EVEN_EVEN_ASYMMETRIC ) inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN;
    inv_param.inv_type = QUDA_GCR_INVERTER;
    inv_param.gcrNkrylov = 20;
    inv_param.inv_type_precondition = QUDA_MG_INVERTER;
    inv_param.schwarz_type = QUDA_ADDITIVE_SCHWARZ;
    inv_param.reliable_delta = 1e-4;
    inv_param.precondition_cycle = 1;
    inv_param.tol_precondition = 1e-1;
    inv_param.maxiter_precondition = 1;
    inv_param.omega = 1.0;
    
    // FIXME select appropriate MG params here
    inv_mg_param = inv_param;
    mg_param.invert_param = &inv_mg_param;
    _setMultigridParam(&mg_param);

    if(g_proc_id==0){
      printf("----------------------------------------\n");
      printQudaInvertParam(&inv_param);
      printf("----------------------------------------\n");
      printQudaInvertParam(&inv_mg_param);
      printf("----------------------------------------\n");
      printQudaMultigridParam(&mg_param);
      printf("----------------------------------------\n");
    }


    if(g_proc_id==0){ printf("calling mg preconditioner\n"); fflush(stdout); }
    mg_preconditioner = newMultigridQuda(&mg_param);
    inv_param.preconditioner = mg_preconditioner;
  }

  if(g_proc_id==0 && inv_param.inv_type_precondition == QUDA_MG_INVERTER){ printf("calling mg solver\n"); fflush(stdout); }
  // perform the inversion
  invertQuda(spinorOut, spinorIn, &inv_param);

  if(use_multigrid_quda){
    destroyMultigridQuda(mg_preconditioner);
  }

  if( inv_param.verbosity == QUDA_VERBOSE )
    if(g_proc_id == 0)
      printf("# QUDA: Device memory used:  Spinor: %f GiB,  Gauge: %f GiB, Clover: %f GiB\n",
             inv_param.spinorGiB, gauge_param.gaugeGiB, inv_param.cloverGiB);
  if( inv_param.verbosity > QUDA_SILENT )
    if(g_proc_id == 0)
      printf("# QUDA: Done: %i iter / %g secs = %g Gflops\n",
             inv_param.iter, inv_param.secs, inv_param.gflops/inv_param.secs);

  // number of CG iterations
  int iteration = inv_param.iter;

  // reorder spinor
  reorder_spinor_fromQuda( (double*)spinorIn,  inv_param.cpu_prec, 0, NULL );
  reorder_spinor_fromQuda( (double*)spinorOut, inv_param.cpu_prec, 0, NULL );
  convert_lexic_to_eo(Even,     Odd,     solver_field[0]);
  convert_lexic_to_eo(Even_new, Odd_new, solver_field[1]);

  finalize_solver(solver_field, nr_sf);
  freeGaugeQuda();
  freeCloverQuda(); // this is safe even if there is no Clover field loaded, at least it was in QUDA v0.7.2

  if(iteration >= max_iter)
    return(-1);

  return(iteration);
}

int invert_doublet_eo_quda(spinor * const Even_new_s, spinor * const Odd_new_s,
                           spinor * const Even_new_c, spinor * const Odd_new_c,
                           spinor * const Even_s, spinor * const Odd_s,
                           spinor * const Even_c, spinor * const Odd_c,
                           const double precision, const int max_iter,
                           const int solver_flag, const int rel_prec, const int even_odd_flag,
                           const SloppyPrecision sloppy_precision,
                           CompressionType compression) {

  spinor ** solver_field = NULL;
  const int nr_sf = 4;
  init_solver_field(&solver_field, VOLUMEPLUSRAND, nr_sf);

  convert_eo_to_lexic(solver_field[0],   Even_s,  Odd_s);
  convert_eo_to_lexic(solver_field[1],   Even_c,  Odd_c);
//  convert_eo_to_lexic(g_spinor_field[DUM_DERI+1], Even_new, Odd_new);

  void *spinorIn    = (void*)solver_field[0]; // source
  void *spinorIn_c  = (void*)solver_field[1]; // charme source
  void *spinorOut   = (void*)solver_field[2]; // solution
  void *spinorOut_c = (void*)solver_field[3]; // charme solution

  if ( rel_prec )
    inv_param.residual_type = QUDA_L2_RELATIVE_RESIDUAL;
  else
    inv_param.residual_type = QUDA_L2_ABSOLUTE_RESIDUAL;

  inv_param.kappa = g_kappa;

  // IMPORTANT: use opposite TM mu-flavor since gamma5 -> -gamma5
  inv_param.mu           = -g_mubar /2./g_kappa;
  inv_param.epsilon      =  g_epsbar/2./g_kappa;
  // FIXME: in principle, there is also QUDA_TWIST_DEG_DOUBLET
  inv_param.twist_flavor =  QUDA_TWIST_NONDEG_DOUBLET; 
  inv_param.Ls = 2;

  // figure out which BC to use (theta, trivial...)
  set_boundary_conditions(&compression);

  // set the sloppy precision of the mixed prec solver
  set_sloppy_prec(sloppy_precision);

  // load gauge after setting precision
   _loadGaugeQuda(compression);

  // choose dslash type
  if( g_c_sw > 0.0 ) {
    inv_param.dslash_type = QUDA_TWISTED_CLOVER_DSLASH;
    inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN; // FIXME: note sure if this is the correct PC type
    inv_param.solution_type = QUDA_MAT_SOLUTION;
    inv_param.clover_order = QUDA_PACKED_CLOVER_ORDER;
    inv_param.clover_coeff = g_c_sw*g_kappa;
    inv_param.compute_clover = 1;
    inv_param.compute_clover_inverse = 1;
  }
  else {
    inv_param.dslash_type = QUDA_TWISTED_MASS_DSLASH;
    inv_param.matpc_type = QUDA_MATPC_EVEN_EVEN_ASYMMETRIC;
    inv_param.solution_type = QUDA_MAT_SOLUTION;
  }

  // choose solver
  if(solver_flag == BICGSTAB) {
    if(g_proc_id == 0) {printf("# QUDA: Using BiCGstab!\n"); fflush(stdout);}
    inv_param.inv_type = QUDA_BICGSTAB_INVERTER;
  }
  else {
    /* Here we invert the hermitean operator squared */
    inv_param.inv_type = QUDA_CG_INVERTER;
    if(g_proc_id == 0) {
      printf("# QUDA: Using mixed precision CG!\n");
      printf("# QUDA: mu = %.12f, kappa = %.12f\n", g_mu/2./g_kappa, g_kappa);
      fflush(stdout);
    }
  }

  if( even_odd_flag ) {
    inv_param.solve_type = QUDA_NORMERR_PC_SOLVE;
    if(g_proc_id == 0) printf("# QUDA: Using preconditioning!\n");
  }
  else {
    inv_param.solve_type = QUDA_NORMERR_SOLVE;
    if(g_proc_id == 0) printf("# QUDA: Not using preconditioning!\n");
  }
  inv_param.verbosity= get_verbosity_type(Quda_invert_input.inv_verbosity);

  inv_param.tol = sqrt(precision);
  inv_param.maxiter = max_iter;

  // NULL pointers to host fields to force
  // construction instead of download of the clover field:
  if( g_c_sw > 0.0 )
    loadCloverQuda(NULL, NULL, &inv_param);

  // reorder spinor
  reorder_spinor_toQuda( (double*)spinorIn,   inv_param.cpu_prec, 1, (double*)spinorIn_c );

  // perform the inversion
  invertQuda(spinorOut, spinorIn, &inv_param);

  if( inv_param.verbosity == QUDA_VERBOSE )
    if(g_proc_id == 0)
      printf("# QUDA: Device memory used:  Spinor: %f GiB,  Gauge: %f GiB, Clover: %f GiB\n",
             inv_param.spinorGiB, gauge_param.gaugeGiB, inv_param.cloverGiB);
  if( inv_param.verbosity > QUDA_SILENT )
    if(g_proc_id == 0)
      printf("# QUDA: Done: %i iter / %g secs = %g Gflops\n",
             inv_param.iter, inv_param.secs, inv_param.gflops/inv_param.secs);

  // number of CG iterations
  int iteration = inv_param.iter;

  // reorder spinor
  reorder_spinor_fromQuda( (double*)spinorIn,    inv_param.cpu_prec, 1, (double*)spinorIn_c );
  reorder_spinor_fromQuda( (double*)spinorOut,   inv_param.cpu_prec, 1, (double*)spinorOut_c );
  convert_lexic_to_eo(Even_s,     Odd_s,     solver_field[0]);
  convert_lexic_to_eo(Even_c,     Odd_c,     solver_field[1]);
  convert_lexic_to_eo(Even_new_s, Odd_new_s, solver_field[2]);
  convert_lexic_to_eo(Even_new_c, Odd_new_c, solver_field[3]);

  finalize_solver(solver_field, nr_sf);
  freeGaugeQuda();
  freeCloverQuda(); // this is safe even if there is no Clover field loaded, at least it was in QUDA v0.7.2

  if(iteration >= max_iter)
    return(-1);

  return(iteration);
}

// if even_odd_flag set
void M_full_quda(spinor * const Even_new, spinor * const Odd_new,  spinor * const Even, spinor * const Odd) {
  inv_param.kappa = g_kappa;
  // IMPORTANT: use opposite TM flavor since gamma5 -> -gamma5 (until LXLYLZT prob. resolved)
  inv_param.mu = -g_mu;
  inv_param.epsilon = 0.0;

  inv_param.twist_flavor = QUDA_TWIST_SINGLET;
  inv_param.Ls = (inv_param.twist_flavor == QUDA_TWIST_NONDEG_DOUBLET ||
       inv_param.twist_flavor == QUDA_TWIST_DEG_DOUBLET ) ? 2 : 1;

  void *spinorIn  = (void*)g_spinor_field[DUM_DERI];   // source
  void *spinorOut = (void*)g_spinor_field[DUM_DERI+1]; // solution

  // reorder spinor
  convert_eo_to_lexic( spinorIn, Even, Odd );
  reorder_spinor_toQuda( (double*)spinorIn, inv_param.cpu_prec, 0, NULL );

  // multiply
  inv_param.solution_type = QUDA_MAT_SOLUTION;
  MatQuda( spinorOut, spinorIn, &inv_param);

  // reorder spinor
  reorder_spinor_fromQuda( (double*)spinorOut, inv_param.cpu_prec, 0, NULL );
  convert_lexic_to_eo( Even_new, Odd_new, spinorOut );
}

// no even-odd
void D_psi_quda(spinor * const P, spinor * const Q) {
  inv_param.kappa = g_kappa;
  // IMPORTANT: use opposite TM flavor since gamma5 -> -gamma5 (until LXLYLZT prob. resolved)
  inv_param.mu = -g_mu;
  inv_param.epsilon = 0.0;

  inv_param.twist_flavor = QUDA_TWIST_SINGLET;
  inv_param.Ls = (inv_param.twist_flavor == QUDA_TWIST_NONDEG_DOUBLET ||
       inv_param.twist_flavor == QUDA_TWIST_DEG_DOUBLET ) ? 2 : 1;

  void *spinorIn  = (void*)Q;
  void *spinorOut = (void*)P;

  // reorder spinor
  reorder_spinor_toQuda( (double*)spinorIn, inv_param.cpu_prec, 0, NULL );

  // multiply
  inv_param.solution_type = QUDA_MAT_SOLUTION;
  MatQuda( spinorOut, spinorIn, &inv_param);

  // reorder spinor
  reorder_spinor_fromQuda( (double*)spinorIn,  inv_param.cpu_prec, 0, NULL );
  reorder_spinor_fromQuda( (double*)spinorOut, inv_param.cpu_prec, 0, NULL );
}

void _setMultigridParam(QudaMultigridParam* mg_param) {
  QudaInvertParam *mg_inv_param = mg_param->invert_param;

  mg_inv_param->Ls = 1;
  mg_inv_param->sp_pad = 0;
  mg_inv_param->cl_pad = 0;

  mg_inv_param->preserve_source = QUDA_PRESERVE_SOURCE_NO;
  mg_inv_param->gamma_basis = QUDA_DEGRAND_ROSSI_GAMMA_BASIS;
  mg_inv_param->dirac_order = QUDA_DIRAC_ORDER;

  mg_inv_param->input_location = QUDA_CPU_FIELD_LOCATION;
  mg_inv_param->output_location = QUDA_CPU_FIELD_LOCATION;
  
  // currently, only QUDA_DIRECT_SOLVE is supported for this, thus also QUDA_MAT_SOLUTION 
  mg_inv_param->solve_type = QUDA_DIRECT_SOLVE;
  mg_inv_param->solution_type = QUDA_MAT_SOLUTION;

  mg_inv_param->dagger = QUDA_DAG_NO;

  // move away from maximal twist for subspace generation and increase mu on the coarse levels to get quick
  // coarse grid solves
  if(mg_inv_param->dslash_type != QUDA_WILSON_DSLASH && mg_inv_param->dslash_type != QUDA_CLOVER_WILSON_DSLASH){
    // FIXME: the extent to which this has to happen depends on the action and should be
    // user configurable
    mg_inv_param->kappa = mg_inv_param->kappa - 0.0025*mg_inv_param->kappa;
    if(mg_inv_param->dslash_type == QUDA_TWISTED_CLOVER_DSLASH) mg_inv_param->clover_coeff = mg_inv_param->kappa*g_c_sw;

    // FIXME: provide access to the scaling parameter
    // not sure if this is required
    if(mg_inv_param->mu > 0.0) mg_inv_param->mu = 5.2*mg_inv_param->mu;
  }

  // FIXME: allow these parameters to be adjusted
  if (Quda_multigrid_input.nlevel > QUDA_MAX_MG_LEVEL ){
    if (g_cart_id == 0) {printf("Error in number of levels in QUDA MG %d\n", Quda_multigrid_input.nlevel );
                         printf("It exceeds the maximal number of levels\n");
                         fflush(stdout);
                         exit(1);
                        }
  }
  mg_param->n_level = Quda_multigrid_input.nlevel;
  if (Quda_multigrid_input.blocksize[0][0] != 0){
      unsigned int extent=1;
      for (int i=0; i<mg_param->n_level; i++){
        extent*=Quda_multigrid_input.blocksize[0][i];
      }
      if (extent != LX){
        if (g_cart_id == 0) { printf("Error in input for block sizes in direction x\n");
                              fflush(stdout);
                              exit(1);
                            }
      }
      for (int i=0; i<mg_param->n_level; i++){
        mg_param->geo_block_size[i][0]=Quda_multigrid_input.blocksize[0][i];
      }
      extent=1;
      for (int i=0; i<mg_param->n_level; i++){
        extent*=Quda_multigrid_input.blocksize[1][i];
      }
      if (extent != LY){
        if (g_cart_id == 0) { printf("Error in input for block sizes in direction y\n");
                              fflush(stdout);
                              exit(1);
                            }
      }
      for (int i=0; i<mg_param->n_level; i++){
        mg_param->geo_block_size[i][1]=Quda_multigrid_input.blocksize[1][i];
      }
      extent=1;
      for (int i=0; i<mg_param->n_level; i++){
        extent*=Quda_multigrid_input.blocksize[2][i];
      }
      if (extent != LZ){
        if (g_cart_id == 0) { printf("Error in input for block sizes in direction z\n");
                              fflush(stdout);
                              exit(1);
                            }
      }
      for (int i=0; i<mg_param->n_level; i++){
        mg_param->geo_block_size[i][2]=Quda_multigrid_input.blocksize[2][i];
      }
      extent=1;
      for (int i=0; i<mg_param->n_level; i++){
        extent*=Quda_multigrid_input.blocksize[3][i];
      }
      if (extent != T){
        if (g_cart_id == 0) { printf("Error in input for block sizes in direction T\n");
                              fflush(stdout);
                              exit(1);
                            }
      }
      for (int i=0; i<mg_param->n_level; i++){
        mg_param->geo_block_size[i][3]=Quda_multigrid_input.blocksize[3][i];
      }
  }
  else{
    for (int i=0; i<mg_param->n_level; i++) {
      for (int j=0; j<QUDA_MAX_DIM; j++) {

        unsigned int extent = (j == 4) ? T : LX;
        // determine how many lattice sites remain at this level
        for(int k = i; k > 0; k--) extent = extent/mg_param->geo_block_size[k-1][j];
        unsigned int even_block_size = 4;
        if(extent < 8) even_block_size = 2;

        // on the finest level, we use a block size of 4^4
        // on all other levels, we compute how many blocks there are and use a block size of 3 for
        // cases divisible by 3, otherwise we use 2
        // if blocking is not possible, we use block size 1
        if( extent == 1 ) mg_param->geo_block_size[i][j] = 1;
        else mg_param->geo_block_size[i][j] = (i == 0) ? 4 : ( (extent % 3 == 0) ? 3 : even_block_size );
      }
    }
  }
  for (int i=0; i<mg_param->n_level; i++) {
    mg_param->verbosity[i]= get_verbosity_type(Quda_multigrid_input.mg_verbosity[i]);
    mg_param->setup_inv_type[i]= get_solver_type(Quda_multigrid_input.setup_inv[i]);
    mg_param->setup_tol[i] = Quda_multigrid_input.setup_tol;
    mg_param->spin_block_size[i] = 1;
    mg_param->n_vec[i] = Quda_multigrid_input.nvec;
    mg_param->nu_pre[i] = Quda_multigrid_input.nu_pre;
    mg_param->nu_post[i] = Quda_multigrid_input.nu_post;
    mg_param->cycle_type[i] = QUDA_MG_CYCLE_RECURSIVE;

    mg_param->smoother[i]= get_solver_type(Quda_multigrid_input.smoother_type);
/*
    if (strcmp(Quda_multigrid_input.smoother_type,"QUDA_MR_INVERTER") == 0 ) {
       mg_param->smoother[i] = QUDA_MR_INVERTER;
    }
    else{
       if ( g_cart_id == 0 ){ 
         printf("For smoother now you should you QUDA_MR_INVERTER\n");
         fflush(stdout);
         exit(1);
       } 
    }
*/
    // set the smoother / bottom solver tolerance (for MR smoothing this will be ignored)
    mg_param->smoother_tol[i] = Quda_multigrid_input.smoother_tol;//0.1; // repurpose heavy-quark tolerance for now

    mg_param->global_reduction[i] = QUDA_BOOLEAN_YES;

    // Kate says this should be EO always for performance
    mg_param->smoother_solve_type[i] = QUDA_DIRECT_PC_SOLVE;

    // set to QUDA_MAT_SOLUTION to inject a full field into coarse grid
    // set to QUDA_MATPC_SOLUTION to inject single parity field into coarse grid
    // if we are using an outer even-odd preconditioned solve, then we
    // use single parity injection into the coarse grid
    mg_param->coarse_grid_solution_type[i] = inv_param.solve_type == QUDA_DIRECT_PC_SOLVE ? QUDA_MATPC_SOLUTION : QUDA_MAT_SOLUTION;

    mg_param->omega[i] = Quda_multigrid_input.omega;//0.85; // over/under relaxation factor

    mg_param->location[i] = QUDA_CUDA_FIELD_LOCATION;
  }

  // only coarsen the spin on the first restriction
  mg_param->spin_block_size[0] = 2;

  // coarse grid solver is GCR
  mg_param->smoother[mg_param->n_level-1] = QUDA_GCR_INVERTER;

  mg_param->compute_null_vector = QUDA_COMPUTE_NULL_VECTOR_YES;
  mg_param->generate_all_levels = QUDA_BOOLEAN_NO;

  mg_param->run_verify = QUDA_BOOLEAN_NO;

  // set file i/o parameters
  strcpy(mg_param->vec_infile, "");
  strcpy(mg_param->vec_outfile, "");

  mg_inv_param->verbosity = QUDA_SUMMARIZE;
  mg_inv_param->verbosity_precondition = QUDA_SUMMARIZE;;
}

