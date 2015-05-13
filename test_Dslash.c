/*
 * test_Dslash.c
 *
 *  Created on: Nov 13, 2014
 *      Author: mario
 */

/*******************************************************************************
*
* test program for Dslash (D_psi)
*
*
*******************************************************************************/
#define TEST_INVERSION 1   // if 0, then test only Dslash
#define TIMESLICE_SOURCE 1 // if 0, then volume source

#ifdef HAVE_CONFIG_H
# include<config.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>
#if (defined BGL && !defined BGP)
#  include <rts.h>
#endif
#ifdef _MPI
# include <mpi.h>
# ifdef HAVE_LIBLEMON
#  include <io/params.h>
#  include <io/gauge.h>
# endif
#endif
#ifdef OMP
# include <omp.h>
# include "init/init_openmp.h"
#endif
#include "gettime.h"
#include "su3.h"
#include "su3adj.h"
#include "ranlxd.h"
#include "geometry_eo.h"
#include "read_input.h"
#include "start.h"
#include "boundary.h"
#include "global.h"
#include "operator/Hopping_Matrix.h"
#include "operator/Hopping_Matrix_nocom.h"
#include "operator/tm_operators.h"
#include "operator/clovertm_operators.h"
#include "operator.h"
#include "solver/cg_her.h"
#include "gamma.h"
#include "xchange/xchange.h"
#include "init/init.h"
#include "test/check_geometry.h"
#include "operator/D_psi.h"
//#include "phmc.h"
#include "mpi_init.h"
#include "linalg/square_norm.h"
#include "linalg/assign_add_mul_r.h"
#include "linalg/convert_eo_to_lexic.h"
#include "prepare_source.h"
#include "operator/clovertm_operators.h"
#include "operator/clover_leaf.h"
#include "invert_clover_eo.h"
#include "qphix_interface.h"

#ifdef PARALLELT
#  define SLICE (LX*LY*LZ/2)
#elif defined PARALLELXT
#  define SLICE ((LX*LY*LZ/2)+(T*LY*LZ/2))
#elif defined PARALLELXYT
#  define SLICE ((LX*LY*LZ/2)+(T*LY*LZ/2) + (T*LX*LZ/2))
#elif defined PARALLELXYZT
#  define SLICE ((LX*LY*LZ/2)+(T*LY*LZ/2) + (T*LX*LZ/2) + (T*LX*LY/2))
#elif defined PARALLELX
#  define SLICE ((LY*LZ*T/2))
#elif defined PARALLELXY
#  define SLICE ((LY*LZ*T/2) + (LX*LZ*T/2))
#elif defined PARALLELXYZ
#  define SLICE ((LY*LZ*T/2) + (LX*LZ*T/2) + (LX*LY*T/2))
#endif

int check_xchange();

void _Q_pm_psi(spinor * const l, spinor * const k)
{
  g_mu = -g_mu;
  D_psi(l, k);
  gamma5(g_spinor_field[4], l, VOLUME);
  g_mu = -g_mu;
  D_psi(l, g_spinor_field[4]);
  gamma5(l, l, VOLUME);
}

void _Qsw_pm_psi(spinor * const l, spinor * const k) {
  /* \hat Q_{-} */
  Hopping_Matrix(EO, g_spinor_field[8+1], k);
  clover_inv(g_spinor_field[8+1], -1, g_mu);
  Hopping_Matrix(OE, g_spinor_field[8], g_spinor_field[8+1]);
  clover_gamma5(OO, g_spinor_field[8], k, g_spinor_field[8], -(g_mu + g_mu3));
  /* \hat Q_{+} */
  Hopping_Matrix(EO, l, g_spinor_field[8]);
  clover_inv(l, +1, g_mu);
  Hopping_Matrix(OE, g_spinor_field[8+1], l);
  clover_gamma5(OO, l, g_spinor_field[8], g_spinor_field[8+1], +(g_mu + g_mu3));
}

// this is the twisted clover Qhat with -mu
void _Qsw_minus_psi(spinor * const l, spinor * const k) {
  Hopping_Matrix(EO, g_spinor_field[8+1], k);
  clover_inv(g_spinor_field[8+1], -1, g_mu);
  Hopping_Matrix(OE, g_spinor_field[8], g_spinor_field[8+1]);
  clover_gamma5(OO, l, k, g_spinor_field[8], -(g_mu + g_mu3));
}

void _M_full(spinor * const Even_new, spinor * const Odd_new,
	    spinor * const Even, spinor * const Odd) {
  /* Even sites */
  Hopping_Matrix(EO, g_spinor_field[8], Odd);
  assign_mul_one_pm_imu(Even_new, Even, 1., VOLUME/2);
  assign_add_mul_r(Even_new, g_spinor_field[8], -1., VOLUME/2);

  /* Odd sites */
  Hopping_Matrix(OE, g_spinor_field[8], Even);
  assign_mul_one_pm_imu(Odd_new, Odd, 1., VOLUME/2);
  assign_add_mul_r(Odd_new, g_spinor_field[8], -1., VOLUME/2);
}

void _Msw_full(spinor * const Even_new, spinor * const Odd_new,
	      spinor * const Even, spinor * const Odd) {
  /* Even sites */
  Hopping_Matrix(EO, g_spinor_field[8], Odd);
  assign_mul_one_sw_pm_imu(EE, Even_new, Even, +g_mu);
  assign_add_mul_r(Even_new, g_spinor_field[8], -1., VOLUME/2);

  /* Odd sites */
  Hopping_Matrix(OE, g_spinor_field[8], Even);
  assign_mul_one_sw_pm_imu(OO, Odd_new, Odd, +g_mu);
  assign_add_mul_r(Odd_new, g_spinor_field[8], -1., VOLUME/2);
}

int main(int argc,char *argv[])
{
  int j,j_max,k;
  int k_max = 2; //two sources, two sinks (two Dslashes)
#ifdef HAVE_LIBLEMON
  paramsXlfInfo *xlfInfo;
#endif
  int status = 0;

  static double t1,t2,dt,sdt,dts,qdt,sqdt;
  double antioptaway=0.0;

#ifdef _MPI
  static double dt2;

  DUM_DERI = 6;
  DUM_SOLVER = DUM_DERI+2;
  DUM_MATRIX = DUM_SOLVER+6;
  NO_OF_SPINORFIELDS = DUM_MATRIX+2;

#  ifdef OMP
  int mpi_thread_provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SERIALIZED, &mpi_thread_provided);
#  else
  MPI_Init(&argc, &argv);
#  endif
  MPI_Comm_rank(MPI_COMM_WORLD, &g_proc_id);

#else
  g_proc_id = 0;
#endif

  g_rgi_C1 = 1.;

    /* Read the input file */
  if((status = read_input("test_Dslash.input")) != 0) {
    fprintf(stderr, "Could not find input file: test_Dslash.input\nAborting...\n");
    exit(-1);
  }


#ifdef OMP
  init_openmp();
#endif

  tmlqcd_mpi_init(argc, argv);

#ifdef QUDA
  if(g_proc_id==0)
	  printf("# We're using QUDA!\n");
  _initQphix(3);
#endif

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
#  ifdef _PERSISTENT
    printf("# The code was compiled for persistent MPI calls (halfspinor only)\n");
#  endif
#endif
#ifdef _MPI
#  ifdef _NON_BLOCKING
    printf("# The code was compiled for non-blocking MPI calls (spinor and gauge)\n");
#  endif
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

  if(even_odd_flag) {
    j = init_spinor_field(VOLUMEPLUSRAND, 4*k_max+2);
  }
  else {
    j = init_spinor_field(VOLUMEPLUSRAND, 2*k_max+2);
  }

  if ( j!= 0) {
    fprintf(stderr, "Not enough memory for spinor fields! Aborting...\n");
    exit(0);
  }

  if(g_proc_id == 0) {
    fprintf(stdout,"# The number of processes is %d \n",g_nproc);
    printf("# The lattice size is %d x %d x %d x %d\n",
	   (int)(T*g_nproc_t), (int)(LX*g_nproc_x), (int)(LY*g_nproc_y), (int)(g_nproc_z*LZ));
    printf("# The local lattice size is %d x %d x %d x %d\n",
	   (int)(T), (int)(LX), (int)(LY),(int) LZ);
    if(even_odd_flag) {
      printf("# testing the even/odd preconditioned Dirac operator\n");
    }
    else {
      printf("# testing the standard Dirac operator\n");
    }
    fflush(stdout);
  }

  /* define the geometry */
  geometry();
  /* define the boundary conditions for the fermion fields */
  boundary(g_kappa);

  //check BC
  if(g_proc_id == 0)
  {
	  printf("\nphase_0 = %f + I*%f\n", creal(phase_0), cimag(phase_0));
	  printf("phase_1 = %f + I*%f\n", creal(phase_1), cimag(phase_1));
	  printf("phase_2 = %f + I*%f\n", creal(phase_2), cimag(phase_2));
	  printf("phase_3 = %f + I*%f\n\n", creal(phase_3), cimag(phase_3));
  }

#ifdef _USE_HALFSPINOR
  j = init_dirac_halfspinor();
  if ( j!= 0) {
    fprintf(stderr, "Not enough memory for halfspinor fields! Aborting...\n");
    exit(0);
  }
  if(g_sloppy_precision_flag == 1) {
    g_sloppy_precision = 1;
    j = init_dirac_halfspinor32();
    if ( j!= 0) {
      fprintf(stderr, "Not enough memory for 32-Bit halfspinor fields! Aborting...\n");
      exit(0);
    }
  }
#  if (defined _PERSISTENT)
  init_xchange_halffield();
#  endif
#endif

  status = check_geometry();
  if (status != 0) {
    fprintf(stderr, "Checking of geometry failed. Unable to proceed.\nAborting....\n");
    exit(1);
  }
//#if (defined _MPI && !(defined _USE_SHMEM))
//  check_xchange();
//#endif

  start_ranlux(1, 123456);
  random_gauge_field(reproduce_randomnumber_flag, g_gauge_field);
//  unit_g_gauge_field();

#ifdef _MPI
  /*For parallelization: exchange the gaugefield */
  xchange_gauge(g_gauge_field);
#endif

#ifdef QUDA
  _loadGaugeQphix();
#endif

  //clover
  init_sw_fields(VOLUME);
  sw_term( (const su3**) g_gauge_field, g_kappa, g_c_sw);
  /* this must be EE here!   */
  /* to match clover_inv in Qsw_psi */
  sw_invert(EE, g_mu);

	/*initialize the spinor fields*/
	j_max=1;
	sdt=0.;

	for( int k=0; k<2*k_max; k++ )
			random_spinor_field_lexic(g_spinor_field[k], reproduce_randomnumber_flag, RN_GAUSS);

	if(even_odd_flag)
		for( int k=2*k_max-1; k>=0; k-=2 )
			convert_lexic_to_eo(g_spinor_field[2*k],g_spinor_field[2*k+1],g_spinor_field[k]);
//			random_spinor_field_eo(g_spinor_field[k], reproduce_randomnumber_flag, RN_GAUSS);



#if TIMESLICE_SOURCE
	if(even_odd_flag)
		for(int ix=LX*LY*LZ/2; ix<VOLUME/2; ix++ )
		{
			// even
			_spinor_null(g_spinor_field[2][ix]);
			// odd
			_spinor_null(g_spinor_field[3][ix]);
		}
	else
		for(int ix=LX*LY*LZ; ix<VOLUME; ix++ )
		{
			_spinor_null(g_spinor_field[1][ix]);
		}
#endif

	// copy
	if(even_odd_flag)
		for(int ix=0; ix<VOLUME/2; ix++ )
		{
			// even
			_spinor_assign(g_spinor_field[6][ix], g_spinor_field[2][ix]);
			// odd
			_spinor_assign(g_spinor_field[7][ix], g_spinor_field[3][ix]);
		}
	else
		for(int ix=0; ix<VOLUME; ix++ )
		{
			_spinor_assign(g_spinor_field[3][ix], g_spinor_field[1][ix]);
		}

#if defined _MPI
	if(even_odd_flag)
	{
		// even fields
		xchange_field(g_spinor_field[2],EO);
		xchange_field(g_spinor_field[6],EO);
		// odd fields
		xchange_field(g_spinor_field[3],OE);
		xchange_field(g_spinor_field[7],OE);
	}
	else
	{
		xchange_lexicfield(g_spinor_field[1]);
		xchange_lexicfield(g_spinor_field[3]);
	}
#endif

	// print L2-norm of source:
	double squarenorm;
	if(even_odd_flag)
	{
		squarenorm = square_norm(g_spinor_field[2], VOLUME/2, 1);
		if(g_proc_id==0) {
			printf("\n# ||source_e||^2 = %e\n", squarenorm);
			fflush(stdout);
		}
		squarenorm = square_norm(g_spinor_field[3], VOLUME/2, 1);
		if(g_proc_id==0) {
			printf("# ||source_o||^2 = %e\n\n", squarenorm);
			fflush(stdout);
		}
	}
	else
	{
		squarenorm = square_norm(g_spinor_field[1], VOLUME, 1);
		if(g_proc_id==0) {
			printf("\n# ||source||^2 = %e\n\n", squarenorm);
			fflush(stdout);
		}
	}

//#if TEST_INVERSION
//  init_operators();
//  //prepare_source(0, 0, 0, 0, read_source_flag, source_location);
//  operator_list[0].sr0 = g_spinor_field[0];
//  operator_list[0].sr1 = g_spinor_field[1];
//  operator_list[0].prop0 = g_spinor_field[2];
//  operator_list[0].prop1 = g_spinor_field[3];
//#endif

	solver_params_t solver_params;

	/************************** D_psi on CPU **************************/
	if(g_proc_id==0)
		printf("\n# Operator 1:\n");

#ifdef _MPI
      MPI_Barrier(MPI_COMM_WORLD);
#endif
      t1 = gettime();

#if TEST_INVERSION
      if(even_odd_flag)
      {
    	  invert_clover_eo(g_spinor_field[0], g_spinor_field[1],
    			  	    g_spinor_field[2], g_spinor_field[3],
					    1.0e-10, 1000,
					    1, 10e-10, solver_params,
					    &g_gauge_field, &_Qsw_pm_psi, &_Qsw_minus_psi);
    	  /* check result */
    	  _Msw_full(g_spinor_field[4], g_spinor_field[5], g_spinor_field[0], g_spinor_field[1]);

//    	  convert_eo_to_lexic(g_spinor_field[0], g_spinor_field[2], g_spinor_field[3]);

    	for(int ix=0; ix<VOLUME/2; ix++ )
		{
			_vector_sub_assign( g_spinor_field[4][ix].s0, g_spinor_field[2][ix].s0 );
			_vector_sub_assign( g_spinor_field[4][ix].s1, g_spinor_field[2][ix].s1 );
			_vector_sub_assign( g_spinor_field[4][ix].s2, g_spinor_field[2][ix].s2 );
			_vector_sub_assign( g_spinor_field[4][ix].s3, g_spinor_field[2][ix].s3 );

			_vector_sub_assign( g_spinor_field[5][ix].s0, g_spinor_field[3][ix].s0 );
			_vector_sub_assign( g_spinor_field[5][ix].s1, g_spinor_field[3][ix].s1 );
			_vector_sub_assign( g_spinor_field[5][ix].s2, g_spinor_field[3][ix].s2 );
			_vector_sub_assign( g_spinor_field[5][ix].s3, g_spinor_field[3][ix].s3 );
		}

		squarenorm = square_norm(g_spinor_field[4], VOLUME/2, 1);
		if(g_proc_id==0) {
			printf("\n# ||Ax-b||^2_e = %e\n", squarenorm);
			fflush(stdout);
		}
		squarenorm = square_norm(g_spinor_field[5], VOLUME/2, 1);
		if(g_proc_id==0) {
			printf("\n# ||Ax-b||^2_o = %e\n\n", squarenorm);
			fflush(stdout);
		}
      }
	  else
	  {
		  // invert
	//      operator_list[0].inverter(0, 0, 1);
		  gamma5(g_spinor_field[1], g_spinor_field[3], VOLUME);
		  cg_her(g_spinor_field[2], g_spinor_field[1], 1000, 1.0e-10,
				  1.0e-10, VOLUME, &_Q_pm_psi);
		  Q_minus_psi(g_spinor_field[0], g_spinor_field[2]);

		  // check inversion
		  D_psi(g_spinor_field[1], g_spinor_field[0]);
		for(int ix=0; ix<VOLUME; ix++ )
		{
			_vector_sub_assign( g_spinor_field[1][ix].s0, g_spinor_field[3][ix].s0 );
			_vector_sub_assign( g_spinor_field[1][ix].s1, g_spinor_field[3][ix].s1 );
			_vector_sub_assign( g_spinor_field[1][ix].s2, g_spinor_field[3][ix].s2 );
			_vector_sub_assign( g_spinor_field[1][ix].s3, g_spinor_field[3][ix].s3 );
		}

		squarenorm = square_norm(g_spinor_field[1], VOLUME, 1);
		if(g_proc_id==0) {
			printf("\n# ||Ax-b||^2 = %e\n\n", squarenorm);
			fflush(stdout);
		}
	  }

	// get pion
      if(even_odd_flag)
    	  convert_eo_to_lexic(g_spinor_field[4], g_spinor_field[0], g_spinor_field[1]);

	printf("\n# pion1: \n");
	double pionr[T];
	double pioni[T];
	for( int t = 0; t < T; t++ )
	{
		pionr[t] = 0.0;
		pioni[t] = 0.0;
		j = g_ipt[t][0][0][0];
    	for( int i = j; i < j+LX*LY*LZ; i++ )
    	{
    		if(even_odd_flag) {
    			pionr[t] += _spinor_prod_re( g_spinor_field[4][i], g_spinor_field[4][i] );
    			pioni[t] += _spinor_prod_im( g_spinor_field[4][i], g_spinor_field[4][i] );
    		} else {
    			pionr[t] += _spinor_prod_re( g_spinor_field[0][i], g_spinor_field[0][i] );
    			pioni[t] += _spinor_prod_im( g_spinor_field[0][i], g_spinor_field[0][i] );
    		}
    	}
    	printf("%i\t%f\t%f\n", t, pionr[t], pioni[t]);
	}

#else
	if(even_odd_flag)
		_M_full(g_spinor_field[0], g_spinor_field[1],
	           g_spinor_field[2], g_spinor_field[3]);
	else
		D_psi(g_spinor_field[0], g_spinor_field[1]);

#endif

      t2 = gettime();
      dt=t2-t1;
#ifdef _MPI
      MPI_Allreduce (&dt, &sdt, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#else
      sdt = dt;
#endif

    if(g_proc_id==0) {
      printf("# Time for Dslash %e sec.\n", sdt);
      printf("\n");
      fflush(stdout);
    }

// print L2-norm of result:
    if(even_odd_flag)
	{
		squarenorm = square_norm(g_spinor_field[0], VOLUME/2, 1);
		if(g_proc_id==0) {
			printf("# ||result1_e||^2 = %e\n", squarenorm);
			fflush(stdout);
		}
		squarenorm = square_norm(g_spinor_field[1], VOLUME/2, 1);
		if(g_proc_id==0) {
			printf("# ||result1_o||^2 = %e\n\n", squarenorm);
			fflush(stdout);
		}
    }
    else
    {
		squarenorm = square_norm(g_spinor_field[0], VOLUME, 1);
		if(g_proc_id==0) {
			printf("# ||result1||^2 = %e\n\n", squarenorm);
			fflush(stdout);
		}
    }


	/************************** D_psi_qphix on AVX/MIC **************************/
	if(g_proc_id==0)
		printf("\n# Operator 2:\n");

#ifdef _MPI
      MPI_Barrier(MPI_COMM_WORLD);
#endif
      t1 = gettime();

#if TEST_INVERSION
      if(even_odd_flag)
      {
    	  // invert
    	convert_eo_to_lexic(g_spinor_field[0],g_spinor_field[2], g_spinor_field[3]);
		invert_qphix(g_spinor_field[4], g_spinor_field[0], 1000, 1.0e-10, 1.0e-10 );
		convert_lexic_to_eo(g_spinor_field[0],g_spinor_field[1],g_spinor_field[4]);

		/* check result */
    	  _Msw_full(g_spinor_field[4], g_spinor_field[5], g_spinor_field[0], g_spinor_field[1]);

//    	  convert_eo_to_lexic(g_spinor_field[0], g_spinor_field[2], g_spinor_field[3]);

    	for(int ix=0; ix<VOLUME/2; ix++ )
		{
			_vector_sub_assign( g_spinor_field[4][ix].s0, g_spinor_field[2][ix].s0 );
			_vector_sub_assign( g_spinor_field[4][ix].s1, g_spinor_field[2][ix].s1 );
			_vector_sub_assign( g_spinor_field[4][ix].s2, g_spinor_field[2][ix].s2 );
			_vector_sub_assign( g_spinor_field[4][ix].s3, g_spinor_field[2][ix].s3 );

			_vector_sub_assign( g_spinor_field[5][ix].s0, g_spinor_field[3][ix].s0 );
			_vector_sub_assign( g_spinor_field[5][ix].s1, g_spinor_field[3][ix].s1 );
			_vector_sub_assign( g_spinor_field[5][ix].s2, g_spinor_field[3][ix].s2 );
			_vector_sub_assign( g_spinor_field[5][ix].s3, g_spinor_field[3][ix].s3 );
		}

		squarenorm = square_norm(g_spinor_field[4], VOLUME/2, 1);
		if(g_proc_id==0) {
			printf("\n# ||Ax-b||^2_e = %e\n", squarenorm);
			fflush(stdout);
		}
		squarenorm = square_norm(g_spinor_field[5], VOLUME/2, 1);
		if(g_proc_id==0) {
			printf("\n# ||Ax-b||^2_o = %e\n\n", squarenorm);
			fflush(stdout);
		}
      }
	  else
	  {
		// invert
		invert_qphix(g_spinor_field[2], g_spinor_field[3], 1000, 1.0e-10, 1.0e-10 );

		// check inversion
		D_psi(g_spinor_field[1], g_spinor_field[2]);
		for(int ix=0; ix<VOLUME; ix++ )
		{
			_vector_sub_assign( g_spinor_field[1][ix].s0, g_spinor_field[3][ix].s0 );
			_vector_sub_assign( g_spinor_field[1][ix].s1, g_spinor_field[3][ix].s1 );
			_vector_sub_assign( g_spinor_field[1][ix].s2, g_spinor_field[3][ix].s2 );
			_vector_sub_assign( g_spinor_field[1][ix].s3, g_spinor_field[3][ix].s3 );
		}

		squarenorm = square_norm(g_spinor_field[1], VOLUME, 1);
		if(g_proc_id==0) {
			printf("\n# ||Ax-b||^2 = %e\n\n", squarenorm);
			fflush(stdout);
		}
	  }

	// get pion
      if(even_odd_flag)
    	  convert_eo_to_lexic(g_spinor_field[4], g_spinor_field[0], g_spinor_field[1]);

	printf("\n# pion2: \n");
	for( int t = 0; t < T; t++ )
	{
		pionr[t] = 0.0;
		pioni[t] = 0.0;
		j = g_ipt[t][0][0][0];
    	for( int i = j; i < j+LX*LY*LZ; i++ )
    	{
    		if(even_odd_flag) {
    			pionr[t] += _spinor_prod_re( g_spinor_field[4][i], g_spinor_field[4][i] );
    			pioni[t] += _spinor_prod_im( g_spinor_field[4][i], g_spinor_field[4][i] );
    		} else {
    			pionr[t] += _spinor_prod_re( g_spinor_field[2][i], g_spinor_field[2][i] );
    			pioni[t] += _spinor_prod_im( g_spinor_field[2][i], g_spinor_field[2][i] );
    		}
    	}
    	printf("%i\t%f\t%f\n", t, pionr[t], pioni[t]);
	}
#else
	if(even_odd_flag)
		M_full_qphix(g_spinor_field[4], g_spinor_field[5],
	                g_spinor_field[6], g_spinor_field[7]);
	else
		D_psi_qphix(g_spinor_field[2], g_spinor_field[3]);
#endif

      t2 = gettime();
      dt=t2-t1;
#ifdef _MPI
      MPI_Allreduce (&dt, &sdt, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#else
      sdt = dt;
#endif

    if(g_proc_id==0) {
      printf("# Time for Dslash %e sec.\n", sdt);
      printf("\n");
      fflush(stdout);
    }

    // print L2-norm of result:
    if(even_odd_flag)
	{
    	// even
		squarenorm = square_norm(g_spinor_field[4], VOLUME/2, 1);
		if(g_proc_id==0) {
			printf("# ||result2_e||^2 = %e\n", squarenorm);
			fflush(stdout);
		}
		// odd
		squarenorm = square_norm(g_spinor_field[5], VOLUME/2, 1);
		if(g_proc_id==0) {
			printf("# ||result2_o||^2 = %e\n\n", squarenorm);
			fflush(stdout);
		}
	}
    else
    {
		squarenorm = square_norm(g_spinor_field[2], VOLUME, 1);
		if(g_proc_id==0) {
			printf("# ||result2||^2 = %e\n\n", squarenorm);
			fflush(stdout);
		}
	}


	/************************** finished: get difference **************************/

	// subract result1 -= result2
    if(even_odd_flag)
	{
		for(int ix=0; ix<VOLUME/2; ix++ )
		{
			// even
			_vector_sub_assign( g_spinor_field[0][ix].s0, g_spinor_field[4][ix].s0 );
			_vector_sub_assign( g_spinor_field[0][ix].s1, g_spinor_field[4][ix].s1 );
			_vector_sub_assign( g_spinor_field[0][ix].s2, g_spinor_field[4][ix].s2 );
			_vector_sub_assign( g_spinor_field[0][ix].s3, g_spinor_field[4][ix].s3 );
			// odd
			_vector_sub_assign( g_spinor_field[1][ix].s0, g_spinor_field[5][ix].s0 );
			_vector_sub_assign( g_spinor_field[1][ix].s1, g_spinor_field[5][ix].s1 );
			_vector_sub_assign( g_spinor_field[1][ix].s2, g_spinor_field[5][ix].s2 );
			_vector_sub_assign( g_spinor_field[1][ix].s3, g_spinor_field[5][ix].s3 );
		}

		// print L2-norm of result1 - result2:
		squarenorm = square_norm(g_spinor_field[0], VOLUME/2, 1);
		if(g_proc_id==0) {
			printf("# ||result1_e-result2_e||^2 = %e\n", squarenorm);
			fflush(stdout);
		}
		squarenorm = square_norm(g_spinor_field[1], VOLUME/2, 1);
		if(g_proc_id==0) {
			printf("# ||result1_o-result2_o||^2 = %e\n\n", squarenorm);
			fflush(stdout);
		}
	}
    else
    {
		for(int ix=0; ix<VOLUME; ix++ )
		{
			_vector_sub_assign( g_spinor_field[0][ix].s0, g_spinor_field[2][ix].s0 );
			_vector_sub_assign( g_spinor_field[0][ix].s1, g_spinor_field[2][ix].s1 );
			_vector_sub_assign( g_spinor_field[0][ix].s2, g_spinor_field[2][ix].s2 );
			_vector_sub_assign( g_spinor_field[0][ix].s3, g_spinor_field[2][ix].s3 );
		}

		// print L2-norm of result1 - result2:
		squarenorm = square_norm(g_spinor_field[0], VOLUME, 1);
		if(g_proc_id==0) {
			printf("# ||result1-result2||^2 = %e\n\n", squarenorm);
			fflush(stdout);
		}
	}

	// ---------------

#ifdef QUDA
  _endQphix(3);
#endif
#ifdef OMP
  free_omp_accumulators();
#endif
  free_gauge_field();
  free_geometry_indices();
  free_spinor_field();
  free_moment_field();
#ifdef _MPI
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();
#endif
  return(0);
}

