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
*
*******************************************************************************/

#ifdef HAVE_CONFIG_H
# include<config.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>
#if (defined BGL && !defined BGP)
#	include <rts.h>
#endif
#ifdef MPI
# include <mpi.h>
# ifdef HAVE_LIBLEMON
#	include <io/params.h>
#	include <io/gauge.h>
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
#include "operator/Hopping_Matrix.h"
#include "operator/Hopping_Matrix_nocom.h"
#include "operator/tm_operators.h"
#include "global.h"
#include "xchange/xchange.h"
#include "init/init.h"
#include "init/init_scalar_field.h"
#include "test/check_geometry.h"
#include "operator/D_psi.h"
#include "operator/D_psi_BSM.h"
//#include "phmc.h"
#include "mpi_init.h"
#include "buffers/utils.h"

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

int main(int argc,char *argv[])
{
#ifdef _USE_HALFSPINOR
	#undef _USE_HALFSPINOR
	printf("# WARNING: USE_HALFSPINOR will be ignored (not supported here).\n");
#endif

	if(even_odd_flag)
	{
		even_odd_flag=0;
		printf("# WARNING: even_odd_flag will be ignored (not supported here).\n");
	}
	int j,j_max,k,k_max = 1;
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

#ifdef OMP
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

	/* Read the input file */
	if((status = read_input("test_Dslash.input")) != 0) {
		fprintf(stderr, "Could not find input file: test_Dslash.input\nAborting...\n");
		exit(-1);
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


	j = init_bispinor_field(VOLUMEPLUSRAND, 2*k_max);
	if ( j!= 0) {
		fprintf(stderr, "Not enough memory for bispinor fields! Aborting...\n");
		exit(0);
	}

	int numbScalarFields = 4;
	j = init_scalar_field(VOLUMEPLUSRAND, numbScalarFields);
	if ( j!= 0) {
		fprintf(stderr, "Not enough memory for scalar fields! Aborting...\n");
		exit(0);
	}

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
	/* define the boundary conditions for the fermion fields */
	boundary(g_kappa);

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
	random_gauge_field(reproduce_randomnumber_flag, g_gauge_field);

#ifdef MPI
	/*For parallelization: exchange the gaugefield */
	xchange_gauge(g_gauge_field);
#endif

	/*initialize the bispinor fields*/
	j_max=1;
	sdt=0.;
	random_spinor_field_lexic( (spinor*)(g_bispinor_field[1]), reproduce_randomnumber_flag, RN_GAUSS);
	random_spinor_field_lexic( (spinor*)(g_bispinor_field[1])+VOLUME, reproduce_randomnumber_flag, RN_GAUSS);
#if defined MPI
//TODO		generic_exchange(g_bispinor_field[1], sizeof(bispinor));
#endif


	// random scalar field
	for( int s=0; s<numbScalarFields; s++ )
	{
		ranlxd(g_scalar_field[s], VOLUME);
#ifdef MPI
//TODO		generic_exchange(g_scalar_field[s], sizeof(scalar));
#endif
	}

#ifdef MPI
	MPI_Barrier(MPI_COMM_WORLD);
#endif
	t1 = gettime();

	/* here the actual Dslash */
	D_psi_BSM(g_bispinor_field[0], g_bispinor_field[1]);

	t2 = gettime();
	dt=t2-t1;
#ifdef MPI
	MPI_Allreduce (&dt, &sdt, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#else
	sdt = dt;
#endif

	if(g_proc_id==0) {
		printf("# Time for Dslash %e sec.\n", sdt);
		printf("\n");
		fflush(stdout);
	}

#ifdef HAVE_LIBLEMON
	if(g_proc_id==0) {
		printf("# Performing parallel IO test ...\n");
	}
	xlfInfo = construct_paramsXlfInfo(0.5, 0);
	write_gauge_field( "conf.test", 64, xlfInfo);
	free(xlfInfo);
	if(g_proc_id==0) {
		printf("# done ...\n");
	}
#endif


#ifdef OMP
	free_omp_accumulators();
#endif
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

