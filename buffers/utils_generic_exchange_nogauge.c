/***********************************************************************
 *
 * Copyright (C) 2009 Carsten Urbach
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
#include "utils_nogauge.h"
#ifdef TM_USE_BSM
#ifndef TM_USE_MPI /*Let's deal with this case once and for all*/
void generic_exchange_nogauge(void *field_in, int bytes_per_site )
{}
#else /* MPI */
void generic_exchange_nogauge(void *field_in, int bytes_per_site )
{
#if defined _NON_BLOCKING
  int cntr=0;
  MPI_Request request[108];
  MPI_Status  status[108];
#else /* _NON_BLOCKING */
  MPI_Status status;
#endif /* _NON_BLOCKING */
  static int initialized = 0;

  /* We start by defining all the MPI datatypes required */
  static MPI_Datatype site_type;

  static MPI_Datatype slice_X_cont_type, slice_Y_cont_type, slice_Z_cont_type, slice_T_cont_type;
  static MPI_Datatype slice_X_subs_type, slice_Y_subs_type;
  static MPI_Datatype slice_X_gath_type, slice_Y_gath_type, slice_Z_gath_type;

  unsigned char(*buffer)[bytes_per_site] = field_in; /* To allow for pointer arithmetic */

  // To avoid continuous MPI operations on these local variables, let's declare them static.
  // That means we should only initialize if this is the first use of the function, or if
  // the existing initialization is for the wrong number of bytes per size!
  if (initialized && (initialized != bytes_per_site))
  {
    MPI_Type_free(&site_type);

    MPI_Type_free(&slice_T_cont_type);
    MPI_Type_free(&slice_X_cont_type);
    MPI_Type_free(&slice_Y_cont_type);
    MPI_Type_free(&slice_Z_cont_type);

    MPI_Type_free(&slice_X_subs_type);
    MPI_Type_free(&slice_Y_subs_type);

    MPI_Type_free(&slice_X_gath_type);
    MPI_Type_free(&slice_Y_gath_type);
    MPI_Type_free(&slice_Z_gath_type);


    /* We're ready to reinitialize all these types now... */
    initialized = 0;
  }

  if (!initialized)
  {
    /* Initialization of the datatypes - adapted from mpi_init.c */
    MPI_Type_contiguous(bytes_per_site, MPI_BYTE, &site_type);
    MPI_Type_commit(&site_type);

    MPI_Type_contiguous(LX * LY *LZ, site_type, &slice_T_cont_type);
    MPI_Type_contiguous( T * LY *LZ, site_type, &slice_X_cont_type);
    MPI_Type_contiguous( T * LX *LZ, site_type, &slice_Y_cont_type);
    MPI_Type_contiguous( T * LX *LY, site_type, &slice_Z_cont_type);

    MPI_Type_commit(&slice_T_cont_type);
    MPI_Type_commit(&slice_X_cont_type);
    MPI_Type_commit(&slice_Y_cont_type);
    MPI_Type_commit(&slice_Z_cont_type);

    MPI_Type_contiguous(LY * LZ, site_type, &slice_X_subs_type);
    MPI_Type_contiguous(LZ, site_type, &slice_Y_subs_type);

    MPI_Type_commit(&slice_X_subs_type);
    MPI_Type_commit(&slice_Y_subs_type);

    MPI_Type_vector(T, 1, LX, slice_X_subs_type, &slice_X_gath_type);
    MPI_Type_vector(T * LX, 1, LY, slice_Y_subs_type, &slice_Y_gath_type);
    MPI_Type_vector(T * LX * LY, 1, LZ, site_type,  &slice_Z_gath_type);

    MPI_Type_commit(&slice_X_gath_type);
    MPI_Type_commit(&slice_Y_gath_type);
    MPI_Type_commit(&slice_Z_gath_type);

    initialized = bytes_per_site;
  }

  /* Following are implementations using different compile time flags */
# include "utils_generic_exchange_nogauge.inc"
}

#endif /* MPI */
#endif /* TM_USE_BSM */
