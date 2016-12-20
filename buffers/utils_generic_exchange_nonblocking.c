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
#include "utils_nonblocking.ih"

#ifndef MPI /*Let's deal with this case once and for all*/
void generic_exchange_direction_nonblocking(void *field_in, int bytes_per_site, int direction, MPI_Request *inreq, int* counter)
{}
#else /* MPI */
void generic_exchange_direction_nonblocking(void *field_in, int bytes_per_site, int direction, MPI_Request *inreq, int* counter)
{
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

  if (direction == TUP){
#    if (defined PARALLELT || defined PARALLELXT || defined PARALLELXYT || defined PARALLELXYZT)
       MPI_Isend(buffer[0],          1, slice_T_cont_type, g_nb_t_dn, 83,
            g_cart_grid, &inreq[*counter  ]);
       MPI_Irecv(buffer[VOLUME],     1, slice_T_cont_type, g_nb_t_up, 83,
            g_cart_grid, &inreq[*counter+1]);
       *counter=*counter+2;
#     endif
  }
  if (direction == TDOWN){
#     if (defined PARALLELT || defined PARALLELXT || defined PARALLELXYT || defined PARALLELXYZT)
       MPI_Isend(buffer[(T-1)*LX*LY*LZ], 1, slice_T_cont_type, g_nb_t_up, 84,
            g_cart_grid, &inreq[*counter  ]);
       MPI_Irecv(buffer[(T+1)*LX*LY*LZ], 1, slice_T_cont_type, g_nb_t_dn, 84,
            g_cart_grid, &inreq[*counter+1]);
       *counter=*counter+2;
#     endif
  }
  if (direction == XUP){
#    if (defined PARALLELXT || defined PARALLELXYT || defined PARALLELXYZT)
      MPI_Isend(buffer[0],              1, slice_X_gath_type, g_nb_x_dn, 87,
            g_cart_grid, &inreq[*counter  ]);
      MPI_Irecv(buffer[(T+2)*LX*LY*LZ], 1, slice_X_cont_type, g_nb_x_up, 87,
            g_cart_grid, &inreq[*counter+1]);
      *counter=*counter+2;
#    endif
  }
  if (direction == XDOWN){
#    if (defined PARALLELXT || defined PARALLELXYT || defined PARALLELXYZT)
      MPI_Isend(buffer[(LX-1)*LY*LZ],             1, slice_X_gath_type, g_nb_x_up, 88,
            g_cart_grid, &inreq[*counter  ]);
      MPI_Irecv(buffer[(T+2)*LX*LY*LZ + T*LY*LZ], 1, slice_X_cont_type, g_nb_x_dn, 88,
            g_cart_grid, &inreq[*counter+1]);
      *counter=*counter+2;
#    endif
  }
  if (direction == YUP){
#    if (defined PARALLELXYT || defined PARALLELXYZT)
      MPI_Isend(buffer[0],                            1, slice_Y_gath_type, g_nb_y_dn, 106,
            g_cart_grid, &inreq[*counter]);
      MPI_Irecv(buffer[VOLUME + 2*LZ*(LX*LY + T*LY)], 1, slice_Y_cont_type, g_nb_y_up, 106,
            g_cart_grid, &inreq[*counter+1]);
      *counter=*counter+2;
#    endif
  }
  if (direction == YDOWN){
#    if (defined PARALLELXYT || defined PARALLELXYZT)
      MPI_Isend(buffer[(LY-1)*LZ],                              1, slice_Y_gath_type, g_nb_y_up, 107,
            g_cart_grid, &inreq[*counter]);
      MPI_Irecv(buffer[VOLUME + 2*LZ*(LX*LY + T*LY) + T*LX*LZ], 1, slice_Y_cont_type, g_nb_y_dn, 107,
            g_cart_grid, &inreq[*counter+1]);
      *counter=*counter+2;
#    endif
  }
  if (direction == ZUP){
#    if defined PARALLELXYZT
      MPI_Isend(buffer[0],
            1, slice_Z_gath_type, g_nb_z_dn, 122,
            g_cart_grid, &inreq[*counter]);
      MPI_Irecv(buffer[VOLUME + 2*LZ*(LX*LY + T*LY) + 2*LZ*T*LX],
            1, slice_Z_cont_type, g_nb_z_up, 122,
            g_cart_grid, &inreq[*counter+1]);
      *counter=*counter+2;
#    endif
  }
  if (direction == ZDOWN){
#    if defined PARALLELXYZT
      MPI_Isend(buffer[LZ-1],
            1, slice_Z_gath_type, g_nb_z_up, 123,
            g_cart_grid, &inreq[*counter]);
      MPI_Irecv(buffer[VOLUME + 2*LZ*(LX*LY + T*LY) + 2*T*LX*LZ + T*LX*LY],
            1, slice_Z_cont_type, g_nb_z_dn, 123,
            g_cart_grid, &inreq[*counter+1]);
      *counter=*counter+2;
#    endif
  }
}
#endif /* MPI */

