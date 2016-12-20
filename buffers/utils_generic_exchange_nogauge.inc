#     if (defined PARALLELT || defined PARALLELXT || defined PARALLELXYT || defined PARALLELXYZT)
       MPI_Isend(buffer[0],          1, slice_T_cont_type, g_nb_t_dn, 83,
	    g_cart_grid, &request[cntr]);
       MPI_Irecv(buffer[VOLUME], 1, slice_T_cont_type, g_nb_t_up, 83, 
	    g_cart_grid, &request[cntr+1]);
       cntr=cntr+2;
       MPI_Waitall(cntr, request, status);
#     endif

#     if (defined PARALLELT || defined PARALLELXT || defined PARALLELXYT || defined PARALLELXYZT)
       MPI_Isend(buffer[(T-1)*LX*LY*LZ], 1, slice_T_cont_type, g_nb_t_up, 84,
            g_cart_grid, &request[cntr]);
       MPI_Irecv(buffer[(T+1)*LX*LY*LZ], 1, slice_T_cont_type, g_nb_t_dn, 84,
            g_cart_grid, &request[cntr+1]);
       cntr=cntr+2;
       MPI_Waitall(cntr, request, status);
#     endif

#    if (defined PARALLELXT || defined PARALLELXYT || defined PARALLELXYZT)
      MPI_Isend(buffer[0],              1, slice_X_gath_type, g_nb_x_dn, 87,
	    g_cart_grid, &request[cntr]);
      MPI_Irecv(buffer[(T+2)*LX*LY*LZ], 1, slice_X_cont_type, g_nb_x_up, 87, 
	    g_cart_grid, &request[cntr+1]);
      cntr=cntr+2;
      MPI_Waitall(cntr, request, status);
#    endif

#    if (defined PARALLELXT || defined PARALLELXYT || defined PARALLELXYZT)
      MPI_Isend(buffer[(LX-1)*LY*LZ],             1, slice_X_gath_type, g_nb_x_up, 88,
	    g_cart_grid, &request[cntr]);
      MPI_Irecv(buffer[(T+2)*LX*LY*LZ + T*LY*LZ], 1, slice_X_cont_type, g_nb_x_dn, 88,
	    g_cart_grid, &request[cntr+1]);
      cntr=cntr+2;
      MPI_Waitall(cntr, request, status);
#    endif

#    if (defined PARALLELXYT || defined PARALLELXYZT)
      MPI_Isend(buffer[0],                            1, slice_Y_gath_type, g_nb_y_dn, 106,
	    g_cart_grid, &request[cntr]);
      MPI_Irecv(buffer[VOLUME + 2*LZ*(LX*LY + T*LY)], 1, slice_Y_cont_type, g_nb_y_up, 106, 
	    g_cart_grid, &request[cntr+1]);
      cntr=cntr+2;
      MPI_Waitall(cntr, request, status);
#    endif

#    if (defined PARALLELXYT || defined PARALLELXYZT)
      MPI_Isend(buffer[(LY-1)*LZ],                              1, slice_Y_gath_type, g_nb_y_up, 107,
	    g_cart_grid, &request[cntr]);
      MPI_Irecv(buffer[VOLUME + 2*LZ*(LX*LY + T*LY) + T*LX*LZ], 1, slice_Y_cont_type, g_nb_y_dn, 107,
	    g_cart_grid, &request[cntr+1]);
      cntr=cntr+2;
      MPI_Waitall(cntr, request, status);
#    endif

#    if defined PARALLELXYZT
      MPI_Isend(buffer[0],
	    1, slice_Z_gath_type, g_nb_z_dn, 122,
	    g_cart_grid, &request[cntr]);
      MPI_Irecv(buffer[VOLUME + 2*LZ*(LX*LY + T*LY) + 2*LZ*T*LX], 
	    1, slice_Z_cont_type, g_nb_z_up, 122, 
	    g_cart_grid, &request[cntr+1]);
      cntr=cntr+2;
      MPI_Waitall(cntr, request, status);
#    endif
#    if defined PARALLELXYZT
      MPI_Isend(buffer[LZ-1],
	    1, slice_Z_gath_type, g_nb_z_up, 123,
	    g_cart_grid, &request[cntr]);
      MPI_Irecv(buffer[VOLUME + 2*LZ*(LX*LY + T*LY) + 2*T*LX*LZ + T*LX*LY],
	    1, slice_Z_cont_type, g_nb_z_dn, 123,
	    g_cart_grid, &request[cntr+1]);
      cntr=cntr+2;
      MPI_Waitall(cntr, request, status);
#    endif
