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
 ***********************************************************************/

#include <errno.h>
#include "global.h"
#include "scalar.h"
#include "buffers/utils_nonblocking.h"

extern int scalar_precision_read_flag;
// TODO consider that input scalar field could be in single prec.

int read_scalar_field(char * filename, scalar ** const sf) {

  FILE *ptr;

  int count = 4*VOLUME;
  int scalarreadsize = ( scalar_precision_read_flag==64 ? sizeof(double) : sizeof(float) );

  ptr = fopen(filename,"rb");  // r for read, b for binary
  // read into buffer
  void *buffer;
  if((buffer = malloc(count*scalarreadsize)) == NULL) {
    printf ("malloc errno : %d\n",errno);
    errno = 0;
    return(2);
  }

  if( count > fread(buffer,scalarreadsize,count,ptr) )
    return(-1);

  // copy to sf
  for( int s = 0; s < 4; s++ ) {
    for( int i = 0; i < VOLUME; i++ ) {
      if ( scalar_precision_read_flag == 64 )
	sf[s][i] = ((double*)buffer)[4*i+s];
      else
	sf[s][i] = ((float*)buffer)[4*i+s];
    }
  }

  return(0);
}
int read_scalar_field_parallel( char * filename, scalar ** const sf){
  int t;
  FILE *ptr;
  int count = 4*LX*N_PROC_X*LY*N_PROC_Y*LZ*N_PROC_Z;
  int scalarreadsize = ( scalar_precision_read_flag==64 ? sizeof(double) : sizeof(float) );

  ptr = fopen(filename,"rb");  // r for read, b for binary

  // read into buffer
  void *buffer;
  if((buffer = malloc(count*scalarreadsize)) == NULL) {
    printf ("malloc errno : %d\n",errno);
    errno = 0;
    return(2);
  }

  for (t=0; t< T_global; ++t){
    
      if ( g_proc_id == 0 ){
         
          int nread= fread(buffer, scalarreadsize, count, ptr);
          if ( nread != count ) { printf("Error in reading the scalar fields, exiting ...\n"); exit(1); }
  
      }
      MPI_Barrier(MPI_COMM_WORLD);
  
      MPI_Bcast(buffer, count,  scalar_precision_read_flag==64 ? MPI_DOUBLE : MPI_FLOAT ,0, MPI_COMM_WORLD );

      int ix, j;
      for (ix=0; ix< VOLUME; ++ix)
         if ( g_coord[ix][0] == t ){
            int ind = LY*N_PROC_Y*LZ*N_PROC_Z*g_coord[ix][1] + LZ*N_PROC_Z*g_coord[ix][2] + g_coord[ix][3];
            for (j=0; j<4; ++j){
               if ( scalar_precision_read_flag == 64 )
                 sf[j][ix] = ((double*)buffer)[4*ind+j];
               else
                 sf[j][ix] = ((float*)buffer)[4*ind+j];

            }
         }
 //     if (g_proc_id == 1) printf("Buffer coordinate %e\n",((double*)buffer)[0]);
      MPI_Barrier(MPI_COMM_WORLD);
           
  }
  free(buffer);
  return(0);
}
void smear_scalar_fields( scalar ** const sf, scalar ** smearedfield ) {

   int ix;
   int in;

   scalar *tmps1= (scalar *)malloc(sizeof(scalar)*VOLUMEPLUSRAND );
   scalar *tmps2= (scalar *)malloc(sizeof(scalar)*VOLUMEPLUSRAND );

   scalar *hyperc= (scalar *)malloc(sizeof(scalar)*VOLUMEPLUSRAND );
   scalar *nearen= (scalar *)malloc(sizeof(scalar)*VOLUMEPLUSRAND );

   int neit, neix, neiy, neiz;
   MPI_Status  statuses[8];
   MPI_Request *request;
   request=( MPI_Request *) malloc(sizeof(MPI_Request)*8);

   int count=0;

// hypercubic smearing 

   for (in = 0; in<4 ; ++in ){
      for (ix=0; ix<VOLUME; ++ix){
         smearedfield[in][ix]=0.0;
      } 
   }
   for (in=0; in<4; ++in) {
      for (ix=0; ix<VOLUME; ++ix){
         nearen[ix]=sf[in][ix];
         hyperc[ix]=sf[in][ix];
      }
      for (neit=0; neit<2; ++neit)
         for (neix=0; neix<2; ++neix)
            for (neiy=0; neiy<2; ++neiy)
               for (neiz=0; neiz<2; ++neiz){

                  count=0;
                  generic_exchange_direction_nonblocking( sf[in], sizeof(scalar), neit ? TDOWN : TUP, request, &count );
                  MPI_Waitall( count, request, statuses);
                  for (ix=0; ix<VOLUME; ++ix)
                     tmps1[ix]= sf[in][    neit ? g_idn[ix][TUP] : g_iup[ix][TUP] ];

                  count=0;
                  generic_exchange_direction_nonblocking(  tmps1, sizeof(scalar), neix ? XDOWN : XUP, request, &count );
                  MPI_Waitall( count, request, statuses);
                  for (ix=0; ix<VOLUME; ++ix)
                     tmps2[ix]= tmps1[ neix ? g_idn[ix][XUP] : g_iup[ix][XUP] ];

                  count=0;
                  generic_exchange_direction_nonblocking(  tmps2, sizeof(scalar), neiy ? YDOWN : YUP, request, &count );
                  MPI_Waitall( count, request, statuses);
                  for (ix=0; ix<VOLUME; ++ix)
                     tmps1[ix]= tmps2[ neiy ? g_idn[ix][YUP] : g_iup[ix][YUP] ];

                  count=0;
                  generic_exchange_direction_nonblocking(  tmps1, sizeof(scalar), neix ? ZDOWN : ZUP, request, &count );
                  MPI_Waitall( count, request, statuses);
                  for (ix=0; ix<VOLUME; ++ix)
                     tmps2[ix]= tmps1[ neiz ? g_idn[ix][ZUP] : g_iup[ix][ZUP] ];

                  for (ix=0; ix<VOLUME; ++ix)
                     hyperc[ix]+=tmps2[ix];
               }
      for (ix =0; ix<VOLUME; ++ix){
         hyperc[ix]/=17.0;
      }
      count=0;
      generic_exchange_direction_nonblocking( sf[in], sizeof(scalar), TDOWN, request, &count );
      MPI_Waitall( count, request, statuses);
      for (ix=0; ix<VOLUME; ++ix)
         tmps1[ix]= sf[in][g_idn[ix][TUP]];
      for (ix=0; ix<VOLUME; ++ix)
         nearen[ix]+= tmps1[ix];
      
      count=0;
      generic_exchange_direction_nonblocking( sf[in], sizeof(scalar), TUP ,  request, &count );
      MPI_Waitall( count, request, statuses);      
      for (ix=0; ix<VOLUME; ++ix)
         tmps1[ix]= sf[in][g_iup[ix][TUP]];
      for (ix=0; ix<VOLUME; ++ix)
         nearen[ix]+= tmps1[ix];
      
      count=0;
      generic_exchange_direction_nonblocking( sf[in], sizeof(scalar), XDOWN, request, &count );
      MPI_Waitall( count, request, statuses);      
      for (ix=0; ix<VOLUME; ++ix)
         tmps1[ix]= sf[in][g_idn[ix][XUP]];
      for (ix=0; ix<VOLUME; ++ix)
         nearen[ix]+= tmps1[ix];

      count=0;
      generic_exchange_direction_nonblocking( sf[in], sizeof(scalar), XUP , request, &count );
      MPI_Waitall( count, request, statuses);
      for (ix=0; ix<VOLUME; ++ix)
         tmps1[ix]= sf[in][g_iup[ix][XUP]];
      for (ix=0; ix<VOLUME; ++ix)
         nearen[ix]+= tmps1[ix];

      count=0;    
      generic_exchange_direction_nonblocking( sf[in], sizeof(scalar), YDOWN, request, &count );
      MPI_Waitall( count, request, statuses);      
      for (ix=0; ix<VOLUME; ++ix)
         tmps1[ix]= sf[in][g_idn[ix][YUP]];
      for (ix=0; ix<VOLUME; ++ix)
         nearen[ix]+= tmps1[ix];
    
      count=0;
      generic_exchange_direction_nonblocking( sf[in], sizeof(scalar), YUP, request, &count );
      MPI_Waitall( count, request, statuses);
      for (ix=0; ix<VOLUME; ++ix)
         tmps1[ix]= sf[in][g_iup[ix][YUP]];
      for (ix=0; ix<VOLUME; ++ix)
         nearen[ix]+= tmps1[ix];

      count=0;
      generic_exchange_direction_nonblocking( sf[in], sizeof(scalar), ZDOWN, request, &count );
      MPI_Waitall( count, request, statuses);      
      for (ix=0; ix<VOLUME; ++ix)
         tmps1[ix]= sf[in][g_idn[ix][ZUP]];
      for (ix=0; ix<VOLUME; ++ix)
         nearen[ix]+= tmps1[ix];

      count=0;
      generic_exchange_direction_nonblocking( sf[in], sizeof(scalar), ZUP, request, &count );
      MPI_Waitall( count, request, statuses);
      for (ix=0; ix<VOLUME; ++ix)
         tmps1[ix]= sf[in][g_iup[ix][ZUP]];
      for (ix=0; ix<VOLUME; ++ix)
         nearen[ix]+= tmps1[ix];


      for (ix =0; ix<VOLUME; ++ix){
         nearen[ix]/=9.0;
      }

      for (ix=0; ix<VOLUME; ++ix){
         smearedfield[in][ix]=0.5*(nearen[ix] + hyperc[ix] );
      }
   }
   free(tmps1);
   free(tmps2);

   free(hyperc);
   free(nearen);
   free(request);
}
void smear_scalar_fields_correlator( scalar ** const sf, scalar ** smearedfield ) {

   int x0,y0,z0,t0;
   double timeslicesum[4];
   double mpi_res;
   int j;
   for (j = 0; j<4 ; ++j ){
      for (x0=0; x0<VOLUME; ++x0){
         smearedfield[j][x0]=0.0;
      }
   }
   for (t0=0; t0<T; ++t0){
      for (j=0; j<4; ++j){
         timeslicesum[j]=0.0;
         mpi_res=0.;
//summing over the local volume

         for (x0; x0<LX; x0++)
            for (y0=0; y0<LY; ++y0)
               for (z0=0; z0<LZ; ++z0){
                  timeslicesum[j]+=sf[j][((t0*LX + x0)*LY + y0)*LZ + z0];
               }
#if defined MPI
         MPI_Reduce(&timeslicesum[j], &mpi_res, 1, MPI_DOUBLE, MPI_SUM, 0, g_mpi_time_slices);
#endif
         mpi_res/=(double)VOLUME*N_PROC_X*N_PROC_Y*N_PROC_Z;
         for (x0; x0<LX; x0++)
            for (y0=0; y0<LY; ++y0)
               for (z0=0; z0<LZ; ++z0){
                  smearedfield[j][((t0*LX + x0)*LY + y0)*LZ + z0]=mpi_res;

         }
      }
   }
}

