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

#ifdef HAVE_CONFIG_H
# include<config.h>
#endif
#include"lime.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <errno.h>
#include <time.h>
#ifdef MPI
#include <mpi.h>
#endif
#include "global.h"
#include "git_hash.h"
#include "getopt.h"
#include "default_input_values.h"
#include "read_input.h"
#include "su3.h"
#include "operator/tm_operators.h"
#include "linalg_eo.h"
#include "geometry_eo.h"
#include "linalg/assign.h"
#include "operator/D_psi.h"
#include "operator/D_psi_BSM.h"
#include "operator/D_psi_BSM2b.h"
#include "operator/D_psi_BSM2f.h"
#include "operator/D_psi_BSM2m.h"
#include "operator/Dov_psi.h"
#include "operator/tm_operators_nd.h"
#include "operator/Hopping_Matrix.h"
#include "invert_eo.h"
#include "invert_doublet_eo.h"
#include "invert_overlap.h"
#include "invert_clover_eo.h"
#include "init/init_scalar_field.h"
#include "init/init_bsm_2hop_lookup.h"
#include "boundary.h"
#include "start.h"
#include "solver/solver.h"
#include <io/params.h>
#include <io/gauge.h>
#include <io/spinor.h>
#include <io/utils.h>
#include "test/overlaptests.h"
#include "solver/index_jd.h"
#include "operator/clovertm_operators.h"
#include "operator/clover_leaf.h"
#include "operator.h"
#include "gettime.h"

#define DAGGER 1
#define NO_DAGG 0 

#define GAMMA_UP 1
#define GAMMA_DN -1
#define NO_GAMMA 0

#define WITH_SCALAR 1
#define NO_SCALAR 0

#define TYPE_A 1
#define TYPE_B 0

#define TYPE_1 1
#define TYPE_2 0
#define TYPE_3 2
#define TYPE_4 3

#define TYPE_I 1
#define TYPE_II 0
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
extern int nstore;
int check_geometry();
static void set_default_filenames(char ** input_filename, char ** filename);
static void process_args(int argc, char *argv[], char ** input_filename, char ** filename);
/* indexing of propfields;
   
   propagator for  (dagger or nondagger source)
              for  flavor component f
              for  color  component c    
              for  spinor component s
   is the following bispinor array of size VOLUME(PLUSRAND)

   propfields[12*s + 4*c + 2*f + dagg ? 1: 0]  
     
 */
/**************************
Multiplication with the backward propagator

S == matrix element of D^-1 between the following states

S( ytilde , x+-dir )       psi   x
   flavor2, flavor1    x         flavor1
   spinor2, spinor1              spinor1
   color 2, color 1              color1

=
Stilde* (x+-dir , ytilde)      psi   x
         flavor1, flavor2  x         flavor1
         spinor1, spinor2            spinor1  
         color 1, color 2            color1
where Stilde is the matrix element of D^dagger^-1 between 
the correspondig states

**************************/
_Complex double bispinor_scalar_product ( bispinor *s1, bispinor *s2 ){
   _Complex double res=0.0;
   res   =s2->sp_up.s0.c0 * conj(s1->sp_up.s0.c0) + s2->sp_up.s0.c1 * conj(s1->sp_up.s0.c1) + s2->sp_up.s0.c2 * conj(s1->sp_up.s0.c2) +
          s2->sp_up.s1.c0 * conj(s1->sp_up.s1.c0) + s2->sp_up.s1.c1 * conj(s1->sp_up.s1.c1) + s2->sp_up.s1.c2 * conj(s1->sp_up.s1.c2) +
          s2->sp_up.s2.c0 * conj(s1->sp_up.s2.c0) + s2->sp_up.s2.c1 * conj(s1->sp_up.s2.c1) + s2->sp_up.s2.c2 * conj(s1->sp_up.s2.c2) +
          s2->sp_up.s3.c0 * conj(s1->sp_up.s3.c0) + s2->sp_up.s3.c1 * conj(s1->sp_up.s3.c1) + s2->sp_up.s3.c2 * conj(s1->sp_up.s3.c2) +
          s2->sp_dn.s0.c0 * conj(s1->sp_dn.s0.c0) + s2->sp_dn.s0.c1 * conj(s1->sp_dn.s0.c1) + s2->sp_dn.s0.c2 * conj(s1->sp_dn.s0.c2) +
          s2->sp_dn.s1.c0 * conj(s1->sp_dn.s1.c0) + s2->sp_dn.s1.c1 * conj(s1->sp_dn.s1.c1) + s2->sp_dn.s1.c2 * conj(s1->sp_dn.s1.c2) +
          s2->sp_dn.s2.c0 * conj(s1->sp_dn.s2.c0) + s2->sp_dn.s2.c1 * conj(s1->sp_dn.s2.c1) + s2->sp_dn.s2.c2 * conj(s1->sp_dn.s2.c2) +
          s2->sp_dn.s3.c0 * conj(s1->sp_dn.s3.c0) + s2->sp_dn.s3.c1 * conj(s1->sp_dn.s3.c1) + s2->sp_dn.s3.c2 * conj(s1->sp_dn.s3.c2);
   return res;
}
void multiply_backward_propagator( bispinor *dest, bispinor **propagator, bispinor *source, int idx, int dir){
   int propcoord;
   bispinor source_copy;
   if (dir == NODIR){
      propcoord=idx;
   }
   else if (dir == TUP){
      propcoord=g_iup[idx][TUP];
   }
   else if (dir == TDOWN){
      propcoord=g_idn[idx][TUP];
   }
   _spinor_assign( source_copy.sp_dn, source->sp_dn);
   _spinor_assign( source_copy.sp_up, source->sp_up);

   dest->sp_up.s0.c0= bispinor_scalar_product ( &propagator[ 1][propcoord], &source_copy );
   dest->sp_up.s0.c1= bispinor_scalar_product ( &propagator[ 5][propcoord], &source_copy );
   dest->sp_up.s0.c2= bispinor_scalar_product ( &propagator[ 9][propcoord], &source_copy );

   dest->sp_up.s1.c0= bispinor_scalar_product ( &propagator[13][propcoord], &source_copy );
   dest->sp_up.s1.c1= bispinor_scalar_product ( &propagator[17][propcoord], &source_copy );
   dest->sp_up.s1.c2= bispinor_scalar_product ( &propagator[21][propcoord], &source_copy );

   dest->sp_up.s2.c0= bispinor_scalar_product ( &propagator[25][propcoord], &source_copy );
   dest->sp_up.s2.c1= bispinor_scalar_product ( &propagator[29][propcoord], &source_copy );
   dest->sp_up.s2.c2= bispinor_scalar_product ( &propagator[33][propcoord], &source_copy );

   dest->sp_up.s3.c0= bispinor_scalar_product ( &propagator[37][propcoord], &source_copy );
   dest->sp_up.s3.c1= bispinor_scalar_product ( &propagator[41][propcoord], &source_copy );
   dest->sp_up.s3.c2= bispinor_scalar_product ( &propagator[45][propcoord], &source_copy );

   dest->sp_dn.s0.c0= bispinor_scalar_product ( &propagator[ 3][propcoord], &source_copy );
   dest->sp_dn.s0.c1= bispinor_scalar_product ( &propagator[ 7][propcoord], &source_copy );
   dest->sp_dn.s0.c2= bispinor_scalar_product ( &propagator[11][propcoord], &source_copy );

   dest->sp_dn.s1.c0= bispinor_scalar_product ( &propagator[15][propcoord], &source_copy );
   dest->sp_dn.s1.c1= bispinor_scalar_product ( &propagator[19][propcoord], &source_copy );
   dest->sp_dn.s1.c2= bispinor_scalar_product ( &propagator[23][propcoord], &source_copy );

   dest->sp_dn.s2.c0= bispinor_scalar_product ( &propagator[27][propcoord], &source_copy );
   dest->sp_dn.s2.c1= bispinor_scalar_product ( &propagator[31][propcoord], &source_copy );
   dest->sp_dn.s2.c2= bispinor_scalar_product ( &propagator[35][propcoord], &source_copy );

   dest->sp_dn.s3.c0= bispinor_scalar_product ( &propagator[39][propcoord], &source_copy );
   dest->sp_dn.s3.c1= bispinor_scalar_product ( &propagator[43][propcoord], &source_copy );
   dest->sp_dn.s3.c2= bispinor_scalar_product ( &propagator[47][propcoord], &source_copy );
}
//dest used as a source, an output it is overwritten
void taui_scalarfield_flavoronly( _Complex double *dest, int tauindex, int dagger ){
   _Complex double *source_copy;
   _Complex double a11, a12, a21, a22;
   int i;
  
   source_copy=(_Complex double *)malloc(sizeof(_Complex double)*2*T_global);
   for (i=0; i<2*T_global; ++i)
     source_copy[i]=dest[i];
   if (dagger == DAGGER){
     if (tauindex == 0){
       a11=  -1.*g_scalar_field[2][0] - I*g_scalar_field[1][0];
       a12=  +1.*g_scalar_field[0][0] - I*g_scalar_field[3][0];

       a21=  +1.*g_scalar_field[0][0] + I*g_scalar_field[3][0];
       a22=  +1.*g_scalar_field[2][0] - I*g_scalar_field[1][0];
     }
     else  if (tauindex == 1){
       a11=  +1.*g_scalar_field[1][0] - I*g_scalar_field[2][0];
       a12=  -1.*g_scalar_field[3][0] - I*g_scalar_field[0][0];

       a21=  -1.*g_scalar_field[3][0] + I*g_scalar_field[0][0];
       a22=  -1.*g_scalar_field[1][0] - I*g_scalar_field[2][0];

     }
     else  if (tauindex == 2){
       a11=  +1.*g_scalar_field[0][0] - I*g_scalar_field[3][0];
       a12=  +1.*g_scalar_field[2][0] + I*g_scalar_field[1][0];

       a21=  +1.*g_scalar_field[2][0] - I*g_scalar_field[1][0];
       a22=  -1.*g_scalar_field[0][0] - I*g_scalar_field[3][0];
     }
   }
   else if (dagger == NO_DAGG){
     if (tauindex == 0){
      a11=  -1.*g_scalar_field[2][0] + I*g_scalar_field[1][0];
      a12=  +1.*g_scalar_field[0][0] - I*g_scalar_field[3][0];

      a21=  +1.*g_scalar_field[0][0] + I*g_scalar_field[3][0];
      a22=  +1.*g_scalar_field[2][0] + I*g_scalar_field[1][0];
     }
     if (tauindex == 1){
      a11=  +1.*g_scalar_field[1][0] + I*g_scalar_field[2][0];
      a12=  -1.*g_scalar_field[3][0] - I*g_scalar_field[0][0];

      a21=  -1.*g_scalar_field[3][0] + I*g_scalar_field[0][0];
      a22=  -1.*g_scalar_field[1][0] + I*g_scalar_field[2][0];
     }
     if (tauindex == 2){
      a11=  +1.*g_scalar_field[0][0] + I*g_scalar_field[3][0];
      a12=  +1.*g_scalar_field[2][0] + I*g_scalar_field[1][0];

      a21=  +1.*g_scalar_field[2][0] - I*g_scalar_field[1][0];
      a22=  -1.*g_scalar_field[0][0] + I*g_scalar_field[3][0];
     }
   }
   for (i=0; i<T_global; ++i){
     dest[2*i +0]= a11* source_copy[2*i + 0] + a12* source_copy[2*i + 1];
     dest[2*i +1]= a21* source_copy[2*i + 0] + a22* source_copy[2*i + 1];
   }
   free(source_copy);  
}
void taui_spinor( bispinor *dest, bispinor *source, int tauindex){

   su3_vector tmp2;
   bispinor tmp;
   _spinor_assign(tmp.sp_up, source->sp_up);
   _spinor_assign(tmp.sp_dn, source->sp_dn);

   if (tauindex == 0 ){
    _vector_assign(tmp2        , tmp.sp_up.s0);
    _vector_assign(tmp.sp_up.s0, tmp.sp_dn.s0);
    _vector_assign(tmp.sp_dn.s0, tmp2);

    _vector_assign(tmp2        , tmp.sp_up.s1);
    _vector_assign(tmp.sp_up.s1, tmp.sp_dn.s1);
    _vector_assign(tmp.sp_dn.s1, tmp2);
 
    _spinor_assign(dest->sp_up, tmp.sp_up);
    _spinor_assign(dest->sp_dn, tmp.sp_dn);
   }
   else if (tauindex == 1 ){
    _vector_assign(tmp2             ,tmp.sp_up.s0);
    _vector_i_mul( tmp.sp_up.s0, -1 ,tmp.sp_dn.s0);
    _vector_i_mul( tmp.sp_dn.s0, +1, tmp2);

    _vector_assign(tmp2             ,tmp.sp_up.s1);
    _vector_i_mul( tmp.sp_up.s1, -1 ,tmp.sp_dn.s1);
    _vector_i_mul( tmp.sp_dn.s1, +1, tmp2);   
     
    _spinor_assign(dest->sp_up, tmp.sp_up);
    _spinor_assign(dest->sp_dn, tmp.sp_dn);
   }
   else if (tauindex == 2 ){
    _vector_mul(tmp.sp_dn.s0, -1, tmp.sp_dn.s0);
    _vector_mul(tmp.sp_dn.s1, -1, tmp.sp_dn.s1);
     
    _spinor_assign(dest->sp_up, tmp.sp_up);
    _spinor_assign(dest->sp_dn, tmp.sp_dn);
   }
}
void taui_scalarfield_spinor( bispinor *dest, bispinor *source, int gamma5, int tauindex, int idx, int direction, int dagger){
    
  su3_vector tmp2;
  bispinor tmp;
  bispinor tmpbi2;
  spinor tmp1;
  _spinor_assign(tmp.sp_up, source->sp_up);
  _spinor_assign(tmp.sp_dn, source->sp_dn);
  _Complex double a11, a12, a21, a22;

 int scalarcoord;
 if (direction == NODIR)
   scalarcoord=idx;
 else if (direction<4){
   scalarcoord= g_iup[idx][direction];
 }
 else if (direction<8){
   scalarcoord= g_idn[idx][7-direction];
 }
 if (dagger == DAGGER){
  if (tauindex == 0){
   a11=  -1.*g_scalar_field[2][scalarcoord] - I*g_scalar_field[1][scalarcoord];
   a12=  +1.*g_scalar_field[0][scalarcoord] - I*g_scalar_field[3][scalarcoord];

   a21=  +1.*g_scalar_field[0][scalarcoord] + I*g_scalar_field[3][scalarcoord];
   a22=  +1.*g_scalar_field[2][scalarcoord] - I*g_scalar_field[1][scalarcoord];
  }
  else  if (tauindex == 1){
   a11=  +1.*g_scalar_field[1][scalarcoord] - I*g_scalar_field[2][scalarcoord];
   a12=  -1.*g_scalar_field[3][scalarcoord] - I*g_scalar_field[0][scalarcoord];

   a21=  -1.*g_scalar_field[3][scalarcoord] + I*g_scalar_field[0][scalarcoord];
   a22=  -1.*g_scalar_field[1][scalarcoord] - I*g_scalar_field[2][scalarcoord];
  }
  else  if (tauindex == 2){
   a11=  +1.*g_scalar_field[0][scalarcoord] - I*g_scalar_field[3][scalarcoord];
   a12=  +1.*g_scalar_field[2][scalarcoord] + I*g_scalar_field[1][scalarcoord];

   a21=  +1.*g_scalar_field[2][scalarcoord] - I*g_scalar_field[1][scalarcoord];
   a22=  -1.*g_scalar_field[0][scalarcoord] - I*g_scalar_field[3][scalarcoord];
  }
 }
 else if (dagger == NO_DAGG){
  if (tauindex == 0){
   a11=  -1.*g_scalar_field[2][scalarcoord] + I*g_scalar_field[1][scalarcoord];
   a12=  +1.*g_scalar_field[0][scalarcoord] - I*g_scalar_field[3][scalarcoord];

   a21=  +1.*g_scalar_field[0][scalarcoord] + I*g_scalar_field[3][scalarcoord];
   a22=  +1.*g_scalar_field[2][scalarcoord] + I*g_scalar_field[1][scalarcoord];
  }
  if (tauindex == 1){
   a11=  +1.*g_scalar_field[1][scalarcoord] + I*g_scalar_field[2][scalarcoord];
   a12=  -1.*g_scalar_field[3][scalarcoord] - I*g_scalar_field[0][scalarcoord];

   a21=  -1.*g_scalar_field[3][scalarcoord] + I*g_scalar_field[0][scalarcoord];
   a22=  -1.*g_scalar_field[1][scalarcoord] + I*g_scalar_field[2][scalarcoord];
  }
  if (tauindex == 2){
   a11=  +1.*g_scalar_field[0][scalarcoord] + I*g_scalar_field[3][scalarcoord];
   a12=  +1.*g_scalar_field[2][scalarcoord] + I*g_scalar_field[1][scalarcoord];

   a21=  +1.*g_scalar_field[2][scalarcoord] - I*g_scalar_field[1][scalarcoord];
   a22=  -1.*g_scalar_field[0][scalarcoord] + I*g_scalar_field[3][scalarcoord];
  }
 }
 _spinor_null(tmpbi2.sp_up);
 _spinor_null(tmpbi2.sp_dn);
 
 if ( gamma5 == GAMMA_UP){
  _vector_mul_complex(    tmpbi2.sp_up.s0, a11, tmp.sp_up.s0);
  _vector_add_mul_complex(tmpbi2.sp_up.s0, a12, tmp.sp_dn.s0);

  _vector_mul_complex    (tmpbi2.sp_dn.s0, a21, tmp.sp_up.s0);
  _vector_add_mul_complex(tmpbi2.sp_dn.s0, a22, tmp.sp_dn.s0);

  _vector_mul_complex(    tmpbi2.sp_up.s1, a11, tmp.sp_up.s1);
  _vector_add_mul_complex(tmpbi2.sp_up.s1, a12, tmp.sp_dn.s1);

  _vector_mul_complex    (tmpbi2.sp_dn.s1, a21, tmp.sp_up.s1);
  _vector_add_mul_complex(tmpbi2.sp_dn.s1, a22, tmp.sp_dn.s1);
 }
 else if  ( gamma5 == GAMMA_DN ){
  _vector_mul_complex(    tmpbi2.sp_up.s2, a11, tmp.sp_up.s2);
  _vector_add_mul_complex(tmpbi2.sp_up.s2, a12, tmp.sp_dn.s2);

  _vector_mul_complex    (tmpbi2.sp_dn.s2, a21, tmp.sp_up.s2);
  _vector_add_mul_complex(tmpbi2.sp_dn.s2, a22, tmp.sp_dn.s2);

  _vector_mul_complex(    tmpbi2.sp_up.s3, a11, tmp.sp_up.s3);
  _vector_add_mul_complex(tmpbi2.sp_up.s3, a12, tmp.sp_dn.s3);

  _vector_mul_complex    (tmpbi2.sp_dn.s3, a21, tmp.sp_up.s3);
  _vector_add_mul_complex(tmpbi2.sp_dn.s3, a22, tmp.sp_dn.s3);
 }
 else if ( gamma5 == NO_GAMMA ){
  _spinor_mul_complex    (tmpbi2.sp_up,    a11, tmp.sp_up);
  _spinor_add_mul_complex(tmpbi2.sp_up,    a12, tmp.sp_dn);

  _spinor_mul_complex    (tmpbi2.sp_dn,    a21, tmp.sp_up);
  _spinor_add_mul_complex(tmpbi2.sp_dn,    a22, tmp.sp_dn);
 }

 _spinor_assign(dest->sp_up, tmpbi2.sp_up);
 _spinor_assign(dest->sp_dn, tmpbi2.sp_dn);

}
void trace_in_spinor( _Complex double *dest, _Complex double *src, int spinorindex){
   int tind, find;
   for (tind=0; tind<T_global; ++tind)
     for (find=0; find<2; ++find){ 
       dest[2*tind+find]+=src[8*tind+4*find+spinorindex];
     }
}
void trace_in_color(_Complex double *dest, bispinor *src, int colorindex){
   if      ( colorindex == 0 ){
     dest[0]+= src->sp_up.s0.c0;
     dest[1]+= src->sp_up.s1.c0;
     dest[2]+= src->sp_up.s2.c0;
     dest[3]+= src->sp_up.s3.c0;
     dest[4]+= src->sp_dn.s0.c0;
     dest[5]+= src->sp_dn.s1.c0;
     dest[6]+= src->sp_dn.s2.c0;
     dest[7]+= src->sp_dn.s3.c0;

   }
   else if ( colorindex == 1 ){
     dest[0]+= src->sp_up.s0.c1;
     dest[1]+= src->sp_up.s1.c1;
     dest[2]+= src->sp_up.s2.c1;
     dest[3]+= src->sp_up.s3.c1;
     dest[4]+= src->sp_dn.s0.c1;
     dest[5]+= src->sp_dn.s1.c1;
     dest[6]+= src->sp_dn.s2.c1;
     dest[7]+= src->sp_dn.s3.c1;
   }
   else if ( colorindex == 2 ){
     dest[0]+= src->sp_up.s0.c2;
     dest[1]+= src->sp_up.s1.c2;
     dest[2]+= src->sp_up.s2.c2;
     dest[3]+= src->sp_up.s3.c2;
     dest[4]+= src->sp_dn.s0.c2;
     dest[5]+= src->sp_dn.s1.c2;
     dest[6]+= src->sp_dn.s2.c2;
     dest[7]+= src->sp_dn.s3.c2;
   }
}
void trace_in_space(_Complex double *dest, _Complex double *source, int idx){
     int i;
     for (i=0; i<8;++i){
       dest[g_coord[idx][TUP]*8+i]+= source[i];
     }
}
void trace_in_flavor(_Complex double *dest, _Complex double *source, int f1){
     int i;
     for (i=0; i<T_global; ++i){
        dest[i]+= source[2*i+f1];
     }
}

void density_density_1234( bispinor ** propfields, int type_1234 ){
   int ix,i;
   int f1,c1,s1,tauindex;
   int spinorstart, spinorend;
   su3 * restrict upm;
   bispinor running;

   _Complex double *colortrace;
   _Complex double *spacetrace;
   _Complex double *spinortrace;
   _Complex double *flavortrace;
   _Complex double *paulitrace;

   colortrace=(_Complex double *)malloc(sizeof(_Complex double) *8);
   spacetrace=(_Complex double *)malloc(sizeof(_Complex double) *8*T_global);
   spinortrace=(_Complex double *)malloc(sizeof(_Complex double)*2*T_global);
   flavortrace=(_Complex double *)malloc(sizeof(_Complex double)*T_global);
   paulitrace= (_Complex double *)malloc(sizeof(_Complex double)*T_global);

   if ( (type_1234 == TYPE_1) || (type_1234 == TYPE_3 )) {
     spinorstart=0;
     spinorend  =2;
   }
   else if ( (type_1234 == TYPE_2) || (type_1234 == TYPE_4) ){
     spinorstart=2;
     spinorend  =4;
   }

//Trace over the Pauli matrices
   for (i=0; i<T_global; ++i)
      paulitrace[i]=0.;

   for (tauindex=0; tauindex<3; ++tauindex){

//Trace over up and down flavors
      for (i=0; i<T_global; ++i)
         flavortrace[i]=0.;

      for (f1=0; f1<2; ++f1){

//Trace over the spinor indices you have to trace only over those two spinor 
//component that appear in the final spinor
         for (i=0; i<2*T_global; ++i)
            spinortrace[i]=0.;

         for (s1= spinorstart; s1<spinorend; ++s1){

//Trace over the spatial indices
            for (i=0; i<8*T_global; ++i)
               spacetrace[i]=0.;
  
          for (ix = 0; ix< VOLUME; ++ix){

//Trace over the color indices for each sites

               for (i=0; i<8; ++i)
                  colortrace[i]=0.;
               for (c1=0; c1<3; ++c1){
/*   
       TYPE  1 OR  2            (1-g5)/2*S(x  ,ytilde) fixed indices (c1, s1, f1)
       TYPE  3 OR  4            (1+g5)/2*S(x  ,ytilde) running indices bispinor
*/
                  upm = &g_gauge_field[g_idn[ix][TUP]][TUP];

//for the up quark
                  if ( (type_1234 == TYPE_1) || (type_1234 == TYPE_2) ){
                    _vector_null( running.sp_up.s0 );
                    _vector_null( running.sp_up.s1 );
                    _vector_assign( running.sp_up.s2, propfields[12*s1+4*c1+2*f1][ix].sp_up.s2 );
                    _vector_assign( running.sp_up.s3, propfields[12*s1+4*c1+2*f1][ix].sp_up.s3 );
                    _vector_null( running.sp_dn.s0 );
                    _vector_null( running.sp_dn.s1 );
                    _vector_assign( running.sp_dn.s2, propfields[12*s1+4*c1+2*f1][ix].sp_dn.s2 );
                    _vector_assign( running.sp_dn.s3, propfields[12*s1+4*c1+2*f1][ix].sp_dn.s3 );
                  }
                  if ((type_1234 == TYPE_3) || (type_1234 == TYPE_4)){
                    _vector_null( running.sp_up.s2 );
                    _vector_null( running.sp_up.s3 );
                    _vector_assign( running.sp_up.s0, propfields[12*s1+4*c1+2*f1][ix].sp_up.s0 );
                    _vector_assign( running.sp_up.s1, propfields[12*s1+4*c1+2*f1][ix].sp_up.s1 );
                    _vector_null( running.sp_dn.s2 );
                    _vector_null( running.sp_dn.s3 );
                    _vector_assign( running.sp_dn.s0, propfields[12*s1+4*c1+2*f1][ix].sp_dn.s0 );
                    _vector_assign( running.sp_dn.s1, propfields[12*s1+4*c1+2*f1][ix].sp_dn.s1 );
                  }

/*   
       TYPE  1 OR  2     phi^dagger(x)*tau_i*  (1-g5)/2*S(x  ,ytilde)
       TYPE  3 OR  4     tau_i*phi(x)          (1+g5)/2*S(x  ,ytilde)
*/
                  if (( type_1234 == TYPE_1) || ( type_1234 == TYPE_2) ){
                    taui_scalarfield_spinor( &running, &running, GAMMA_DN, tauindex, ix, NODIR, DAGGER );
                  }
                  else if ( (type_1234 == TYPE_3) || (type_1234 == TYPE_4) ){
                    taui_scalarfield_spinor( &running, &running, GAMMA_UP, tauindex, ix, NODIR, NO_DAGG);
                  }
/*   
       TYPE  1 OR  2     S(ytilde, x)*phi^dagger(x)*tau_i*  (1-g5)/2*S(x  ,ytilde)
       TYPE  3 OR  4     S(ytilde, x)*tau_i*phi(x)          (1+g5)/2*S(x  ,ytilde)
*/
                  multiply_backward_propagator(&running, propfields, &running, ix, NODIR );

                  //delta( color component of bispinor running, c1) for all spinor and flavor indices
                  trace_in_color(colortrace,&running,c1);

               }  //End of trace color
               //sum over all lattice sites the result of the color trace
               trace_in_space(spacetrace,colortrace,ix);

            } //End of trace space

//Gather the results from all nodes to complete the trace in space
            for (i=0; i<8*T_global; ++i){
               _Complex double tmp;
               MPI_Allreduce(&spacetrace[i], &tmp, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
               spacetrace[i]= tmp;
            }
            // delta (spinor components of spacetrace, s1) for all time slices and flavor indices
            trace_in_spinor(spinortrace, spacetrace, s1);

         }//End of trace in spinor space
/*   
       TYPE  1      tau_i*phi(ytilde)*       (1+gamma5)/2*S(ytilde, x)*phi^dagger(x)*tau_i*  (1-g5)/2*S(x  ,ytilde)
       TYPE  2      phi(ytilde)^dagger*tau_i*(1-gamma5)/2*S(ytilde, x)*phi^dagger(x)*tau_i*  (1-g5)/2*S(x  ,ytilde)
       TYPE  3      tau_i*phi(ytilde)*       (1+gamma5)/2*S(ytilde, x)*tau_i*phi(x)          (1+g5)/2*S(x  ,ytilde)
       TYPE  4      phi(ytilde)^dagger*tau_i*(1-gamma5)/2*S(ytilde, x)*tau_i*phi(x)          (1+g5)/2*S(x  ,ytilde)
*/
         if ( (type_1234 == TYPE_1) || (type_1234 == TYPE_3) ){
           taui_scalarfield_flavoronly( spinortrace, tauindex, NO_DAGG );
         }
         else if ( ( type_1234 == TYPE_4) || ( type_1234 == TYPE_2 )){
           taui_scalarfield_flavoronly( spinortrace, tauindex, DAGGER  );
         }
         //delta(flavor component in spinortrace, f1) for all time slices
         trace_in_flavor( flavortrace, spinortrace, f1 );
      }  //End of trace in flavor space
      //sum for all Pauli matrices
      for (i=0;i<T_global; ++i)
         paulitrace[i]+=flavortrace[i];
   } //End of trace for Pauli matrices

   if (g_cart_id == 0){printf("Density Density correlator type (%s) results\n", type_1234 == TYPE_1 ? "1" : type_1234 == TYPE_2 ? "2" : type_1234 == TYPE_3 ? "3" : "4");}
   for (i=0; i<T_global; ++i){
      if (g_cart_id == 0){
        printf("%3d %10.10e %10.10e\n", i, creal(paulitrace[i]),cimag(paulitrace[i]));
      }
   }
   free(flavortrace);
   free(paulitrace);
   free(spacetrace);
   free(spinortrace);
   free(colortrace);

}



void naivedirac_current_density_12ab( bispinor ** propfields, int type_12, int type_ab ){
   int ix,i;
   int f1,c1,s1,tauindex;
   int spinorstart, spinorend;
   su3 * restrict upm;
   bispinor running;
   int count;
   MPI_Status  statuses[8];
   MPI_Request *request;
   request=( MPI_Request *) malloc(sizeof(MPI_Request)*8);

   _Complex double *colortrace;
   _Complex double *spacetrace;
   _Complex double *spinortrace;
   _Complex double *flavortrace;
   _Complex double *paulitrace;

   colortrace=(_Complex double *)malloc(sizeof(_Complex double) *8);
   spacetrace=(_Complex double *)malloc(sizeof(_Complex double) *8*T_global);
   spinortrace=(_Complex double *)malloc(sizeof(_Complex double)*2*T_global);
   flavortrace=(_Complex double *)malloc(sizeof(_Complex double)*T_global);
   paulitrace= (_Complex double *)malloc(sizeof(_Complex double)*T_global);

   if ( type_ab == TYPE_A ) {
     spinorstart=0;
     spinorend  =2;
   }
   else if ( type_ab == TYPE_B ){
     spinorstart=2;
     spinorend  =4;
   }
//Doing the neccessary communication
   for (s1=spinorstart; s1<spinorend; ++s1)
      for (c1=0; c1<3; ++c1)
         for (f1=0; f1<2; ++f1){
            count=0;
            generic_exchange_direction_nonblocking( propfields[12*s1 + 4*c1 + 2*f1 + 0], sizeof(bispinor), TDOWN, request, &count );
            MPI_Waitall( count, request, statuses);
            count=0; 
            generic_exchange_direction_nonblocking( propfields[12*s1 + 4*c1 + 2*f1 + 1], sizeof(bispinor), TDOWN, request, &count);
            MPI_Waitall( count, request, statuses);
         }
   free(request);
   
//Trace over the Pauli matrices
   for (i=0; i<T_global; ++i)
      paulitrace[i]=0.;

   for (tauindex=0; tauindex<3; ++tauindex){
      for (i=0; i<T_global; ++i)
         flavortrace[i]=0.;

      for (f1=0; f1<2; ++f1){

//Trace over the spinor indices
         for (i=0; i<2*T_global; ++i)
            spinortrace[i]=0.;

         for (s1= spinorstart; s1<spinorend; ++s1){

//Trace over the spatial indices
            for (i=0; i<8*T_global; ++i)
               spacetrace[i]=0.;
            for (ix = 0; ix< VOLUME; ++ix){

//Trace over the color indices for each sites

               for (i=0; i<8; ++i) 
                  colortrace[i]=0.;
               for (c1=0; c1<3; ++c1){    
/*   
       TYPE  IA OR  IB     U0(x-0)*       (1-g5)/2*S(x  ,ytilde) fixed indices (c1, s1, f1)
       TYPE IIA OR IIB     U0^dagger(x-0)*(1-g5)/2*S(x-0,ytilde) running indices bispinor
*/
                  upm = &g_gauge_field[g_idn[ix][TUP]][TUP];

//for the up quark
                  _vector_null( running.sp_up.s0 ); 
                  _vector_null( running.sp_up.s1 ); 
             
                  if  ( type_12 == TYPE_I ){
                    _su3_multiply( running.sp_up.s2, (*upm), propfields[12*s1+4*c1+2*f1][ix].sp_up.s2 ); 
                    _su3_multiply( running.sp_up.s3, (*upm), propfields[12*s1+4*c1+2*f1][ix].sp_up.s3 ); 
                  }
                  else if ( type_12 == TYPE_II ){
                    _su3_inverse_multiply( running.sp_up.s2, (*upm), propfields[12*s1+4*c1+2*f1][g_idn[ix][TUP]].sp_up.s2 ); 
                    _su3_inverse_multiply( running.sp_up.s3, (*upm), propfields[12*s1+4*c1+2*f1][g_idn[ix][TUP]].sp_up.s3 ); 
                  }

//for the up quark
                  _vector_null( running.sp_dn.s0 ); 
                  _vector_null( running.sp_dn.s1 ); 
                  if  ( type_12 == TYPE_I ){
                    _su3_multiply( running.sp_dn.s2, (*upm), propfields[12*s1+4*c1+2*f1][ix].sp_dn.s2 ); 
                    _su3_multiply( running.sp_dn.s3, (*upm), propfields[12*s1+4*c1+2*f1][ix].sp_dn.s3 ); 
                  }
                  else if ( type_12 == TYPE_II ){
                    _su3_inverse_multiply( running.sp_dn.s2, (*upm), propfields[12*s1+4*c1+2*f1][g_idn[ix][TUP]].sp_dn.s2 );
                    _su3_inverse_multiply( running.sp_dn.s3, (*upm), propfields[12*s1+4*c1+2*f1][g_idn[ix][TUP]].sp_dn.s3 );
                  }

/*   
       TYPE  IA OR  IB     gamma0*U0(x-0)*       (1-g5)/2*S(x  ,ytilde)
       TYPE IIA OR IIB     gamma0*U0^dagger(x-0)*(1-g5)/2*S(x-0,ytilde)
*/
                  _vector_add_assign(running.sp_up.s0, running.sp_up.s2);
                  _vector_add_assign(running.sp_up.s1, running.sp_up.s3);
                  _vector_null(running.sp_up.s2);
                  _vector_null(running.sp_up.s3);

                  _vector_add_assign(running.sp_dn.s0, running.sp_dn.s2);
                  _vector_add_assign(running.sp_dn.s1, running.sp_dn.s3);
                  _vector_null(running.sp_dn.s2);
                  _vector_null(running.sp_dn.s3);

/*   
       TYPE  IA OR  IB     tau_i*gamma0*U0(x-0)*       (1-g5)/2*S(x  ,ytilde)
       TYPE IIA OR IIB     tau_i*gamma0*U0^dagger(x-0)*(1-g5)/2*S(x-0,ytilde)
*/
                  taui_spinor( &running, &running, tauindex  );

/*   
       TYPE  IA OR  IB     S(ytilde, x-0)* tau_i*gamma0*U0(x-0)*       (1-g5)/2*S(x  ,ytilde)
       TYPE IIA OR IIB     S(ytilde, x  )* tau_i*gamma0*U0^dagger(x-0)*(1-g5)/2*S(x-0,ytilde)
*/
                  if ( type_12 == TYPE_I ){ 
                    multiply_backward_propagator(&running, propfields, &running, ix, TDOWN);
                  }
                  else if ( type_12 == TYPE_II ){
                    multiply_backward_propagator(&running, propfields, &running, ix,NODIR);
                  }
                  //delta( color component of bispinor running, c1) for all spinor and flavor indices
                  trace_in_color(colortrace,&running,c1);

               }  //End of trace color
               //sum over all lattice sites the result of the color trace
               trace_in_space(spacetrace,colortrace,ix);

            } //End of trace space

//Gather the results from all nodes to complete the trace in space
            for (i=0; i<8*T_global; ++i){
               _Complex double tmp;
               MPI_Allreduce(&spacetrace[i], &tmp, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
               spacetrace[i]= tmp;
            }
            // delta (spinor components of spacetrace, s1) for all time slices and flavor components
            trace_in_spinor(spinortrace, spacetrace, s1);

         }//End of trace in spinor space
   
/*   
       TYPE  IA tau_i*phi(ytilde)        *  (1+gamma5)/2  *   S(ytilde, x-0)*   tau_i*gamma0*U0(x-0)*       (1-g5)/2*   S(x  ,ytilde)
       TYPE  IB phi^dagger(ytilde)*tau_i *  (1-gamma5)/2  *   S(ytilde, x-0)*   tau_i*gamma0*U0(x-0)*       (1-g5)/2*   S(x  ,ytilde)

       TYPE IIA tau_i*phi(ytilde)        *  (1+gamma5)/2  *   S(ytilde, x  )*   tau_i*gamma0*U0^dagger(x-0)*(1-g5)/2*   S(x-0,ytilde)
       TYPE IIB phi^dagger(ytilde)*tau_i *  (1-gamma5)/2  *   S(ytilde, x  )*   tau_i*gamma0*U0^dagger(x-0)*(1-g5)/2*   S(x-0,ytilde)

*/
         if ( type_ab == TYPE_A ){
           taui_scalarfield_flavoronly( spinortrace, tauindex, NO_DAGG );
         }
         else if ( type_ab == TYPE_B){
           taui_scalarfield_flavoronly( spinortrace, tauindex, DAGGER  );
         }
         //delta(flavor component in spinortrace, f1) for all time slices 
         trace_in_flavor( flavortrace, spinortrace, f1 );
      } //End of trace in flavor space
      //sum for all Pauli matrices
      for (i=0;i<T_global; ++i)
         paulitrace[i]+=flavortrace[i];
   } //End of trace for Pauli matrices
   
   if (g_cart_id == 0){printf("NaiveDirac Current Density correlator type (%s %s) results\n", type_12 == TYPE_I ? "I" : "II",type_ab == TYPE_A ? 'a' :'b');}
   for (i=0; i<T_global; ++i){
      if (g_cart_id == 0){
        printf("%3d %10.10e %10.10e\n", i, creal(paulitrace[i]),cimag(paulitrace[i]));
      }
   }
   free(flavortrace);
   free(paulitrace);
   free(spacetrace);
   free(spinortrace);
   free(colortrace);

}
void wilsonterm_current_density_312ab( bispinor ** propfields, int type_12, int type_ab ){
   int ix,i;
   int f1,c1,s1,tauindex;
   int spinorstart, spinorend;
   su3 * restrict upm;
   su3_vector tmpvec;
   bispinor running;
   int count;
   MPI_Status  statuses[8];
   MPI_Request *request;
   request=( MPI_Request *) malloc(sizeof(MPI_Request)*8);
   _Complex double *colortrace;
   _Complex double *spacetrace;
   _Complex double *spinortrace;
   _Complex double *flavortrace;
   _Complex double *paulitrace;

   colortrace= (_Complex double *)malloc(sizeof(_Complex double) *8);
   spacetrace= (_Complex double *)malloc(sizeof(_Complex double) *8*T_global);
   spinortrace=(_Complex double *)malloc(sizeof(_Complex double)*2*T_global);
   flavortrace=(_Complex double *)malloc(sizeof(_Complex double)*T_global);
   paulitrace= (_Complex double *)malloc(sizeof(_Complex double)*T_global);

   if ( type_ab == TYPE_A ) {
      spinorstart=0;
      spinorend  =2;
   }
   else if ( type_ab == TYPE_B ){
      spinorstart=2;
      spinorend  =4;
   }
// Doing the neccessary communication
   for (s1=spinorstart; s1<spinorend; ++s1)
      for (c1=0; c1<3; ++c1)
         for (f1=0; f1<2; ++f1){
            count=0;
            generic_exchange_direction_nonblocking( propfields[12*s1 + 4*c1 + 2*f1 + 0], sizeof(bispinor), TUP   , request, &count );
            MPI_Waitall( count, request, statuses);
            count=0;
            generic_exchange_direction_nonblocking( propfields[12*s1 + 4*c1 + 2*f1 + 0], sizeof(bispinor), TDOWN , request, &count );
            MPI_Waitall( count, request, statuses);
            count=0;
            generic_exchange_direction_nonblocking( propfields[12*s1 + 4*c1 + 2*f1 + 1], sizeof(bispinor), TDOWN , request, &count);
            MPI_Waitall( count, request, statuses);
         }
   free(request);

//Trace over the Pauli matrices
   for (i=0; i<T_global; ++i){
      paulitrace[i]=0.;
   }
   for (tauindex= 0; tauindex <3; ++tauindex){
//Trace over flavour degrees of freedom
      for (i=0; i<T_global; ++i)
         flavortrace[i]=0.;

      for (f1=0; f1<2; ++f1){

//Trace over spinor indices
         for (i=0; i<8*T_global; ++i){
            spinortrace[i]=0.;
         }

         for (s1=spinorstart; s1<spinorend; ++s1){

//Trace over spatial indices
            for (i=0; i<8*T_global; ++i){
               spacetrace[i]=0.;
            }
            for (ix=0; ix<VOLUME; ++ix){

//Trace over the color indices for each sites
               for (i=0; i<8; ++i)
                  colortrace[i]=0.;
               for (c1=0; c1<3; ++c1){
/*   
       TYPE III.1.a OR  III.1.b     U0(x-0)*U0(x)* (1+gamma5)/2 *  S(x+0,ytilde)
       TYPE III.2.a OR  III.2.b                    (1+gamma5)/2 *  S(x-0,ytilde)
*/
                  _vector_null(running.sp_up.s2);
                  _vector_null(running.sp_up.s3);
                  _vector_null(running.sp_dn.s2);
                  _vector_null(running.sp_dn.s3);

                  if ( type_12 == TYPE_1){
                    upm = &g_gauge_field[ix][TUP];
//for the up quark
                    _su3_multiply(running.sp_up.s0, (*upm), propfields[12*s1 + 4*c1 + 2*f1][g_iup[ix][TUP]].sp_up.s0);
                    _su3_multiply(running.sp_up.s1, (*upm), propfields[12*s1 + 4*c1 + 2*f1][g_iup[ix][TUP]].sp_up.s1);

//for the down quark
                    _su3_multiply(running.sp_dn.s0, (*upm), propfields[12*s1 + 4*c1 + 2*f1][g_iup[ix][TUP]].sp_dn.s0);
                    _su3_multiply(running.sp_dn.s1, (*upm), propfields[12*s1 + 4*c1 + 2*f1][g_iup[ix][TUP]].sp_dn.s1);

                    upm = &g_gauge_field[g_idn[ix][TUP]][TUP];

//for the up quark

                    _su3_multiply(tmpvec, (*upm), running.sp_up.s0);
                    _vector_assign(  running.sp_up.s0, tmpvec);
                    _su3_multiply(tmpvec, (*upm), running.sp_up.s1);
                    _vector_assign(  running.sp_up.s1, tmpvec);

//for the down quark

                    _su3_multiply(tmpvec, (*upm), running.sp_dn.s0);
                    _vector_assign(  running.sp_dn.s0, tmpvec);
                    _su3_multiply(tmpvec, (*upm), running.sp_dn.s1);
                    _vector_assign(  running.sp_dn.s1, tmpvec);
                  }
                  else if ( type_12 == TYPE_2){
                    _vector_assign( running.sp_up.s0, propfields[12*s1 + 4*c1 + 2*f1][g_idn[ix][TUP]].sp_up.s0 );
                    _vector_assign( running.sp_up.s1, propfields[12*s1 + 4*c1 + 2*f1][g_idn[ix][TUP]].sp_up.s1 );
                    _vector_assign( running.sp_dn.s0, propfields[12*s1 + 4*c1 + 2*f1][g_idn[ix][TUP]].sp_dn.s0 );
                    _vector_assign( running.sp_dn.s1, propfields[12*s1 + 4*c1 + 2*f1][g_idn[ix][TUP]].sp_dn.s1 );
                  }
/*   
       TYPE III.1.a OR  III.1.b     tau_i*phi(x)*U0(x-0)*U0(x)* (1+gamma5)/2 *  S(x+0,ytilde)
       TYPE III.2.a OR  III.2.b     tau_i*phi(x)*               (1+gamma5)/2 *  S(x-0,ytilde)
*/
                  taui_scalarfield_spinor( &running, &running, GAMMA_UP, tauindex, ix, NODIR, NO_DAGG);

/*   
       TYPE III.1.a OR  III.1.b     S(ytilde, x-0)*tau_i*phi(x)*U0(x-0)*U0(x)* (1+gamma5)/2 *  S(x+0,ytilde)
       TYPE III.2.a OR  III.2.b     S(ytilde, x-0)*tau_i*phi(x)*               (1+gamma5)/2 *  S(x-0,ytilde)
*/
                  multiply_backward_propagator(&running, propfields, &running, ix,-1);
                  //delta( color component of bispinor running, c1) for all spinor and flavor indices                  
                  trace_in_color(colortrace, &running, c1 );
               } //End of trace color
               //sum over all lattice sites the result of the color trace
               trace_in_space( spacetrace, colortrace, ix);
            }  //End of trace in space

//Gather the results from all nodes to complete the trace in space
            for (i=0; i<8*T_global; ++i){
               _Complex double tmp;
               MPI_Allreduce(&spacetrace[i], &tmp, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
               spacetrace[i]= tmp;
            }
            // delta (spinor components of spacetrace, s1) for all time slices and flavor indices 
            trace_in_spinor(spinortrace, spacetrace, s1);

         } //End of trace in spinor space

/*   
       TYPE III.1.a                 tau_i*phi(ytilde)*       (1+gamma5)/2* S(ytilde, x-0)*tau_i*phi(x)*U0(x-0)*U0(x)* (1+gamma5)/2 *  S(x+0,ytilde)
       TYPE III.1.b                 phi^dagger(ytilde)*tau_i*(1-gamma5)/2* S(ytilde, x-0)*tau_i*phi(x)*U0(x-0)*U0(x)* (1+gamma5)/2 *  S(x+0,ytilde)

       TYPE III.2.a                 tau_i*phi(ytilde)*       (1+gamma5)/2* S(ytilde, x-0)*tau_i*phi(x)*               (1+gamma5)/2 *  S(x-0,ytilde)
       TYPE III.2.b                 phi^dagger(ytilde)*tau_i*(1-gamma5)/2* S(ytilde, x-0)*tau_i*phi(x)*               (1+gamma5)/2 *  S(x-0,ytilde)

*/
         if ( type_ab == TYPE_A ){
           taui_scalarfield_flavoronly( spinortrace, tauindex, NO_DAGG );
         }
         else if ( type_ab == TYPE_B){
           taui_scalarfield_flavoronly( spinortrace, tauindex, DAGGER  );
         }
         //delta(flavor component in spinortrace, f1) for all time slices
         trace_in_flavor( flavortrace, spinortrace, f1 );

      } //End of trace in flavor space
      //sum for all Pauli matrices
      for (i=0;i<T_global; ++i)
         paulitrace[i]+=flavortrace[i];
   } //End of trace for Pauli matrices

 
   if (g_cart_id == 0){printf("Wilson term Dirac Current Density correlator typeIII results= %10.10e %10.10e\n", type_12 == TYPE_1 ? '1' : '2',type_ab == TYPE_A ? 'a' :'b');}
   for (i=0; i<T_global; ++i){
      if (g_cart_id == 0){
        printf("%3d %10.10e %10.10e\n", i, creal(paulitrace[i]), cimag(paulitrace[i]));
      }
   }
   free(flavortrace);
   free(paulitrace);
   free(spacetrace);
   free(spinortrace);
   free(colortrace);
}
void wilsonterm_current_density_412ab( bispinor ** propfields, int type_12, int type_ab ){
   int ix,i;
   int f1,c1,s1,tauindex;
   int spinorstart, spinorend;
   bispinor **propsecneighbour;
   bispinor **tmpbisp2d;
   su3 * restrict upm;
   bispinor running;
   su3_vector tmpvec;
   int count;
   MPI_Status  statuses[8];
   MPI_Request *request;
   _Complex double *colortrace;
   _Complex double *spacetrace;
   _Complex double *spinortrace;
   _Complex double *flavortrace;
   _Complex double *paulitrace;

   colortrace= (_Complex double *)malloc(sizeof(_Complex double) *8);
   spacetrace= (_Complex double *)malloc(sizeof(_Complex double) *8*T_global);
   spinortrace=(_Complex double *)malloc(sizeof(_Complex double)*2*T_global);
   flavortrace=(_Complex double *)malloc(sizeof(_Complex double)*T_global);
   paulitrace= (_Complex double *)malloc(sizeof(_Complex double)*T_global);

   request=( MPI_Request *) malloc(sizeof(MPI_Request)*8);
   if ( type_ab == TYPE_A ) {
        spinorstart=0;
        spinorend  =2;
   }
   else if ( type_ab == TYPE_B ){
        spinorstart=2;
        spinorend  =4;
   }

   if (type_12 == TYPE_2){
/**********************************
Creating U^dagger(x-0)*U^dagger(x-2*0)*S(x-2*0,ytilde) in three steps:
1; Creating U^dagger(x)*S(x,ytilde)
2; Creating U^dagger(x+0)Ü0^dagger(x)*S(x, ytilde)
3; Gathering two times in direction TDOWN
***********************************/
      tmpbisp2d= (bispinor **)malloc(sizeof(bispinor *)*24);
      propsecneighbour=(bispinor **)malloc(sizeof(bispinor *)*24);
      for (i=0; i<24; ++i){
        propsecneighbour[i]=(bispinor *)malloc(sizeof(bispinor)*VOLUMEPLUSRAND);
        tmpbisp2d[i]=(bispinor *)malloc(sizeof(bispinor)*VOLUMEPLUSRAND); 
      }
      for (i=0; i<24; ++i)
        for (ix=0; ix<VOLUME; ++ix)
          _bispinor_null(tmpbisp2d[i][ix]);

      for (ix = 0; ix< VOLUME; ++ix)
        for (s1=spinorstart;s1<spinorend; ++s1)
          for (c1=0; c1<3; ++c1)
            for (f1=0; f1<2; ++f1){

              upm = &g_gauge_field[ix][TUP];

              _vector_null(tmpbisp2d[12*f1 + 3*s1 + c1][ix].sp_up.s2);
              _vector_null(tmpbisp2d[12*f1 + 3*s1 + c1][ix].sp_up.s3);
              _su3_inverse_multiply(tmpbisp2d[12*f1+3*s1+c1][ix].sp_up.s0, (*upm), propfields[12*s1 + 4*c1 + 2*f1][ix].sp_up.s0);
              _su3_inverse_multiply(tmpbisp2d[12*f1+3*s1+c1][ix].sp_up.s1, (*upm), propfields[12*s1 + 4*c1 + 2*f1][ix].sp_up.s1);

              _vector_null(tmpbisp2d[12*f1 + 3*s1 + c1][ix].sp_dn.s2);
              _vector_null(tmpbisp2d[12*f1 + 3*s1 + c1][ix].sp_dn.s3);
              _su3_inverse_multiply(tmpbisp2d[12*f1+3*s1+c1][ix].sp_dn.s0, (*upm), propfields[12*s1 + 4*c1 + 2*f1][ix].sp_dn.s0);
              _su3_inverse_multiply(tmpbisp2d[12*f1+3*s1+c1][ix].sp_dn.s1, (*upm), propfields[12*s1 + 4*c1 + 2*f1][ix].sp_dn.s1);

               upm = &g_gauge_field[g_iup[ix][TUP]][TUP];
               
               _vector_null( tmpvec );
               _su3_inverse_multiply(tmpvec, (*upm), tmpbisp2d[12*f1+3*s1+c1][ix].sp_up.s0);
               _vector_assign(  tmpbisp2d[12*f1+3*s1+c1][ix].sp_up.s0, tmpvec);
               _vector_null( tmpvec );
               _su3_inverse_multiply(tmpvec, (*upm), tmpbisp2d[12*f1+3*s1+c1][ix].sp_up.s1);
               _vector_assign(  tmpbisp2d[12*f1+3*s1+c1][ix].sp_up.s1, tmpvec);

               _vector_null( tmpvec );
               _su3_inverse_multiply(tmpvec, (*upm), tmpbisp2d[12*f1+3*s1+c1][ix].sp_dn.s0);
               _vector_assign(  tmpbisp2d[12*f1+3*s1+c1][ix].sp_dn.s0, tmpvec);
               _vector_null( tmpvec );
               _su3_inverse_multiply(tmpvec, (*upm), tmpbisp2d[12*f1+3*s1+c1][ix].sp_dn.s1);
               _vector_assign(  tmpbisp2d[12*f1+3*s1+c1][ix].sp_dn.s1, tmpvec);
            }
      for (s1=spinorstart;s1<spinorend; ++s1)
        for (c1=0; c1<3; ++c1)
          for (f1=0; f1<2; ++f1){
            count=0;
            generic_exchange_direction_nonblocking( tmpbisp2d[12*f1+3*s1+c1], sizeof(bispinor), TDOWN, request, &count );
            MPI_Waitall( count, request, statuses);
          }
      for (s1=spinorstart;s1<spinorend; ++s1)
        for (c1=0; c1<3; ++c1)
          for (f1=0; f1<2; ++f1){
            for (ix=0; ix<VOLUMEPLUSRAND; ++ix)
               _bispinor_null(propsecneighbour[12*f1+3*s1+c1][ix]);
            for (ix=0; ix<VOLUME; ++ix){
               _spinor_assign( propsecneighbour[12*f1+3*s1+c1][ix].sp_up, tmpbisp2d[12*f1+3*s1+c1][g_idn[ix][TUP]].sp_up);
               _spinor_assign( propsecneighbour[12*f1+3*s1+c1][ix].sp_dn, tmpbisp2d[12*f1+3*s1+c1][g_idn[ix][TUP]].sp_dn);
            }
          }
      for (s1=spinorstart;s1<spinorend; ++s1)
       for (c1=0; c1<3; ++c1)
         for (f1=0; f1<2; ++f1){
           count=0;
           generic_exchange_direction_nonblocking( propsecneighbour[12*f1+3*s1+c1], sizeof(bispinor), TDOWN, request, &count );
           MPI_Waitall( count, request, statuses);
        }
   }
   for (i=0; i<T_global; ++i)
      paulitrace[i]=0.;

// Trace over the Pauli matrices
   for (tauindex=0; tauindex<3; ++tauindex){

//Trace over flavour degrees of freedom
      for (i=0; i<T_global; ++i)
         flavortrace[i]=0.;

      for (f1=0; f1<2; ++f1){

//Trace over spinor indices
         for (i=0; i<8*T_global; ++i){
            spinortrace[i]=0.;
         }

         for (s1=spinorstart; s1<spinorend; ++s1){

//Trace over spatial indices
            for (i=0; i<8*T_global; ++i){
               spacetrace[i]=0.;
            }
            for (ix=0; ix<VOLUME; ++ix){

//Trace over the color indices for each sites
               for (i=0; i<8; ++i)
                  colortrace[i]=0.;
               for (c1=0; c1<3; ++c1){

/*   
       TYPE IV.1.a OR  IV.1.b                                     (1+gamma5)/2*S(x    ,ytilde)
       TYPE IV.2.a OR  IV.2.b     U0^dagger(x-0)*U0^dagger(x-2*0)*(1+gamma5)/2*S(x-2*0,ytilde)
*/
                  _vector_null(running.sp_up.s2);
                  _vector_null(running.sp_up.s3);
                  _vector_null(running.sp_dn.s2);
                  _vector_null(running.sp_dn.s3);
                  if ( type_12 == TYPE_2){
//for the up quark
                     _vector_assign( running.sp_up.s0, propsecneighbour[12*f1 + 3*s1 + c1][g_idn[ix][TUP]].sp_up.s0);
                     _vector_assign( running.sp_up.s1, propsecneighbour[12*f1 + 3*s1 + c1][g_idn[ix][TUP]].sp_up.s1);

//for the down quark
                     _vector_assign( running.sp_dn.s0, propsecneighbour[12*f1 + 3*s1 + c1][g_idn[ix][TUP]].sp_dn.s0);
                     _vector_assign( running.sp_dn.s1, propsecneighbour[12*f1 + 3*s1 + c1][g_idn[ix][TUP]].sp_dn.s1);
                  }
                  else if ( type_12 == TYPE_1){
                     _vector_assign( running.sp_up.s0, propfields[12*s1 + 4*c1 + 2*f1][ix].sp_up.s0 );
                     _vector_assign( running.sp_up.s1, propfields[12*s1 + 4*c1 + 2*f1][ix].sp_up.s1 );

                     _vector_assign( running.sp_dn.s0, propfields[12*s1 + 4*c1 + 2*f1][ix].sp_dn.s0 );
                     _vector_assign( running.sp_dn.s1, propfields[12*s1 + 4*c1 + 2*f1][ix].sp_dn.s1 );
                  }
/*   
       TYPE IV.1.a OR  IV.1.b   tau_i*phi(x)*                                  (1+gamma5)/2*S(x+   ,ytilde)
       TYPE IV.2.a OR  IV.2.b   tau_i*phi(x)*  U0^dagger(x-0)*U0^dagger(x-2*0)*(1+gamma5)/2*S(x-2*0,ytilde)
*/
                  taui_scalarfield_spinor( &running, &running, GAMMA_UP, tauindex, ix, TDOWN, NO_DAGG);

/*   
       TYPE IV.1.a OR  IV.1.b   S(ytilde, x)*tau_i*phi(x)*                                  (1+gamma5)/2*S(x+   ,ytilde)
       TYPE IV.2.a OR  IV.2.b   S(ytilde, x)*tau_i*phi(x)*  U0^dagger(x-0)*U0^dagger(x-2*0)*(1+gamma5)/2*S(x-2*0,ytilde)
*/

                  multiply_backward_propagator(&running, propfields, &running, ix, NODIR);

/*   
       TYPE IV.1.a tau_i*phi(ytilde)*         S(ytilde, x)*tau_i*phi(x)*                                  (1+gamma5)/2*S(x+   ,ytilde)
       TYPE IV.1.b phi^dagger(ytilde)*tau_i*  S(ytilde, x)*tau_i*phi(x)*                                  (1+gamma5)/2*S(x+   ,ytilde)

       TYPE IV.2.a tau_i*phi(ytilde)*         S(ytilde, x)*tau_i*phi(x)*  U0^dagger(x-0)*U0^dagger(x-2*0)*(1+gamma5)/2*S(x-2*0,ytilde)
       TYPE IV.2.b phi^dagger(ytilde)*tau_i*  S(ytilde, x)*tau_i*phi(x)*  U0^dagger(x-0)*U0^dagger(x-2*0)*(1+gamma5)/2*S(x-2*0,ytilde)
*/
                  //delta( color component of bispinor running, c1) for all spinor and flavor indices
                  trace_in_color(colortrace, &running, c1 );
               } //End of trace color
               //sum over all lattice sites the result of the color trace
               trace_in_space( spacetrace, colortrace, ix);
            }  //End of trace in space

//Gather the results from all nodes to complete the trace in space
            for (i=0; i<8*T_global; ++i){
               _Complex double tmp;
               MPI_Allreduce(&spacetrace[i], &tmp, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
               spacetrace[i]= tmp;
            }
            // delta (spinor components of spacetrace, s1) for all time slices and flavor indices
            trace_in_spinor(spinortrace, spacetrace, s1);

         } //End of trace in spinor space

         if ( type_ab == TYPE_A ){
           taui_scalarfield_flavoronly( spinortrace, tauindex, NO_DAGG );
         }
         else if ( type_ab == TYPE_B){
           taui_scalarfield_flavoronly( spinortrace, tauindex, DAGGER  );
         }
         //delta(flavor component in spinortrace, f1) for all time slices
         trace_in_flavor( flavortrace, spinortrace, f1 );

      } //End of trace in flavor space
      //sum for all Pauli matrices
      for (i=0;i<T_global; ++i)
         paulitrace[i]+=flavortrace[i];
   } //End of trace for Pauli matrices


   if (g_cart_id == 0){printf("Wilson term Dirac Current Density correlator typeIV results= %10.10e %10.10e\n", type_12 == TYPE_1 ? '1' : '2',type_ab == TYPE_A ? 'a' :'b');}
   for (i=0; i<T_global; ++i){
      if (g_cart_id == 0){
        printf("%3d %10.10e %10.10e\n", i, creal(paulitrace[i]), cimag(paulitrace[i]));
      }
   }
   free(flavortrace);
   free(paulitrace);
   free(spacetrace);
   free(spinortrace);
   free(colortrace);

   if (type_12 == TYPE_2){
     for(i=0;i<24;++i){
       free(tmpbisp2d[i]);
       free(propsecneighbour[i]);
     }
     free(tmpbisp2d);
     free(propsecneighbour);
   }
   free(request); 
}
void wilsonterm_current_density_512ab( bispinor ** propfields, int type_12, int type_ab ){
   int ix,i;
   int f1,c1,s1,tauindex;
   int spinorstart, spinorend;
   su3 * restrict upm;
   bispinor running;
   su3_vector tmpvec;
   int count;
   MPI_Status  statuses[8];
   MPI_Request *request;
   request=( MPI_Request *) malloc(sizeof(MPI_Request)*8);

   _Complex double *colortrace;
   _Complex double *spacetrace;
   _Complex double *spinortrace;
   _Complex double *flavortrace;
   _Complex double *paulitrace;

   colortrace= (_Complex double *)malloc(sizeof(_Complex double)*8);
   spacetrace= (_Complex double *)malloc(sizeof(_Complex double)*8*T_global);
   spinortrace=(_Complex double *)malloc(sizeof(_Complex double)*2*T_global);
   flavortrace=(_Complex double *)malloc(sizeof(_Complex double)*T_global);
   paulitrace= (_Complex double *)malloc(sizeof(_Complex double)*T_global);

   if ( type_ab == TYPE_A ) {
     spinorstart=0;
     spinorend  =2;
   }
   else if ( type_ab == TYPE_B ){
     spinorstart=2;
     spinorend  =4;
   }
//Doing the neccesary communication
   for (s1=spinorstart; s1<spinorend; ++s1)
     for (c1=0; c1<3; ++c1)
       for (f1=0; f1<2; ++f1){
           count=0;
           generic_exchange_direction_nonblocking( propfields[12*s1 + 4*c1 + 2*f1 + 0], sizeof(bispinor), TDOWN , request, &count );
           MPI_Waitall( count, request, statuses);
           count=0;
           generic_exchange_direction_nonblocking( propfields[12*s1 + 4*c1 + 2*f1 + 1], sizeof(bispinor), TUP   , request, &count );
           MPI_Waitall( count, request, statuses);
           count=0;
           generic_exchange_direction_nonblocking( propfields[12*s1 + 4*c1 + 2*f1 + 1], sizeof(bispinor), TDOWN , request, &count);
           MPI_Waitall( count, request, statuses);
       }
   free(request);
// Trace over the Pauli matrices
   for (tauindex=0; tauindex<3; ++tauindex){

//Trace over flavour degrees of freedom
      for (i=0; i<T_global; ++i)
         flavortrace[i]=0.;

      for (f1=0; f1<2; ++f1){

//Trace over spinor indices
         for (i=0; i<8*T_global; ++i){
            spinortrace[i]=0.;
         }

         for (s1=spinorstart; s1<spinorend; ++s1){

//Trace over spatial indices
            for (i=0; i<8*T_global; ++i){
               spacetrace[i]=0.;
            }
            for (ix=0; ix<VOLUME; ++ix){

//Trace over the color indices for each sites
               for (i=0; i<8; ++i)
                  colortrace[i]=0.;
               for (c1=0; c1<3; ++c1){
/*   
       TYPE V.1.a OR  V.1.b     U0^dagger(x)*U0^dagger(x-0)* (1-gamma5)/2 *  S(x-0,ytilde)
       TYPE V.2.a OR  V.2.b                                  (1-gamma5)/2 *  S(x-0,ytilde)
*/
                  _vector_null(running.sp_up.s0);
                  _vector_null(running.sp_up.s1);
                  _vector_null(running.sp_dn.s0);
                  _vector_null(running.sp_dn.s1);

                  if ( type_12 == TYPE_1){
                    upm = &g_gauge_field[g_idn[ix][TUP]][TUP];
//for the up quark
                    _su3_inverse_multiply(running.sp_up.s2, (*upm), propfields[12*s1 + 4*c1 + 2*f1][g_idn[ix][TUP]].sp_up.s2);
                    _su3_inverse_multiply(running.sp_up.s3, (*upm), propfields[12*s1 + 4*c1 + 2*f1][g_idn[ix][TUP]].sp_up.s3);

//for the down quark
                    _su3_inverse_multiply(running.sp_dn.s2, (*upm), propfields[12*s1 + 4*c1 + 2*f1][g_idn[ix][TUP]].sp_dn.s2);
                    _su3_inverse_multiply(running.sp_dn.s3, (*upm), propfields[12*s1 + 4*c1 + 2*f1][g_idn[ix][TUP]].sp_dn.s3);

                    upm = &g_gauge_field[ix][TUP];

//for the up quark
                    _vector_null( tmpvec );
                    _su3_inverse_multiply(tmpvec, (*upm), running.sp_up.s0);
                    _vector_assign(  running.sp_up.s0, tmpvec);
 
                    _vector_null( tmpvec );
                    _su3_inverse_multiply(tmpvec, (*upm), running.sp_up.s1);
                    _vector_assign(  running.sp_up.s1, tmpvec);

//for the down quark

                    _vector_null( tmpvec );
                    _su3_inverse_multiply(tmpvec, (*upm), running.sp_dn.s0);
                    _vector_assign(  running.sp_dn.s0, tmpvec);

                    _vector_null( tmpvec );
                    _su3_inverse_multiply(tmpvec, (*upm), running.sp_dn.s1);
                    _vector_assign(  running.sp_dn.s1, tmpvec);
                  }
                  else if ( type_12 == TYPE_2){
                    _vector_assign( running.sp_up.s0, propfields[12*s1 + 4*c1 + 2*f1][g_idn[ix][TUP]].sp_up.s2 );
                    _vector_assign( running.sp_up.s1, propfields[12*s1 + 4*c1 + 2*f1][g_idn[ix][TUP]].sp_up.s3 );

                    _vector_assign( running.sp_dn.s0, propfields[12*s1 + 4*c1 + 2*f1][g_idn[ix][TUP]].sp_dn.s2 );
                    _vector_assign( running.sp_dn.s1, propfields[12*s1 + 4*c1 + 2*f1][g_idn[ix][TUP]].sp_dn.s3 );
                  }
/*   
       TYPE V.1.a OR  V.1.b     phi^dagger(x)*tau_i*     U0^dagger(x)*U0^dagger(x-0)* (1-gamma5)/2 *  S(x-0,ytilde)
       TYPE V.2.a OR  V.2.b     phi^dagger(x)*tau_i                                        (1-gamma5)/2 *  S(x-0,ytilde)
*/
                 
                  taui_scalarfield_spinor( &running, &running, GAMMA_DN, tauindex, ix, NODIR, DAGGER);

/*   
       TYPE V.1.a OR  V.1.b     S(ytilde,x+0)*phi^dagger(x)*tau_i*     U0^dagger(x)*U0^dagger(x-0)* (1-gamma5)/2 *  S(x-0,ytilde)
       TYPE V.2.a OR  V.2.b     S(ytilde,x-0)*phi^dagger(x)*tau_i                                   (1-gamma5)/2 *  S(x-0,ytilde)
*/

                  if (type_12 == TYPE_1){
                    multiply_backward_propagator(&running, propfields, &running, ix, TUP);
                  }
                  else if (type_12 == TYPE_2){
                    multiply_backward_propagator(&running, propfields, &running, ix, TDOWN);
                  }
/*   
       TYPE V.1.a                 tau_i*phi(ytilde)*       (1+gamma5)/2* S(ytilde, x-0)*tau_i*phi(x)*U0(x-0)*U0(x)* (1+gamma5)/2 *  S(x+0,ytilde)
       TYPE V.1.b                 phi^dagger(ytilde)*tau_i*(1-gamma5)/2* S(ytilde, x-0)*tau_i*phi(x)*U0(x-0)*U0(x)* (1+gamma5)/2 *  S(x+0,ytilde)

       TYPE V.2.a                 tau_i*phi(ytilde)*       (1+gamma5)/2* S(ytilde, x-0)*tau_i*phi(x)*               (1+gamma5)/2 *  S(x-0,ytilde)
       TYPE V.2.b                 phi^dagger(ytilde)*tau_i*(1-gamma5)/2* S(ytilde, x-0)*tau_i*phi(x)*               (1+gamma5)/2 *  S(x-0,ytilde)

*/
                  //delta( color component of bispinor running, c1) for all spinor and flavor indices
                  trace_in_color(colortrace, &running, c1 );
               } //End of trace color
               //sum over all lattice sites the result of the color trace
               trace_in_space( spacetrace, colortrace, ix);
            }  //End of trace in space

//Gather the results from all nodes to complete the trace in space
            for (i=0; i<8*T_global; ++i){
               _Complex double tmp;
               MPI_Allreduce(&spacetrace[i], &tmp, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
               spacetrace[i]= tmp;
            }
            // delta (spinor components of spacetrace, s1) for all time slices and flavor indices
            trace_in_spinor(spinortrace, spacetrace, s1);

         } //End of trace in spinor space

         if ( type_ab == TYPE_A ){
           taui_scalarfield_flavoronly( spinortrace, tauindex, NO_DAGG );
         }
         else if ( type_ab == TYPE_B){
           taui_scalarfield_flavoronly( spinortrace, tauindex, DAGGER  );
         }
         //delta(flavor component in spinortrace, f1) for all time slices
         trace_in_flavor( flavortrace, spinortrace, f1 );

      } //End of trace in flavor space

      for (i=0;i<T_global; ++i)
         paulitrace[i]+=flavortrace[i];
   } //End of trace for Pauli matrices


   if (g_cart_id == 0){printf("Wilson term Dirac Current Density correlator typeV results= %10.10e %10.10e\n", type_12 == TYPE_1 ? '1' : '2',type_ab == TYPE_A ? 'a' :'b');}
   for (i=0; i<T_global; ++i){
      if (g_cart_id == 0){
        printf("%3d %10.10e %10.10e\n", i, creal(paulitrace[i]), cimag(paulitrace[i]));
      }
   }
   free(flavortrace);
   free(paulitrace);
   free(spacetrace);
   free(spinortrace);
   free(colortrace);
}
void wilsonterm_current_density_612ab( bispinor ** propfields, int type_12, int type_ab ){
   int ix,i;
   int f1,c1,s1,tauindex;
   int spinorstart, spinorend;
   bispinor **starting2d;
   bispinor **running2d;
   bispinor **tmpbisp2d;
   su3 * restrict upm;
   su3_vector tmpvec;
   int count;
   MPI_Status  statuses[8];
   MPI_Request *request;
   _Complex double *colortrace;
   _Complex double *spacetrace;
   _Complex double *spinortrace;
   _Complex double *flavortrace;
   _Complex double *paulitrace;

   colortrace= (_Complex double *)malloc(sizeof(_Complex double)*8);
   spacetrace= (_Complex double *)malloc(sizeof(_Complex double)*8*T_global);
   spinortrace=(_Complex double *)malloc(sizeof(_Complex double)*2*T_global);
   flavortrace=(_Complex double *)malloc(sizeof(_Complex double)*T_global);
   paulitrace= (_Complex double *)malloc(sizeof(_Complex double)*T_global);


   request=( MPI_Request *) malloc(sizeof(MPI_Request)*8);
   if ( type_ab == TYPE_A ) {
        spinorstart=0;
        spinorend  =2;
   }
   else if ( type_ab == TYPE_B ){
        spinorstart=2;
        spinorend  =4;
   }
   tmpbisp2d= (bispinor **)malloc(sizeof(bispinor *)*24);
   running2d= (bispinor **)malloc(sizeof(bispinor *)*24);
   starting2d=(bispinor **)malloc(sizeof(bispinor *)*24);
   for (i=0; i<24; ++i){
     tmpbisp2d[i] =(bispinor *)malloc(sizeof(bispinor)*VOLUMEPLUSRAND);
     starting2d[i]=(bispinor *)malloc(sizeof(bispinor)*VOLUMEPLUSRAND);
     running2d[i] =(bispinor *)malloc(sizeof(bispinor)*VOLUMEPLUSRAND);
     for (ix=0; ix<VOLUME; ++ix){
       _bispinor_null( running2d[i][ix]);
       _bispinor_null(starting2d[i][ix]);
       _bispinor_null( tmpbisp2d[i][ix]);
     }
   } 
//Doing the neccesary communication
   for (s1=spinorstart; s1<spinorend; ++s1)
     for (c1=0; c1<3; ++c1)
       for (f1=0; f1<2; ++f1){
           count=0;
           generic_exchange_direction_nonblocking( propfields[12*s1 + 4*c1 + 2*f1 + 0], sizeof(bispinor), TUP   , request, &count );
           MPI_Waitall( count, request, statuses);
           count=0;
           generic_exchange_direction_nonblocking( propfields[12*s1 + 4*c1 + 2*f1 + 1], sizeof(bispinor), TDOWN , request, &count );
           MPI_Waitall( count, request, statuses);
   }
/*************************
Creating U0(x-2*0)U0(x-0)*S(x, ytilde) in two steps:
1; Doing the product U0(x-0)*U0(x)*S( x+0, ytilde)
2; Gathering in direction TDOWN
**************************/
   if (type_12 == TYPE_2){
      for (ix = 0; ix< VOLUME; ++ix)
        for (s1=spinorstart;s1<spinorend; ++s1)
          for (c1=0; c1<3; ++c1)
            for (f1=0; f1<2; ++f1){

               upm = &g_gauge_field[ix][TUP];

               _su3_multiply(starting2d[12*f1+3*s1+c1][ix].sp_up.s2, (*upm), propfields[12*s1 + 4*c1 + 2*f1][g_iup[ix][TUP]].sp_up.s2);
               _su3_multiply(starting2d[12*f1+3*s1+c1][ix].sp_up.s3, (*upm), propfields[12*s1 + 4*c1 + 2*f1][g_iup[ix][TUP]].sp_up.s3);

               _su3_multiply(starting2d[12*f1+3*s1+c1][ix].sp_dn.s2, (*upm), propfields[12*s1 + 4*c1 + 2*f1][g_iup[ix][TUP]].sp_dn.s2);
               _su3_multiply(starting2d[12*f1+3*s1+c1][ix].sp_dn.s3, (*upm), propfields[12*s1 + 4*c1 + 2*f1][g_iup[ix][TUP]].sp_dn.s3);

               upm = &g_gauge_field[g_idn[ix][TUP]][TUP];

               _su3_multiply(tmpvec, (*upm), starting2d[12*f1+3*s1+c1][ix].sp_up.s2);
               _vector_assign(  starting2d[12*f1+3*s1+c1][ix].sp_up.s2, tmpvec);
               _su3_multiply(tmpvec, (*upm), starting2d[12*f1+3*s1+c1][ix].sp_up.s3);
               _vector_assign(  starting2d[12*f1+3*s1+c1][ix].sp_up.s3, tmpvec);

               _su3_multiply(tmpvec, (*upm), starting2d[12*f1+3*s1+c1][ix].sp_dn.s2);
               _vector_assign(  starting2d[12*f1+3*s1+c1][ix].sp_dn.s2, tmpvec);
               _su3_multiply(tmpvec, (*upm), starting2d[12*f1+3*s1+c1][ix].sp_dn.s3);
               _vector_assign(  starting2d[12*f1+3*s1+c1][ix].sp_dn.s3, tmpvec);
            }
      for (s1=spinorstart;s1<spinorend; ++s1)
        for (c1=0; c1<3; ++c1)
          for (f1=0; f1<2; ++f1){
            count=0;
            generic_exchange_direction_nonblocking( starting2d[12*f1+3*s1+c1], sizeof(bispinor), TDOWN, request, &count );
            MPI_Waitall( count, request, statuses);
      }
   }
// Trace over the Pauli matrices
   for (tauindex=0; tauindex<3; ++tauindex){

//Trace over flavour degrees of freedom
      for (i=0; i<T_global; ++i)
         flavortrace[i]=0.;

      for (f1=0; f1<2; ++f1){

//Trace over spinor indices
         for (i=0; i<8*T_global; ++i){
            spinortrace[i]=0.;
         }

         for (s1=spinorstart; s1<spinorend; ++s1){

//Trace over spatial indices
            for (i=0; i<8*T_global; ++i){
               spacetrace[i]=0.;
            }
            for (ix=0; ix<VOLUME; ++ix){

//Trace over the color indices for each sites
               for (c1=0; c1<3; ++c1){
/*   
       TYPE VI.1.a OR  VI.1.b                                     (1-gamma5)/2*S(x    ,ytilde)
       TYPE VI.2.a OR  VI.2.b                   U0(x-2*0)*U0(x-0)*(1-gamma5)/2*S(x    ,ytilde)
*/
                  _vector_null(running2d[12*f1 + 3*s1 + c1][ix].sp_up.s0);
                  _vector_null(running2d[12*f1 + 3*s1 + c1][ix].sp_up.s1);
                  _vector_null(running2d[12*f1 + 3*s1 + c1][ix].sp_dn.s0);
                  _vector_null(running2d[12*f1 + 3*s1 + c1][ix].sp_dn.s1);
                  if ( type_12 == TYPE_2){
//for the up quark
                    _vector_assign( running2d[12*f1+3*s1+c1][ix].sp_up.s2, starting2d[12*f1 + 3*s1 + c1][g_idn[ix][TUP]].sp_up.s2);
                    _vector_assign( running2d[12*f1+3*s1+c1][ix].sp_up.s3, starting2d[12*f1 + 3*s1 + c1][g_idn[ix][TUP]].sp_up.s3);

//for the down quark
                    _vector_assign( running2d[12*f1+3*s1+c1][ix].sp_dn.s2, starting2d[12*f1 + 3*s1 + c1][g_idn[ix][TUP]].sp_dn.s2);
                    _vector_assign( running2d[12*f1+3*s1+c1][ix].sp_dn.s3, starting2d[12*f1 + 3*s1 + c1][g_idn[ix][TUP]].sp_dn.s3);

                  }
                  else if ( type_12 == TYPE_1){
                    _vector_assign( running2d[12*f1+3*s1+c1][ix].sp_up.s2, propfields[12*s1 + 4*c1 + 2*f1][ix].sp_up.s2 );
                    _vector_assign( running2d[12*f1+3*s1+c1][ix].sp_up.s3, propfields[12*s1 + 4*c1 + 2*f1][ix].sp_up.s3 );

                    _vector_assign( running2d[12*f1+3*s1+c1][ix].sp_dn.s2, propfields[12*s1 + 4*c1 + 2*f1][ix].sp_dn.s2 );
                    _vector_assign( running2d[12*f1+3*s1+c1][ix].sp_dn.s3, propfields[12*s1 + 4*c1 + 2*f1][ix].sp_dn.s3 );
                  }
/*   
       TYPE VI.1.a OR  VI.1.b   phi^dagger(x-0)*tau_i*                    (1-gamma5)/2*S(x,ytilde)
       TYPE VI.2.a OR  VI.2.b   phi^dagger(x-0)*tau_i*  U0(x-2*0)*U0(x-0)*(1-gamma5)/2*S(x,ytilde)
*/
                  taui_scalarfield_spinor( &running2d[12*f1 + 3*s1 +c1][ix], &running2d[12*f1 + 3*s1 +c1][ix], GAMMA_DN, tauindex, ix, TDOWN, DAGGER);


               }

            }
/*   
       TYPE VI.1.a OR  VI.1.b   S(ytilde, x)    *phi^dagger(x-0)*tau_i*                    (1-gamma5)/2*S(x,ytilde)
       TYPE VI.2.a OR  VI.2.b   S(ytilde, x-2*0)*phi^dagger(x-0)*tau_i*  U0(x-2*0)*U0(x-0)*(1-gamma5)/2*S(x,ytilde)
*/

/**************************
Multiplication with Stilde(x-2*0,ytilde)P(x) in three steps:
1; Gathering P(x) from direction +0
2; Multiplying Stilde(x-O,ytilde) with P(x+0)
3; Gathering The result in direction -0

**************************/


            if ( type_12 == TYPE_2 ) {
              for (c1=0; c1<3; ++c1){
                 count=0;
                 generic_exchange_direction_nonblocking( running2d[12*f1+3*s1+c1], sizeof(bispinor), TUP, request, &count );
                 MPI_Waitall( count, request, statuses);
              }
            }
            for (ix = 0; ix< VOLUME; ++ix){
               for (c1=0; c1<3; ++c1){
                  if (type_12 == TYPE_2){
                    multiply_backward_propagator(&tmpbisp2d[12*f1+3*s1+c1][ix], propfields, &running2d[12*f1+3*s1+c1][g_iup[ix][TUP]], ix, TDOWN);
                  }
                  else if (type_12 == TYPE_1){
                    multiply_backward_propagator(&running2d[12*f1+3*s1+c1][ix], propfields, &running2d[12*f1+3*s1+c1][ix]            , ix, NODIR);
                  }
               }
            }
            if ( type_12 == TYPE_2 ) {
               for (c1=0; c1<3; ++c1){
                  count=0;
                  generic_exchange_direction_nonblocking( tmpbisp2d[12*f1+3*s1+c1], sizeof(bispinor), TDOWN, request, &count );
                  MPI_Waitall( count, request, statuses);
               }
               for (ix = 0; ix< VOLUME; ++ix){
                  for (c1=0; c1<3; ++c1){
                     if (type_12 == TYPE_2){
                        _spinor_assign( running2d[12*f1+3*s1+c1][ix].sp_up, tmpbisp2d[12*f1 + 3*s1 +c1][g_idn[ix][TUP]].sp_up );
                        _spinor_assign( running2d[12*f1+3*s1+c1][ix].sp_dn, tmpbisp2d[12*f1 + 3*s1 +c1][g_idn[ix][TUP]].sp_dn );
                     }
                  }
               }
            }

            for (ix = 0; ix< VOLUME; ++ix){
//Trace over the color indices for each sites
               for (i=0; i<8; ++i)
                  colortrace[i]=0.;
               for (c1=0; c1<3; ++c1){
                  //delta( color component of bispinor running, c1) for all spinor and flavor indices
                  trace_in_color(colortrace,&running2d[12*f1 + 3*s1 +c1][ix],c1);
               
               }  //End of trace in color
               //sum over all lattice sites the result of the color trace
               trace_in_space(spacetrace,colortrace,ix);

            } //End of trace in space


//Gather the results from all nodes to complete the trace in space
            for (i=0; i<T_global; ++i){
               _Complex double tmp;
               MPI_Allreduce(&spacetrace[i], &tmp, 1, MPI_DOUBLE_COMPLEX, MPI_SUM, MPI_COMM_WORLD);
               spacetrace[i]= tmp;
            }
            // delta (spinor components of spacetrace, s1) for all time slices and flavor indices
            trace_in_spinor(spinortrace, spacetrace, s1);

         }//End of trace in spinor space

/*   
       TYPE VI.1.a    tau_i*phi(ytilde)*       (1+gamma5)/2*S(ytilde, x)        *phi^dagger(x-0)*tau_i*                    (1-gamma5)/2*S(x,ytilde)
       TYPE VI.1.b    phi^dagger(ytilde)*tau_i*(1-gamma5)/2*S(ytilde, x)        *phi^dagger(x-0)*tau_i*                    (1-gamma5)/2*S(x,ytilde)

       TYPE VI.2.a    tau_i*phi(ytilde)*       (1+gamma5)/2*S(ytilde, x-2*0)    *phi^dagger(x-0)*tau_i*  U0(x-2*0)*U0(x-0)*(1-gamma5)/2*S(x,ytilde)
       TYPE VI.2.b    phi^dagger(ytilde)*tau_i*(1-gamma5)/2*S(ytilde, x-2*0)    *phi^dagger(x-0)*tau_i*  U0(x-2*0)*U0(x-0)*(1-gamma5)/2*S(x,ytilde)

*/
         if ( type_ab == TYPE_A ){
           taui_scalarfield_flavoronly( spinortrace, tauindex, NO_DAGG );
         }
         else if ( type_ab == TYPE_B ){
           taui_scalarfield_flavoronly( spinortrace, tauindex, DAGGER  );
         }
         //delta(flavor component in spinortrace, f1) for all time slices
         trace_in_flavor( flavortrace, spinortrace, f1 );

      } //End of trace in flavor space
      //sum for all Pauli matrices
      for (i=0;i<T_global; ++i)
         paulitrace[i]+=flavortrace[i];
   } //End of trace for Pauli matrices


   if (g_cart_id == 0){printf("Wilson term Dirac Current Density correlator typeVI results= %10.10e %10.10e\n", type_12 == TYPE_1 ? '1' : '2',type_ab == TYPE_A ? 'a' :'b');}
   for (i=0; i<T_global; ++i){
      if (g_cart_id == 0){
        printf("%3d %10.10e %10.10e\n", i, creal(paulitrace[i]), cimag(paulitrace[i]));
      }
   }
   free(flavortrace);
   free(paulitrace);
   free(spacetrace);
   free(spinortrace);
   free(colortrace);
   free(request);
}
void main(int argc, char *argv[]){
  FILE *parameterfile = NULL;
  char datafilename[206];
  char parameterfilename[206];
  char conf_filename[50];
  char scalar_filename[50];
  char * input_filename = NULL;
  char * filename = NULL;
  double plaquette_energy;
  int i;
  char prop_fname[200];
  int src_idx, pos;
  int count;
  int status_geo;
  MPI_Status  statuses[8];
  MPI_Request *request;
  spinor *tmpspinoru;
  spinor *tmpspinord;
  request=( MPI_Request *) malloc(sizeof(MPI_Request)*8);

  MPI_Init(&argc, &argv);

  MPI_Comm_rank(MPI_COMM_WORLD, &g_proc_id);

  process_args(argc, argv, &input_filename,&filename);
  set_default_filenames(&input_filename, &filename);

  /* Read the input file */
  if( (i = read_input(input_filename)) != 0) {
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
  init_gauge_field(VOLUMEPLUSRAND + g_dbw2rand, 0);

  init_geometry_indices(VOLUMEPLUSRAND + g_dbw2rand);

  int numbScalarFields = 4;
  i = init_scalar_field(VOLUMEPLUSRAND, numbScalarFields);
  if ( i!= 0) {
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

  boundary(-1.0);

  status_geo = check_geometry();
  if (status_geo != 0) {
    fprintf(stderr, "Checking of geometry failed. Unable to proceed.\nAborting....\n");
    exit(1);
  }
  if (even_odd_flag) {
    i = init_spinor_field(VOLUMEPLUSRAND / 2, 2);
  }
  else {
    i = init_spinor_field(VOLUMEPLUSRAND, 2);
  }

  start_ranlux(1, 123456);
  i = init_bispinor_field(VOLUMEPLUSRAND, 48);
  if ( i!= 0) {
    fprintf(stderr, "Not enough memory for bispinor fields! Aborting...\n");
    exit(0);
  }
  sprintf(conf_filename, "%s.%.4d", gauge_input_filename, nstore);
  if (g_cart_id == 0) {
    printf("#\n# Trying to read gauge field from file %s in %s precision.\n",
           conf_filename, (gauge_precision_read_flag == 32 ? "single" : "double"));
    fflush(stdout);
  }
  if ( (i = read_gauge_field(conf_filename,g_gauge_field)) !=0) {
    fprintf(stderr, "Error %d while reading gauge field from %s\n Aborting...\n", i, conf_filename);
    exit(-2);
  }
  if (g_cart_id == 0) {
    printf("# Finished reading gauge field.\n");
    fflush(stdout);
  }
  sprintf(scalar_filename, "%s.%d", scalar_input_filename, nscalar);
  if (g_cart_id == 0) {
    printf("#\n# Trying to read scalar field from file %s in %s precision.\n",
           scalar_filename, (scalar_precision_read_flag == 32 ? "single" : "double"));
    fflush(stdout);
  }
  if ( (i = read_scalar_field_parallel(scalar_filename,g_scalar_field)) !=0) {
    fprintf(stderr, "Error %d while reading scalar field from %s\n Aborting...\n", i, scalar_filename);
    exit(-2);
  }
  if (g_cart_id == 0) {
    printf("# Finished reading scalar field.\n");
    fflush(stdout);
  }
  g_smearedscalar=(scalar *)malloc(sizeof(scalar *)*4);
  for (i=0; i<4; ++i)
    g_smearedscalar[i]= (scalar *)malloc(sizeof(scalar)*(VOLUMEPLUSRAND));
  smear_scalar_fields(g_scalar_field, g_smearedscalar);

  xchange_gauge(g_gauge_field);
  /*compute the energy of the gauge field*/
  plaquette_energy = measure_plaquette( (const su3**) g_gauge_field);


  if (g_cart_id == 0) {
    printf("# The computed plaquette value is %e.\n", plaquette_energy / (6.*VOLUME*g_nproc));
    fflush(stdout);
  }

  for( int s=0; s<numbScalarFields; s++ ){
    count=0;
    generic_exchange_direction_nonblocking( g_scalar_field[s], sizeof(scalar), TDOWN, request, &count );
    MPI_Waitall( count, request, statuses);
    count=0;
    generic_exchange_direction_nonblocking( g_scalar_field[s], sizeof(scalar), TUP  , request, &count );
    MPI_Waitall( count, request, statuses);
  }


  spinor ** temp_field = NULL;
  init_solver_field(&temp_field, VOLUMEPLUSRAND, 2);
  for( src_idx = 0; src_idx < 12; src_idx++ ){
    snprintf(prop_fname,200,"bsm2prop.0400.00.%02d.000.inverted",src_idx);
    for(pos = 0; pos < 8; ){
      printf("READCHECK: Propagator in pos %02d from file %s\n", pos/2,prop_fname);
   
//read the propagator from source d to sink d 
      read_spinor(g_spinor_field[0], g_spinor_field[1], prop_fname, pos);
      convert_eo_to_lexic(temp_field[0], g_spinor_field[0], g_spinor_field[1]);
      pos+=1;

//read the propagator from source d to sink u
      read_spinor(g_spinor_field[0], g_spinor_field[1], prop_fname, pos);
      convert_eo_to_lexic(temp_field[1], g_spinor_field[0], g_spinor_field[1]);
      pos+=1;
//create a bispinor first insert sink u then sink d
//Store them in such a way that the u-ones should come first
      compact(g_bispinor_field[pos > 4 ? src_idx*4+pos/2-3 : src_idx*4+pos/2+ 1], temp_field[1], temp_field[0]);
    }
  }
  if (g_cart_id == 0) printf("Reading is successfull\n");
  finalize_solver(temp_field,2);
  free(request);
  free_gauge_field();
  free_geometry_indices();
  free_bispinor_field();
  free_scalar_field();
  int ii;	
  for ( ii= 0; ii< 4; ++ii)
     free(g_smearedscalar[ii]);
  free(g_smearedscalar);
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Finalize();

}
