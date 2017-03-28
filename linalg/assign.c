/***********************************************************************
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
 ***********************************************************************/
/*******************************************************************************
 *
 * File assign.c
 *
 *   void assign(spinor * const R, spinor * const S)
 *     Assign (*R) = (*S)
 *
 *******************************************************************************/

#ifdef HAVE_CONFIG_H
# include<config.h>
#endif
#include <stdlib.h>
#include <stdio.h>
#include <complex.h>
#include <math.h>
#include <string.h>
#include "su3.h"
#include "assign.h"


/* S input, R output        */
/* S and R must not overlap */
void assign(spinor * const R, spinor * const S, const int N)
{
  memcpy(R, S, N*sizeof(spinor));
  return;
}

void bispinor_assign(bispinor * const R, bispinor * const S, const int N)
{
  memcpy(R, S, N*sizeof(bispinor));
  return;
}

//copy a complex double S of size N into a spinor R of size N/24 
/* S input, R output        */
/* S and R must not overlap */
void assign_complex_to_bispinor(bispinor * const R, _Complex double * const S, const int N)
{

  int k; //spinor index
  bispinor *r;
  _Complex double *s;

  k=0;
  for(int ix=0; ix<N ; ix +=24)
  {
     s=S + ix;
     r=R + k;

     (r->sp_up).s0.c0 = *s;
     (r->sp_up).s0.c1 = *(s+1);
     (r->sp_up).s0.c2 = *(s+2);

     (r->sp_up).s1.c0 = *(s+3);
     (r->sp_up).s1.c1 = *(s+4);
     (r->sp_up).s1.c2 = *(s+5);

     (r->sp_up).s2.c0 = *(s+6);
     (r->sp_up).s2.c1 = *(s+7);
     (r->sp_up).s2.c2 = *(s+8);


     (r->sp_up).s3.c0 = *(s+9);
     (r->sp_up).s3.c1 = *(s+10);
     (r->sp_up).s3.c2 = *(s+11);
     s=S + ix + 12;
     
     (r->sp_dn).s0.c0 = *s;
     (r->sp_dn).s0.c1 = *(s+1);
     (r->sp_dn).s0.c2 = *(s+2);

     (r->sp_dn).s1.c0 = *(s+3);
     (r->sp_dn).s1.c1 = *(s+4);
     (r->sp_dn).s1.c2 = *(s+5);

     (r->sp_dn).s2.c0 = *(s+6);
     (r->sp_dn).s2.c1 = *(s+7);
     (r->sp_dn).s2.c2 = *(s+8);


     (r->sp_dn).s3.c0 = *(s+9);
     (r->sp_dn).s3.c1 = *(s+10);
     (r->sp_dn).s3.c2 = *(s+11);

     k++;
  }

  return;
}



//copy a spinor S of size N into a complex double R of size 24*N 
/* S input, R output        */
/* S and R must not overlap */
void assign_bispinor_to_complex(_Complex double * const R, bispinor * const S, const int N)
{

  int k; //complex double index
  _Complex double *r;
  bispinor *s;
  int n=N/24;
  k=0;
  for(int ix=0; ix<n ; ix++)
  {
     s=S+ix;
     r=R+k;

     *r     =  (s->sp_up).s0.c0;
     *(r+1) =  (s->sp_up).s0.c1;
     *(r+2) =  (s->sp_up).s0.c2;

     *(r+3) =  (s->sp_up).s1.c0;
     *(r+4) =  (s->sp_up).s1.c1;
     *(r+5) =  (s->sp_up).s1.c2;

     *(r+6) =  (s->sp_up).s2.c0;
     *(r+7) =  (s->sp_up).s2.c1;
     *(r+8) =  (s->sp_up).s2.c2;

     *(r+9)  =  (s->sp_up).s3.c0;
     *(r+10) =  (s->sp_up).s3.c1;
     *(r+11) =  (s->sp_up).s3.c2;

     k +=12;

     r=R+k;

     *r     =  (s->sp_dn).s0.c0;
     *(r+1) =  (s->sp_dn).s0.c1;
     *(r+2) =  (s->sp_dn).s0.c2;

     *(r+3) =  (s->sp_dn).s1.c0;
     *(r+4) =  (s->sp_dn).s1.c1;
     *(r+5) =  (s->sp_dn).s1.c2;

     *(r+6) =  (s->sp_dn).s2.c0;
     *(r+7) =  (s->sp_dn).s2.c1;
     *(r+8) =  (s->sp_dn).s2.c2;

     *(r+9)  =  (s->sp_dn).s3.c0;
     *(r+10) =  (s->sp_dn).s3.c1;
     *(r+11) =  (s->sp_dn).s3.c2;
    
     k+=12;


  }

  return;
}          

/* S and R must not overlap */
void assign_complex_to_spinor(spinor * const R, _Complex double * const S, const int N)
{

  int k; //spinor index
  spinor *r;
  _Complex double *s;

  k=0;
  for(int ix=0; ix<N ; ix +=12)
  {
     s=S+ix;
     r=R+k;

     (r->s0).c0 = *s;
     (r->s0).c1 = *(s+1);
     (r->s0).c2 = *(s+2);
             
     (r->s1).c0 = *(s+3);
     (r->s1).c1 = *(s+4);
     (r->s1).c2 = *(s+5);

     (r->s2).c0 = *(s+6);
     (r->s2).c1 = *(s+7);
     (r->s2).c2 = *(s+8);


     (r->s3).c0 = *(s+9);
     (r->s3).c1 = *(s+10);
     (r->s3).c2 = *(s+11);

     k++;
  }
  
  return;
}



//copy a spinor S of size N into a complex double R of size 12*N 
/* S input, R output        */
/* S and R must not overlap */
void assign_spinor_to_complex(_Complex double * const R, spinor * const S, const int N)
{

  int k; //complex double index
  _Complex double *r;
  spinor *s;

  k=0;
  int n=N/12;
  for(int ix=0; ix<n ; ix++)
  {
     s=S+ix;
     r=R+k;

     *r     =  (s->s0).c0;
     *(r+1) =  (s->s0).c1;
     *(r+2) =  (s->s0).c2;

     *(r+3) =  (s->s1).c0;
     *(r+4) =  (s->s1).c1;
     *(r+5) =  (s->s1).c2;

     *(r+6) =  (s->s2).c0;
     *(r+7) =  (s->s2).c1;
     *(r+8) =  (s->s2).c2;

     *(r+9)  =  (s->s3).c0;
     *(r+10) =  (s->s3).c1;
     *(r+11) =  (s->s3).c2;

     k +=12;
  }

  return;
}


#ifdef WITHLAPH
void assign_su3vect(su3_vector * const R, su3_vector * const S, const int N)
{
  su3_vector *r,*s;

  for (int ix = 0; ix < N; ++ix) 
  {
    r=R+ix;      
    s=S+ix;
    
    r->c0 = s->c0;
    r->c1 = s->c1;
    r->c2 = s->c2;
  }
}
#endif
