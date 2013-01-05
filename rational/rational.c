/***********************************************************************
 *
 * Copyright (C) 2013 Carsten Urbach
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
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "global.h"
#include "zolotarev.h"
#include "rational.h"

// init a rational approximation in range [a:b]
// a,b should be the spectral range of the squared operator already
// order is the order n of the rational approximation [n,n]
// ca and cb specify the range of monomials to use (0 to order-1)

int init_rational(rational_t * rat, const int order, const double a, const double b, 
		  const int ca, const int cb) {
  double * ars = malloc(2*order*sizeof(double));
  double * ar;
  double pmu, pnu, np;

  // sanity check of input parameters
  if(ca > order-1 || cb > order-1 || ca < 0 || cb < 0 || ca > cb || order < 1) {
    fprintf(stderr, "parameters to init_rational out of range\n");
    return(-1);
  }
  np = cb - ca + 1;

  rat->order = order;
  rat->np = np;
  rat->crange[0] = ca;
  rat->crange[1] = cb;
  if(((rat->mu = (double*)malloc(np*sizeof(double))) == NULL)  ||
     ((rat->rmu = (double*)malloc(np*sizeof(double))) == NULL) ||
     ((rat->nu = (double*)malloc(np*sizeof(double))) == NULL)  ||
     ((rat->rnu = (double*)malloc(np*sizeof(double))) == NULL)) {
    fprintf(stderr, "Could not allocate memory for coefficients in init_rational\n");
    return(-2);
  }
  rat->eps = a/b;
  rat->range[0] = a;
  rat->range[1] = b;

  // compute optimal zolotarev approximation
  zolotarev(order, rat->eps, &rat->A, ars, &rat->delta);
  // FIX: do we have to divide A by sqrt(b)
  // restrict to relevant coefficients [2*ca:2*cb]
  ar = ars + 2*ca;
  // compute mu[] and nu[] = M*sqrt(ar), mu: r even, nu: r odd (M = sqrt(b))
  for (int i = 0; i < np; i++) {
    rat->mu[i] = sqrt(b * ar[2*i + 1]);
    rat->nu[i] = sqrt(b * ar[2*i]);
  }
  // compute the partial fraction coefficients rmu and rnu
  for (int i = 0; i < np; i++) {  
    pmu=1.0;
    pnu=1.0;

    for (int j = 0; j < np; j++) {
      if (j!=i) {
	pmu*=((ar[2*j]-ar[2*i+1])/(ar[2*j+1]-ar[2*i+1]));
	pnu*=((rat->mu[j]-rat->nu[i])/(rat->nu[j]-rat->nu[i]));
      }
    }

    rat->rmu[i]=b*(ars[2*i]-ars[2*i+1])*pmu;
    rat->rnu[i]=(rat->mu[i]-rat->nu[i])*pnu;
  }

  free(ars);
  return(0);
}

