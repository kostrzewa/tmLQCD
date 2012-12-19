/***********************************************************************
 *
 * Copyright (C) 2001 Martin Hasebusch
 *               2002,2003,2004,2005,2006,2007,2008,2012 Carsten Urbach
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
#include <time.h>
#include "global.h"
#include "su3.h"
#include "su3adj.h"
#include "su3spinor.h"
#include "monomial/monomial.h"
#include "xchange/xchange.h"
#include "operator/clover_leaf.h"
#include "read_input.h"
#include "hamiltonian_field.h"
#include "update_momenta.h"
#include "gettime.h"

#include <buffers/utils.h>

#include <dirty_shameful_business.h>

/* Updates the momenta: equation 16 of Gottlieb */
void update_momenta(int * mnllist, double step, const int no, hamiltonian_field_t * const hf)
{
  double atime=0., etime=0.;
  int *relevant_smearings = malloc(no_smearings_monomial * sizeof(int));
  int no_relevant_smearings = 0;
  
  for(int k = 0; k < no; k++)
  {
    int current_idx = 0;
    while ((current_idx < no_relevant_smearings) && (monomial_list[ mnllist[k] ].smearing != relevant_smearings[current_idx]))
      ++current_idx;
    if (current_idx == no_relevant_smearings)
    {
      relevant_smearings[current_idx] = monomial_list[ mnllist[k] ].smearing;
      ++no_relevant_smearings;
    }
  }
  
  //adjoint_field_t tmp_derivative = get_adjoint_field();
  
  zero_adjoint_field(&df);
  //zero_adjoint_field(&tmp_derivative);

  //ohnohack_remap_df0(tmp_derivative); /* FIXME Such that we can aggregate results per smearing type. */
  for (int s_ctr = 0; s_ctr < no_relevant_smearings; ++s_ctr)
  {
    int s_type = relevant_smearings[s_ctr];
    
    smear(smearing_control_monomial[s_type], g_gf);
    ohnohack_remap_g_gauge_field(smearing_control_monomial[s_type]->result);
    g_update_gauge_energy=1;
    g_update_rectangle_energy=1;

    for(int k = 0; k < no; k++)
    {
      if (monomial_list[ mnllist[k] ].smearing == s_type)
      {
        
        if(monomial_list[ mnllist[k] ].derivativefunction != NULL)
        {
#ifdef MPI
          atime = MPI_Wtime();
#else
          atime = (double)clock()/(double)(CLOCKS_PER_SEC);
#endif
          monomial_list[ mnllist[k] ].derivativefunction(mnllist[k], hf);

#ifdef MPI
          etime = MPI_Wtime();
#else
          etime = (double)clock()/(double)(CLOCKS_PER_SEC);
#endif
        }
      }
    }
    smear_forces(smearing_control_monomial[s_type], df);
  
    int x = 1, mu = 1;
    double *ar_df = (double*)&df[x][mu];
    double *ar_ne = (double*)&(smearing_control_monomial[s_type]->force_result[x][mu]);
    fprintf(stderr, "[DEBUG] Comparison of force calculation at [%d][%d]!\n",x,mu);
    fprintf(stderr, "        Involves monomials: ");
    for (int monnum = 0; monnum < no; ++monnum)
      fprintf(stderr, "%s ", monomial_list[ mnllist[monnum] ].name);
    fprintf(stderr, "\n");
    fprintf(stderr, "   Numerical total force <-> smear forces\n");
    for (int component = 0; component < 8; ++component)
      fprintf(stderr, "    [%d]  %+14.12f <-> %+14.12f\n", component, ar_df[component], ar_ne[component]); //*/


    //for(int i = 0; i < (VOLUMEPLUSRAND + g_dbw2rand); ++i)
    //{ 
    //  for(int mu = 0; mu < 4; ++mu)
    //  {
    //    _sub_su3adj(df[i][mu], smearing_control_monomial[s_type]->force_result[i][mu]);
    //  }
    //}
  }
  //ohnohack_remap_df0(df);
  ohnohack_remap_g_gauge_field(g_gf);

#ifdef MPI
  xchange_deri(hf->derivative);
#endif

  /* Debugging code inserted here 
//  su3adj num_eval;
//  calculate_forces_numerically(&num_eval, mnllist, no);
  double *ar_df = (double*)&df[0][0];
  double *ar_ne = (double*)&(smearing_control_monomial[s_type]->force_result[0][0]);
//  fprintf(stderr, "[DEBUG] Comparison of force calculation!\n");
//  fprintf(stderr, "        Involves monomial types: ");
  for (int monnum = 0; monnum < no; ++monnum)
    fprintf(stderr, "%d ", monomial_list[ mnllist[monnum] ].name);
  fprintf(stderr, "\n");
//  fprintf(stderr, "        Analytical vs. numerical\n");
  for (int component = 0; component < 8; ++component)
    fprintf(stderr, "    [%d]  %+14.12f <-> %+14.12f\n", component, ar_df[component], ar_ne[component]); //*/

  //exit(0);

#ifdef OMP
#pragma omp parallel for
#endif
  for(int i = 0; i < VOLUME; i++) {
    for(int mu = 0; mu < 4; mu++) {
      /* the minus comes from an extra minus in trace_lambda */
      _su3adj_minus_const_times_su3adj(hf->momenta[i][mu], step, hf->derivative[i][mu]); 
    }
  }
  //return_adjoint_field(&tmp_derivative);
  free(relevant_smearings);
  return;
}

