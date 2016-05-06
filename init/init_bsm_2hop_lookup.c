/***********************************************************************
 * Copyright (C) 2016 Bartosz Kostrzewa
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
#include <errno.h>
#include "global.h"
#include "operator/bsm_2hop_dirs.h"
#include "init_bsm_2hop_lookup.h"
#include "fatal_error.h"

int init_bsm_2hop_lookup(const int V) {
  static int bsm_2hop_lookup_initialised = 0;
  
  if( bsm_2hop_lookup_initialised == 0 ){
    g_bsm_2hop_lookup = malloc(V*32*sizeof(int));
    
    if((void*)g_bsm_2hop_lookup == NULL) fatal_error("malloc failed","init_bsm_2hop_lookup");
                                              
    for(int ix = 0; ix < V; ++ix){
      g_bsm_2hop_lookup[32*ix + BSM_2HOP_S_P0  ] = g_iup[ix][0];
      g_bsm_2hop_lookup[32*ix + BSM_2HOP_S_PP0 ] = g_iup[ g_iup[ix][0] ][0];
      g_bsm_2hop_lookup[32*ix + BSM_2HOP_U_0   ] = ix;
      g_bsm_2hop_lookup[32*ix + BSM_2HOP_U_P0  ] = g_iup[ix][0];  
  
      g_bsm_2hop_lookup[32*ix + BSM_2HOP_S_M0  ] = g_idn[ix][0];             
      g_bsm_2hop_lookup[32*ix + BSM_2HOP_S_MM0 ] = g_idn[ g_idn[ix][0] ][0]; 
      g_bsm_2hop_lookup[32*ix + BSM_2HOP_U_M0  ] = g_idn[ix][0];
      g_bsm_2hop_lookup[32*ix + BSM_2HOP_U_MM0 ] = g_idn[ g_idn[ix][0] ][0]; 
  
      g_bsm_2hop_lookup[32*ix + BSM_2HOP_S_P1  ] = g_iup[ix][1];                
      g_bsm_2hop_lookup[32*ix + BSM_2HOP_S_PP1 ] = g_iup[ g_iup[ix][1] ][1];   
      g_bsm_2hop_lookup[32*ix + BSM_2HOP_U_1   ] = ix;                         
      g_bsm_2hop_lookup[32*ix + BSM_2HOP_U_P1  ] = g_iup[ix][1];               
  
      g_bsm_2hop_lookup[32*ix + BSM_2HOP_S_M1  ] = g_idn[ix][1];               
      g_bsm_2hop_lookup[32*ix + BSM_2HOP_S_MM1 ] = g_idn[ g_idn[ix][1] ][1];   
      g_bsm_2hop_lookup[32*ix + BSM_2HOP_U_M1  ] = g_idn[ix][1];              
      g_bsm_2hop_lookup[32*ix + BSM_2HOP_U_MM1 ] = g_idn[ g_idn[ix][1] ][1];  
  
      g_bsm_2hop_lookup[32*ix + BSM_2HOP_S_P2  ] = g_iup[ix][2];                 
      g_bsm_2hop_lookup[32*ix + BSM_2HOP_S_PP2 ] = g_iup[ g_iup[ix][2] ][2];     
      g_bsm_2hop_lookup[32*ix + BSM_2HOP_U_2   ] = ix;                           
      g_bsm_2hop_lookup[32*ix + BSM_2HOP_U_P2  ] = g_iup[ix][2];                 
  
      g_bsm_2hop_lookup[32*ix + BSM_2HOP_S_M2  ] = g_idn[ix][2];                 
      g_bsm_2hop_lookup[32*ix + BSM_2HOP_S_MM2 ] = g_idn[ g_idn[ix][2] ][2];     
      g_bsm_2hop_lookup[32*ix + BSM_2HOP_U_M2  ] = g_idn[ix][2];                 
      g_bsm_2hop_lookup[32*ix + BSM_2HOP_U_MM2 ] = g_idn[ g_idn[ix][2] ][2];    
  
      g_bsm_2hop_lookup[32*ix + BSM_2HOP_S_P3  ] = g_iup[ix][3];                 
      g_bsm_2hop_lookup[32*ix + BSM_2HOP_S_PP3 ] = g_iup[ g_iup[ix][3] ][3];     
      g_bsm_2hop_lookup[32*ix + BSM_2HOP_U_3   ] = ix;                           
      g_bsm_2hop_lookup[32*ix + BSM_2HOP_U_P3  ] = g_iup[ix][3];                 
  
      g_bsm_2hop_lookup[32*ix + BSM_2HOP_S_M3  ] = g_idn[ix][3];                 
      g_bsm_2hop_lookup[32*ix + BSM_2HOP_S_MM3 ] = g_idn[ g_idn[ix][3] ][3];     
      g_bsm_2hop_lookup[32*ix + BSM_2HOP_U_M3  ] = g_idn[ix][3];                 
      g_bsm_2hop_lookup[32*ix + BSM_2HOP_U_MM3 ] = g_idn[ g_idn[ix][3] ][3];    
    }
    bsm_2hop_lookup_initialised = 1;
  }

  return(0);
}

void free_bsm_2hop_lookup() {
  if((void*)g_bsm_2hop_lookup != NULL)
    free(g_bsm_2hop_lookup);
}
