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

/*********************************************************************************
 *
 * allocate memory and initialise index lookup table for 2-hop Frezzotti-Rossi BSM
 * operator
 * must be called after geometry()! (geometry_eo.h)
 *
 *********************************************************************************/
#ifdef TM_USE_BSM
#ifndef _INIT_BSM_2HOP_LOOKUP_H
#define _INIT_BSM_2HOP_LOOKUP_H

int init_bsm_2hop_lookup(const int V);
void free_bsm_2hop_lookup();

#endif
#endif
