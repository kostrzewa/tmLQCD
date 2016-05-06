/***************************************************************************
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
 ****************************************************************************/

#ifndef _BSM_2HOP_DIRS_H
#define _BSM_2HOP_DIRS_H

/* enumeration of indices for the lookup table for gauge and spinor fields in D_psi_BSM2 */

typedef enum bsm_2hop_dirs {
  BSM_2HOP_S_P0 = 0,
  BSM_2HOP_S_PP0,
  BSM_2HOP_U_0,
  BSM_2HOP_U_P0,

  BSM_2HOP_S_M0,
  BSM_2HOP_S_MM0,
  BSM_2HOP_U_M0,
  BSM_2HOP_U_MM0,

  BSM_2HOP_S_P1,
  BSM_2HOP_S_PP1,
  BSM_2HOP_U_1,
  BSM_2HOP_U_P1,

  BSM_2HOP_S_M1,
  BSM_2HOP_S_MM1,
  BSM_2HOP_U_M1,
  BSM_2HOP_U_MM1,

  BSM_2HOP_S_P2,
  BSM_2HOP_S_PP2,
  BSM_2HOP_U_2,
  BSM_2HOP_U_P2,

  BSM_2HOP_S_M2,
  BSM_2HOP_S_MM2,
  BSM_2HOP_U_M2,
  BSM_2HOP_U_MM2,

  BSM_2HOP_S_P3,
  BSM_2HOP_S_PP3,
  BSM_2HOP_U_3,
  BSM_2HOP_U_P3,

  BSM_2HOP_S_M3,
  BSM_2HOP_S_MM3,
  BSM_2HOP_U_M3,
  BSM_2HOP_U_MM3

} bsm_2hop_dirs;

#endif


 
