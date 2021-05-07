/***********************************************************************
 *
 * Copyright (C) 2015 Mario Schroeck
 *               2016 Bartosz Kostrzewa
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

#ifndef _D_PSI_BSM2f_H
#define _D_PSI_BSM2f_H

//#include "block.h"

void init_D_psi_BSM2f(void);
void free_D_psi_BSM2f(void);
void D_psi_BSM2f(bispinor * const P, bispinor * const Q);
void D_psi_dagger_BSM2f(bispinor * const P, bispinor * const Q);
void Q2_psi_BSM2f(bispinor * const P, bispinor * const Q);

#endif
