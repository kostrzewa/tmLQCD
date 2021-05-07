/***********************************************************************
 *
 * Copyright (C) 2015 Ferenc Pittler
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

#ifndef _CONTRACTIONS_CURRENTDENSITYEXTENDED_H
#define _CONTRACTIONS_CURRENTDENSITYEXTENDED_H
void wilsonterm_current_density_312ab( bispinor ** propagators, int type_12, int type_ab, _Complex double **res );
void wilsonterm_current_density_412ab( bispinor ** propagators, int type_12, int type_ab, _Complex double **res );
void wilsonterm_current_density_512ab( bispinor ** propagators, int type_12, int type_ab, _Complex double **res );
void wilsonterm_current_density_612ab( bispinor ** propagators, int type_12, int type_ab, _Complex double **res );
#endif
