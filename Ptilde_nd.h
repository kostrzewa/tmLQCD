/***********************************************************************
 * Copyright (C) 2006 Thomas Chiarappa
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

#ifndef _PTILDE_ND_H
#define _PTILDE_ND_H


double func_tilde(double u, double exponent);

void Ptilde_cheb_coefs(double a, double b, double dd[], int n, double exponent);

void Poly_tilde_ND(spinor *R_s, spinor *R_c, double *dd, int n, spinor *S_s, spinor *S_c);

double chebtilde_eval(int M, double *dd, double s);

void degree_of_Ptilde();

#endif
