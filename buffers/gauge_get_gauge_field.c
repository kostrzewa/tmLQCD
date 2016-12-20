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
#include "gauge.ih"

/* This routine not only malloc's a field, but immediately aligns it.
   To keep track of the original address to free the field eventually,
   we store that address _before_ the actual buffer.
   The end user should never have to see the alignment after this. */

gauge_field_t get_gauge_field()
{
  gauge_field_t gauge_field;

  if (g_gauge_buffers.free == 0) /* Need to allocate a new buffer */
    allocate_gauge_buffers(1);
  --g_gauge_buffers.free;
  
  gauge_field.field = g_gauge_buffers.reserve[g_gauge_buffers.free];
  g_gauge_buffers.reserve[g_gauge_buffers.free] = NULL;

  return gauge_field;
}

