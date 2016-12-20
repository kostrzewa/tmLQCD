#include "gauge.ih"
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
gauge_field_array_t get_gauge_field_array(unsigned int length)
{
  gauge_field_array_t gauge_field_array;
  gauge_field_array.length = length;
  gauge_field_array.field_array = (gauge_field_t*)calloc(length, sizeof(gauge_field_t));

  if (g_gauge_buffers.free < length) /* Need to allocate more buffers */
    allocate_gauge_buffers(length - g_gauge_buffers.free);

  for (unsigned int ctr = 0; ctr < length; ++ctr)
  {
    --g_gauge_buffers.free;
    gauge_field_array.field_array[ctr].field = g_gauge_buffers.reserve[g_gauge_buffers.free];
    g_gauge_buffers.reserve[g_gauge_buffers.free] = NULL;
  }

  return gauge_field_array;
}

