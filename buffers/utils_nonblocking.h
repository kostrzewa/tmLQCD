#ifndef _UTILS_NONBLOCKING_H
#define _UTILS_NONBLOCKING_H

#ifndef TM_USE_MPI
void generic_exchange_direction_nonblocking(void *field_in, int bytes_per_site, int direction, int *counter);
#else
void generic_exchange_direction_nonblocking(void *field_in, int bytes_per_site, int direction, MPI_Request *inreq, int *counter);
#endif
void generic_exchange_nogauge(void *field_in, int bytes_per_site );

#endif
