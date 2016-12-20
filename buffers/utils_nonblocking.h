#ifndef _UTILS_NONBLOCKING_H
#define _UTILS_NONBLOCKING_H

void generic_exchange_direction_nonblocking(void *field_in, int bytes_per_site, int direction, MPI_Request *inreq, int *counter);

void generic_exchange_nogauge(void *field_in, int bytes_per_site );

#endif
