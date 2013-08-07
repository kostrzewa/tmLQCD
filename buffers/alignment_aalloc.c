#include "alignment.ih"

int allocation_id_ctr = 0;

void *aalloc(size_t const bytes)
{
  void *raw = malloc(sizeof(void*) + ALIGN_BASE + sizeof(meta_data_t) + bytes);

  if (raw == NULL)
    return NULL;

  size_t ptr = (size_t)raw + sizeof(void*) + sizeof(meta_data_t);
  ptr = ((ptr + ALIGN_BASE) & ~ALIGN_BASE);
  ((void**)ptr)[-1] = raw;
  meta_data_t *meta_data = (meta_data_t*)raw;
  meta_data->bytes = bytes;
  meta_data->id = allocation_id_ctr++;
  strncpy(meta_data->note, "NONE", 4);
  return (void*)ptr;
}
