#ifndef FMI_WRAPPER_H
#define FMI_WRAPPER_H

#include "bwa.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct FMI_wrapper FMI_wrapper;

FMI_wrapper* FMI_wrapper_create(const char *prefix);
void FMI_wrapper_load_index(FMI_wrapper *obj, fmindex_t *loadedIndex);

void FMI_wrapper_destroy(FMI_wrapper *obj);

#ifdef __cplusplus
}
#endif

#endif
