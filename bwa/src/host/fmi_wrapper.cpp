#include "fmi_wrapper.h"
#include "fmi_search.h"

struct FMI_wrapper {
	FMI_search* instance;
};

FMI_wrapper* FMI_wrapper_create(const char *filename)
{
	FMI_wrapper *obj = new FMI_wrapper{new FMI_search(filename)};
	return obj;
}

void FMI_wrapper_load_index(FMI_wrapper *obj, fmindex_t *loadedIndex)
{
    obj->instance->load_index();
    loadedIndex->oneHot = obj->instance->one_hot_mask_array;
    loadedIndex->cpOcc = obj->instance->cp_occ;
    loadedIndex->cpOcc2 = obj->instance->cp_occ2;
    loadedIndex->cpOccSize = obj->instance->cp_occ_size;
    loadedIndex->count = obj->instance->count;
    loadedIndex->count2 = obj->instance->count2;
    loadedIndex->firstBase = &(obj->instance->first_base);
    loadedIndex->sentinelIndex = &(obj->instance->sentinel_index);
    loadedIndex->suffixArrayMsByte = obj->instance->sa_ms_byte;
    loadedIndex->suffixArrayLsWord = obj->instance->sa_ls_word;
    loadedIndex->referenceLen = &(obj->instance->reference_seq_len);
    loadedIndex->packedBwt = obj->instance->packed_bwt;
}

void FMI_wrapper_destroy(FMI_wrapper *obj)
{
	delete obj->instance;
	delete obj;
}
