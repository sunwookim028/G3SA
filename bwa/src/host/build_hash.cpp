// Standalone tool: build .hash file from existing .bwt file.
// Usage: ./build_hash.exe <ref.fasta.bwt>
// Output: writes <ref.fasta.hash> next to the input .bwt file.
// Mirrors the hash-build step in bwtindex.cpp.
// Not part of `make all`; build manually from the repo root with:
//   g++ -std=c++17 -Isrc -Isrc/host -Isrc/types -Isrc/legacy -Isrc/third_party \
//       src/host/build_hash.cpp src/host/hash_kmer.cpp src/legacy/bwt.c \
//       src/legacy/malloc_wrap.c src/legacy/utils.cpp -o build_hash.exe
#include "hash_kmer.h"
#include "datadump.h"
#include <cstring>
#include <cstdio>
#include <cmath>

extern "C" {
#include "bwt.h"
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s <ref.fasta.bwt>\n", argv[0]);
        return 1;
    }
    const char *bwt_path = argv[1];

    fprintf(stderr, "[build_hash] Loading BWT from %s ...\n", bwt_path);
    bwt_t *bwt = bwt_restore_bwt(bwt_path);
    if (!bwt) {
        fprintf(stderr, "ERROR: bwt_restore_bwt failed\n");
        return 1;
    }

    fprintf(stderr, "[build_hash] Building KMER-K=%d hash table ...\n", KMER_K);
    kmers_bucket_t *hashTable = createHashKTable(bwt);
    bwt_destroy(bwt);

    char hash_path[4096];
    strncpy(hash_path, bwt_path, sizeof(hash_path) - 1);
    hash_path[sizeof(hash_path) - 1] = '\0';
    size_t len = strlen(hash_path);
    if (len > 4 && strcmp(hash_path + len - 4, ".bwt") == 0)
        strcpy(hash_path + len - 4, ".hash");
    else
        strncat(hash_path, ".hash", sizeof(hash_path) - len - 1);

    fprintf(stderr, "[build_hash] Writing to %s ...\n", hash_path);
    dumpArray(hashTable, (unsigned long long)pow4(KMER_K), std::string(hash_path));
    free(hashTable);
    fprintf(stderr, "[build_hash] Done.\n");
    return 0;
}
