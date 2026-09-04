# G3SA

G3SA is a GPU-accelerated implementation of BWA-MEM for short-read DNA sequence
alignment. It reproduces bwa-mem2 alignment results on the GPU. Published at
ICS 2025 (DOI: 10.1145/3721145.3729516).

## Requirements

- NVIDIA GPU, CUDA compute capability >= 6.1 (tested on RTX A6000, sm_86)
- VRAM: >= 4 GB for a bacterial genome (E. coli); ~48 GB for the human genome (GRCh38)
- CUDA Toolkit (nvcc), GCC/G++ with C++17 support, pthread, GNU Make

## Dependencies

Bundled under `ext/` and linked at build time (build them first per their own
instructions):

- zlib
- safestringlib
- bwa-mem2 index libraries (libbntseq, libreadindexele) and headers

## Build

```
make all                 # default CU_ARCH=sm_86 (RTX A6000 / 3090)
make all CU_ARCH=sm_89   # RTX 4090
make clean
```

**Note: the build architecture must match the GPU on the machine that runs it; a
wrong `CU_ARCH` fails at runtime with "named symbol not found".**

## Usage

Build the index once per reference, then align:

```
./g3.exe index ref.fasta
./g3.exe mem ref.fasta reads.fastq -o output.sam
```

### Options

| Flag | Description | Default |
| ---- | ----------- | ------- |
| -o | Output SAM path | stdout |
| -g | Number of GPUs | 1 |
| -Z | Batch size (reads per chunk) | 10000 |
| -M | GPU memory pool size (MB) | 1000 |

## Example

Minimal E. coli reproduction:

```
./g3.exe index ecoli.fasta
./g3.exe mem ecoli.fasta ecoli_reads.fastq -o g3.sam
# g3.sam is a standard SAM file and can be compared against bwa-mem2's output:
```

## Limitations

- Single-end reads only
- Read length < 700 bp
- Default BWA-MEM scoring parameters only

## License

GPL-3.0 (see LICENSE). This project incorporates code from:

- **BWA** ([lh3/bwa](https://github.com/lh3/bwa), Heng Li / Genome Research Ltd,
  Broad Institute, Dana-Farber Cancer Institute). The bwa repository is itself
  GPL-3.0-licensed, though a few individual files (e.g. `bwamem.c`, `ksw.c`,
  `bwt.c`) carry MIT-style headers from Heng Li's other projects.
- **BWA-MEM2** (Intel Corporation, Heng Li, Sanchit Misra, Vasimuddin Md),
  MIT-licensed.
- **[minhhpham/bwa](https://github.com/minhhpham/bwa)** (GitHub user minhhpham),
  GPL-3.0-licensed. Substantial portions of this project's GPU-porting layer —
  memory management, string/sort utilities, k-mer hash indexing, and several
  core GPU kernels — are adapted from this fork.
- klib (khash/kseq/ksort/kvec, AttractiveChaos / Heng Li), Intel's
  safestringlib, Yuta Mori's sais-lite, and Cameron Desrochers'
  moodycamel::ConcurrentQueue, bundled under `third_party/` with their
  original licenses preserved.
