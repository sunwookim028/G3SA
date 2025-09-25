# minimap2-gpu

This is the minimap2-based GPU implementation described in the paper:

**"G³SA: A GPU-Accelerated Gold Standard Genomics Library for End-to-End Sequence Alignment"**  
Yeejoo Han, Sunwoo Kim, Seongyeon Park, and Jinho Lee
Presented at International Conference on Supercomputing (ICS), 2025  
[[Paper Link]](https://dl.acm.org/doi/10.1145/3721145.3729516)


## Key Features

* GPU-accelerated long-read alignment pipeline based on Minimap2
* Full CIGAR string output
* Single affine gap penalty 

## Build Instructions

Assuming you are in the root directory of the project:

```bash
cd minimap2_gpu
make
````

You may also use the provided Docker environment from the root directory.

```bash
docker build -t gpu-aligner -f docker/Dockerfile .
docker run --gpus all -it --rm -v $(pwd):/workspace gpu-aligner
cd minimap2_gpu
make
```

## Usage

### 1. Basic Usage

```bash
./bin/minimap2-gpu -x map-ont ref.mmi input.fasta output.sam
```

Supported options include:

* `-x`: preset (e.g., `map-ont`, `map-pb`, etc. refer to: minimap2 link)
* `ref.mmi`: pre-built reference index
* `input.fasta`: query reads
* `output.sam`: output SAM file

Note: Not all minimap2 options are supported. This version was developed based on `-ax map-ont` option of original minimap2.

### 2. Input format

* Query reads

This implementation accepts standard **FASTA** files for query reads.
A small example dataset is provided in the repository under the `data/` directory for testing purposes.

Larger datasets can be downloaded from various sources such as [GenBank](https://www.ncbi.nlm.nih.gov/genbank/) or projects such as [Genome in a Bottle](https://www.nist.gov/programs-projects/genome-bottle). 

* Reference index
  
The index must be pre-built from a reference genome using **minimap2**. You can build the index as follows:

```bash
minimap2 -d ref.mmi ref.fasta
```

For more details on how to build and use minimap2 indexes, please refer to the [official minimap2 repository](https://github.com/lh3/minimap2).


### 3. Output format
The output is a SAM file with full CIGAR string representation of each alignment.


## Reproducing Results

To reproduce the results reported in the paper (e.g., throughput or accuracy evaluation):

```bash
bash ./scripts/run_minimap2_gpu_benchmark.sh
```

Modify the script to match your dataset and hardware setup.

## Contact

For questions or issues, please contact \[[hyjoo16@snu.ac.kr](mailto:hyjoo16@snu.ac.kr)]


