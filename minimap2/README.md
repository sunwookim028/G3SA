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
cd minimap2
make
```

## Usage

### 1. Basic Usage

```bash
./bin/minimap2-gpu -x map-ont ref.mmi input.fasta output.sam
```

Supported options include:

* `ref.mmi`: pre-built reference index
* `input.fasta`: query reads
* `output.sam`: output SAM file

Note: Not all minimap2 options are supported. This version was developed based on `-ax map-ont` option of original minimap2. (please refer to the [minimap2 manual](https://github.com/lh3/minimap2))

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

### 4. Configuration
The program reads settings from `config.json` (if present) and allows CLI overrides.  
Priority: **defaults → config.json → CLI**.

* Example `config.json`
```bash
{
  "batch_size": 4000, # number of reads in a GPU batch
  "num_reads": 10000, # total number of reads
  "max_seq_len": 10000 # maximum length of a single read
}
```

* Usage
- Run with defaults:  
  ```bash
  ./bin/minimap2-gpu ref.mmi input.fasta 
  ```
- Use a config file:  
  ```bash
  ./bin/minimap2-gpu ref.mmi input.fasta --config config.json
  ```
- Override on CLI:  
  ```bash
  ./bin/minimap2-gpu ref.mmi input.fasta --batch_size 2000 --max_seq_len=5000
  ```

## Running minimap2-gpu

We provide a sample script below : 

```bash
bash ./scripts/run_minimap2_gpu_benchmark.sh
```
The provided parameter settings are optimized to a a6000 GPU.
Please modify the script/parameters to match your dataset and hardware setup.

## Contact

For questions or issues, please contact \[[hyjoo16@snu.ac.kr](mailto:hyjoo16@snu.ac.kr)]


