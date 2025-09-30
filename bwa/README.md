# BWA-MEM
This is the BWA-MEM pipeline GPU implementation described in the paper:

**"G³SA: A GPU-Accelerated Gold Standard Genomics Library for End-to-End Sequence Alignment"**  
Yeejoo Han, Sunwoo Kim, Seongyeon Park, and Jinho Lee
Presented at International Conference on Supercomputing (ICS), 2025  
[[Paper Link]](https://dl.acm.org/doi/10.1145/3721145.3729516)

## Key Features
* GPU-accelerated short-read (<700bp) BWA-MEM alignment
* Aligned position & CIGAR string output

## Build Instructions
Assuming you are in the root directory of the project:
```bash
cd bwa
make all
```

## Usage
### 1. Basic Usage
```bash
./g3.exe index ref.fasta
./g3.exe mem ref.fasta input.fasta -o output.sam
```

Supported options include:

* `ref.fasta`: path to the reference file where other index files reside as well
* `input.fasta`: query reads
* `output.sam`: output file

Note: Not all bwa-mem options are supported. This is a research code, targetting the default option of original bwa-mem. Please reach out to discuss something else. (original [bwa-mem manual](https://github.com/lh3/bwa))

### 2. Input format
* Query reads and reference

This implementation accepts FASTA or FASTQ files for query reads.
Small example inputs are provided in the repository under the `test/` directory for testing purposes.
```bash
test/run.sh
```

Larger datasets can be downloaded from various sources such as [GenBank](https://www.ncbi.nlm.nih.gov/genbank/) or projects such as [Genome in a Bottle](https://www.nist.gov/programs-projects/genome-bottle). 

### 3. Output format
The output is a SAM file with CIGAR string representation of each alignment.

### 4. Reproducing
Datasets used in our paper can be downloaded from SRA, then they can be aligned using the same command from above. Modify the script to match your dataset and hardware setup.

## Acknowledgement
Many designs are inspired or inherited from bwa, bwa-mem2, and bwa-mem-gpu projects.

## Contact
For questions or issues, please contact \[[sk3463@cornell.edu](mailto:sk3463@cornell.edu)]
