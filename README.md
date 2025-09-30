# G<sup>3</sup>SA

This repository contains the official implementation of the paper:

**"G³SA: A GPU-Accelerated Gold Standard Genomics Library for End-to-End Sequence Alignment"**  
Yeejoo Han, Sunwoo Kim, Seongyeon Park, and Jinho Lee
Presented at International Conference on Supercomputing (ICS), 2025  
[[Paper Link]](https://dl.acm.org/doi/10.1145/3721145.3729516)

We provide two GPU-accelerated versions of the alignment algorithm described in the paper:

- **bwa/**: based on the BWA-MEM algorithm  
- **minimap2/**: based on the minimap2 algorithm

Both versions share the same core GPU design, adapted to different alignment workflows.

## Requirements

- OS: Ubuntu 22.04
- CUDA Toolkit: 12.1

## Directory Structure

```
.
├── bwa/ # GPU implementation based on BWA
│ └── README.md
├── minimap2/ # GPU implementation based on minimap2
│ └── README.md
├── docker/ 
└── README.md 
```

## Getting Started

Each subdirectory contains a README with detailed build and run instructions:

* [bwa/README.md](./bwa/README.md)
* [minimap2/README.md](./minimap2/README.md)

## Contact

For questions or issues, please contact \[[hyjoo16@snu.ac.kr](mailto:hyjoo16@snu.ac.kr)] or \[[sk3463@cornell.edu](mailto:sk3463@cornell.edu)]
