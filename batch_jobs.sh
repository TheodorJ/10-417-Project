#!/bin/bash

- Mutation learning, 1 convolution & 1 maxpool layer
- Mutation learning, 1 convolution & 1 max pool layer, no knowledge transfer
- Mutation learning, 2 convolution & 1 maxpool layer
- Mutation learning, 2 convolution & 2 maxpool layer

mkdir mutation_results/
timeout 2h python3 expand.py --results_file mutation_results/1conv_1maxp.csv
timeout 2h python3 expand.py --no_knowledge_transfer --results_file mutation_results/1conv_1maxp_nokt.csv
timeout 2h python3 expand.py --two_conv --results_file mutation_results/2conv_1maxp.csv
timeout 2h python3 expand.py --two_conv --no_knowledge_transfer --results_file mutation_results/2conv_1maxp_nokt.csv
timeout 2h python3 expand.py --two_conv_mp --results_file mutation_results/2conv_2maxp_.csv
timeout 2h python3 expand.py --two_conv_mp --no_knowledge_transfer --results_file mutation_results/2conv_2maxp_nokt.csv
python3 baseline.py
