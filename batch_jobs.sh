#!/bin/bash

mkdir mutation_results/
echo "Mutation learning, 1conv_1maxp"
timeout 2h python3 expand.py --results_file mutation_results/1conv_1maxp.csv
echo "Mutation learning, 1conv_1maxp_nokt"
timeout 2h python3 expand.py --no_knowledge_transfer --results_file mutation_results/1conv_1maxp_nokt.csv
echo "Mutation learning, 2conv_1maxp"
timeout 2h python3 expand.py --two_conv --results_file mutation_results/2conv_1maxp.csv
echo "Mutation learning, 2conv_1maxp_nokt"
timeout 2h python3 expand.py --two_conv --no_knowledge_transfer --results_file mutation_results/2conv_1maxp_nokt.csv
echo "Mutation learning, 2conv_2maxp"
timeout 2h python3 expand.py --two_conv_mp --results_file mutation_results/2conv_2maxp.csv
echo "Mutation learning, 2conv_2maxp_nokt"
timeout 2h python3 expand.py --two_conv_mp --no_knowledge_transfer --results_file mutation_results/2conv_2maxp_nokt.csv
echo "Serial Baseline"
python3 baseline.py
