#!/bin/bash
#$ -l h_vmem=30G
#$ -N vSetTestE
#$ -cwd
#$ -V

python /home/fortino/DESworkspace/scripts/runGPR.py /home/fortino/vSetTesting/SubsetE -e 361580 -v "Subset E" -r
