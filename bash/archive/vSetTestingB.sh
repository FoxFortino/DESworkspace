#!/bin/bash
#$ -l h_vmem=30G
#$ -N vSetTestB
#$ -cwd
#$ -V

python /home/fortino/DESworkspace/scripts/runGPR.py /home/fortino/vSetTesting/SubsetB -e 361580 -v "Subset B" -r
