#!/bin/bash
#$ -l h_vmem=30G
#$ -N vSetTestA
#$ -cwd
#$ -V

python /home/fortino/DESworkspace/scripts/runGPR.py /home/fortino/vSetTesting/SubsetA -e 361580 -v "Subset A" -r
