#!/bin/bash
#$ -l h_vmem=30G
#$ -N vSetTestD
#$ -cwd
#$ -V

python /home/fortino/DESworkspace/scripts/runGPR.py /home/fortino/vSetTesting/SubsetD -e 361580 -v "Subset D" -r
