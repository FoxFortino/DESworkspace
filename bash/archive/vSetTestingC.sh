#!/bin/bash
#$ -l h_vmem=30G
#$ -N vSetTestC
#$ -cwd
#$ -V

python /home/fortino/DESworkspace/scripts/runGPR.py /home/fortino/vSetTesting/SubsetC -e 361580 -v "Subset C" -r
