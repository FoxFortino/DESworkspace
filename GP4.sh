#!/bin/bash
#$ -l h_vmem=40G
#$ -N DES_GP
#$ -e err.err
#$ -o out.out
#$ -cwd
#$ -V


python runGP4.py


