#!/bin/bash
#$ -l h_vmem=40G
#$ -N minmaxErr
#$ -e err.err
#$ -o out.out
#$ -cwd
#$ -V


python rerunG_minmaxErr.py
