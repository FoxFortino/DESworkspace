#!/bin/bash
#$ -l h_vmem=40G
#$ -N useCov
#$ -e err.err
#$ -o out.out
#$ -cwd
#$ -V


python rerunG_useCov.py
