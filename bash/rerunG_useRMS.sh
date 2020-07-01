#!/bin/bash
#$ -l h_vmem=40G
#$ -N useRMS
#$ -e err.err
#$ -o out.out
#$ -cwd
#$ -V


python rerunG_useRMS.py
