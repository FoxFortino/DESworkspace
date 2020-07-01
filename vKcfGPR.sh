#!/bin/bash
#$ -l h_vmem=40G
#$ -N curlfreeGPR
#$ -cwd
#$ -V

python vKcfGPR.py
