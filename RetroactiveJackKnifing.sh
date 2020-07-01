#!/bin/bash
#$ -l h_vmem=40G
#$ -N JackKnifing
#$ -cwd
#$ -V

python RetroactiveJackKnifing.py
