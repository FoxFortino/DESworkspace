for i in {0..22}
do
qsub -l h='!node23' -l h_vmem=30G -N zone132g${i} -cwd -V /home/fortino/DESworkspace/scripts/runGPR.py /home/fortino/GPRSolutions/zone132 -z 132 -b g -s ${i} -arc --max 250
qsub -l h='!node23' -l h_vmem=30G -N zone132r${i} -cwd -V /home/fortino/DESworkspace/scripts/runGPR.py /home/fortino/GPRSolutions/zone132 -z 132 -b r -s ${i} -arc --max 250
qsub -l h='!node23' -l h_vmem=30G -N zone132i${i} -cwd -V /home/fortino/DESworkspace/scripts/runGPR.py /home/fortino/GPRSolutions/zone132 -z 132 -b i -s ${i} -arc --max 250
qsub -l h='!node23' -l h_vmem=30G -N zone132z${i} -cwd -V /home/fortino/DESworkspace/scripts/runGPR.py /home/fortino/GPRSolutions/zone132 -z 132 -b z -s ${i} -arc --max 250
qsub -l h='!node23' -l h_vmem=30G -N zone132Y${i} -cwd -V /home/fortino/DESworkspace/scripts/runGPR.py /home/fortino/GPRSolutions/zone132 -z 132 -b Y -s ${i} -arc --max 250
done
