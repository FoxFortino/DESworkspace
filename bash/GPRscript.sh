for i in {0..10}
do
qsub -l h='!node23' -l h_vmem=30G -N g_RCL_${i} -cwd -V /home/fortino/DESworkspace/scripts/runGPR.py /home/fortino/GPRSolutions/L_BFGS_B_max2 -b g -s ${i} -arc --max 250
qsub -l h='!node23' -l h_vmem=30G -N r_RCL_${i} -cwd -V /home/fortino/DESworkspace/scripts/runGPR.py /home/fortino/GPRSolutions/L_BFGS_B_max2 -b r -s ${i} -arc --max 250
qsub -l h='!node23' -l h_vmem=30G -N i_RCL_${i} -cwd -V /home/fortino/DESworkspace/scripts/runGPR.py /home/fortino/GPRSolutions/L_BFGS_B_max2 -b i -s ${i} -arc --max 250
qsub -l h='!node23' -l h_vmem=30G -N z_RCL_${i} -cwd -V /home/fortino/DESworkspace/scripts/runGPR.py /home/fortino/GPRSolutions/L_BFGS_B_max2 -b z -s ${i} -arc --max 250
done
