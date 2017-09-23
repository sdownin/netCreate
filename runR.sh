#!/bin/bash
#$ -cwd
#$ -S /bin/bash

#export PATH=$PATH:$SGE_O_PATH
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH

#cd $SGE_O_WORKDIR
Rscript cconma_rscript1_windows.R > cconma_rscript1_windows.Rout
#Rscript bench.R > bench.Rout
