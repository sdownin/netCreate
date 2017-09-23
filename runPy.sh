#!/bin/bash
#$ -cwd
#$ -S /bin/bash

#export PATH=$PATH:$SGE_O_PATH
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH

#cd $SGE_O_WORKDIR
python cconma_analysis_batch.py > cconma_analysis_output.txt
#Rscript bench.R > bench.Rout
