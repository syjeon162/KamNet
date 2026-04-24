#!/bin/bash -l

#$$ -P snoplus
#$$ -N ${JOBNAME}

#$$ -l h_rt=${TIME}
#$$ -l gpus=1
#$$ -l gpu_c=8.0

#$$ -j y
#$$ -V

cd $$TMPDIR

module unload python3
module load miniconda
mamba activate pytorch2.8
module load cuda/12.8

python ${PYFILE} ${OPTIONS}
