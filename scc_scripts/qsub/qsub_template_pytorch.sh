#!/bin/bash -l

#$$ -P snoplus
#$$ -N ${JOBNAME}

#$$ -l h_rt=${TIME}
#$$ -l gpus=1
#$$ -l gpu_c=8.0

#$$ -j y
#$$ -V

cd $$TMPDIR

# Unloads all loaded modules
module purge

# load env
module load cuda/12.8
module load miniconda
mamba activate /projectnb/snoplus/conda_envs/pytorch2_11

python ${PYFILE} ${OPTIONS}
