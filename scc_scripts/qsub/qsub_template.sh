#!/bin/bash -l

#$$ -P snoplus
#$$ -N ${JOBNAME}

#$$ -l h_rt=${TIME}

#$$ -j y
#$$ -V

cd $$TMPDIR

# Unloads all loaded modules
module purge

# load env
module load python3/3.12.4
source /project/snoplus/ROOT6.28/root_install/bin/thisroot.sh

python ${PYFILE} ${OPTIONS}