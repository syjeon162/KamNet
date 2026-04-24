#!/bin/bash -l

#$$ -P snoplus

#$$ -l h_rt=${TIME}

#$$ -j y
#$$ -V

conda deactivate
module load python3/3.12.4
source /project/snoplus/ROOT6.28/root_install/bin/thisroot.sh

python ${PROCESSOR} --input ${INPUT} --outputdir ${OUTPUT} --process_index ${PROCESSING_UNIT}
