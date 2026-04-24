'''
script for submitting batch jobs to SCC
'''

import os
import string
import sys
import re
import subprocess

from tools import cd, finalizeDir, getFileName, naturalSort

if __name__ == "__main__" :
    qsub_script = "/project/snoplus/SoYoung/klz/KamNet/scc_scripts/qsub_template_pytorch.sh"
    dumpdir     = "/projectnb/snoplus/SoYoung/dumpfiles"
    pyfile      =  "/project/snoplus/SoYoung/klz/KamNet/model/run_KamNet.py"

    finalizeDir(dumpdir)

    elo, ehi = 1.5, 2.5
    
    for bkg_name in ["I130", "Sb118"]:
        jobname = f"260326_2nu1st0p_vs_BGs_{bkg_name}_{elo}_{ehi}"
        options = f"--bkg {bkg_name} --elow {elo} --ehi {ehi}"

        # Job Parameters
        TIME = "1:00:00"

        pyname = getFileName(pyfile)
        print(f'Running {pyname}.py with options {options}')

        with cd(dumpdir):
            # Write qsub script and submit
            scripttemplate = string.Template(open(qsub_script, 'r').read())
            scriptstring = scripttemplate.substitute(
                    JOBNAME=f"{pyname}_{jobname}",
                    TIME=TIME,
                    PYFILE=pyfile,
                    OPTIONS=options)
            scriptfilename = f'{pyname}_{jobname}.sh'
            script = open(os.path.join(dumpdir, scriptfilename), 'w')
            script.write(scriptstring)
            script.close()

            try:
                command = ['qsub', os.path.join(dumpdir, scriptfilename)]
                process = subprocess.call(command)
            except:
                raise Exception()