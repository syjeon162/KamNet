'''
script for submitting batch jobs to SCC
'''
import os
import string
import sys
import subprocess

from tools import *

if __name__ == "__main__" :
    # ======================================================================
    # Job Parameters
    TIME = "0:10:00"
    homedir = "/projectnb/snoplus/SoYoung/klz"

    isotope = "XeLS_2nu_1st0p_Xe136"
    input_dir = f"{homedir}/root-files/{isotope}"
    output_dir = f"{homedir}/processed-for-kamnet/{isotope}"
    rows, cols = 38, 38
    elow, ehigh = 0.5, 5.0
    useGoodPMTs = True
    # ======================================================================

    # directories and files to use
    qsub_script = "/project/snoplus/SoYoung/klz/KamNet/scc_scripts/qsub/qsub_template.sh"
    toml_script = "/project/snoplus/SoYoung/klz/KamNet/scc_scripts/toml/settings_template_data.toml"
    pyfile      = "/project/snoplus/SoYoung/klz/KamNet/data/process_kamland_mc.py"
    dumpdir     = "/projectnb/snoplus/SoYoung/dumpfiles"
    finalizeDir(dumpdir)
    checkFile(qsub_script, toml_script, pyfile)
    
    for rfile in naturalSort(getFilesUnderFolder(input_dir)):
        pfilename = f"{getFileName(rfile)}_{elow}-{ehigh}"
        jobname = f"KamNet_dataproc_{pfilename}"
        tomlfilepath = os.path.join(dumpdir, f'{jobname}.toml')
        qsubfilepath = os.path.join(dumpdir, f'{jobname}.sh')

        toml_config = {
            "INPUT"   : rfile,
            "OUTPUT"  : os.path.join(output_dir, f"{pfilename}.pickle"),
            "ROWS"    : rows,
            "COLS"    : cols,
            "ELOW"    : elow,
            "EHIGH"   : ehigh,
            "GOODPMT" : "true" if useGoodPMTs else "false",
        }
        
        qsub_config = {
            "JOBNAME" : jobname,
            "TIME"    : TIME,
            "PYFILE"  : pyfile,
            "OPTIONS" : tomlfilepath,
        }

        with cd(dumpdir):
            # Write toml script
            tomltemplate = string.Template(open(toml_script, 'r').read())
            tomlstring = tomltemplate.substitute(toml_config)
            tomlfile = open(tomlfilepath, 'w')
            tomlfile.write(tomlstring)
            tomlfile.close()

            # Write qsub script
            qsubtemplate = string.Template(open(qsub_script, 'r').read())
            qsubstring = qsubtemplate.substitute(qsub_config)
            qsubfile = open(qsubfilepath, 'w')
            qsubfile.write(qsubstring)
            qsubfile.close()

            # submit qsub
            try:
                command = ['qsub', os.path.join(dumpdir, qsubfilepath)]
                process = subprocess.call(command)
            except:
                raise Exception()