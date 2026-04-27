'''
script for submitting batch jobs to SCC
'''
import os
import string
import sys
import subprocess

from tools import cd, finalizeDir, getFileName, checkFile

if __name__ == "__main__" :
    # ==========================================================================
    # Job Parameters
    TIME = "6:00:00"

    homedir = "/projectnb/snoplus/SoYoung/klz"
    jobname = "training_2nu1st0p-vs-2nu"
    subname = "trial1"

    input_files = {
        "2nu_1st0p": f"{homedir}/processed-for-kamnet/XeLS_2nu_1st0p_Xe136",
        "Xe136"    : f"{homedir}/processed-for-kamnet/XeLS_2nu_Xe136",
    }

    toml_config = {
        "NUM_EPOCHS": 30,
        "DSIZE"     : 20000,
        "SIGNAL"    : "2nu_1st0p",
        "ELOW"      : 1.5,
        "EHIGH"     : 2.3,
        "MAXNFILES" : 200,
        "OUTDIR"    : f"{homedir}/kamnet-results/{jobname}/{subname}",
    }
    # ==========================================================================

    # directories and files to use
    qsub_script = "/project/snoplus/SoYoung/klz/KamNet/scc_scripts/qsub/qsub_template_pytorch.sh"
    toml_script = "/project/snoplus/SoYoung/klz/KamNet/scc_scripts/toml/settings_template_train.toml"
    pyfile      = "/project/snoplus/SoYoung/klz/KamNet/model/run_KamNet.py"
    dumpdir     = "/projectnb/snoplus/SoYoung/dumpfiles"
    finalizeDir(dumpdir)
    checkFile(qsub_script, toml_script, pyfile)

    tomlfilepath = os.path.join(dumpdir, f'{jobname}_{subname}.toml')
    qsubfilepath = os.path.join(dumpdir, f'{jobname}_{subname}.sh')

    qsub_config = {
        "JOBNAME" : f"KamNet_{jobname}_{subname}",
        "TIME"    : TIME,
        "PYFILE"  : pyfile,
        "OPTIONS" : tomlfilepath,
    }

    with cd(dumpdir):
        # Write toml script
        tomltemplate = string.Template(open(toml_script, 'r').read())
        tomlstring = tomltemplate.substitute(toml_config)
        for isotope, filepath in input_files.items(): # add list of input files
            tomlstring += f'{isotope} = "{filepath}"\n'
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