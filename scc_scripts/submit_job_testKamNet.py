'''
script for submitting batch jobs to SCC
'''
import os
import string
import sys
import subprocess

from tools import cd, finalizeDir, getFileName, checkFile

klzdat = "/projectnb/snoplus/SoYoung/klz"
data_dirs = {
    "I122"     : f"{klzdat}/processed-for-kamnet/file_lists/XeLS_I122_FV157.dat",
    "I124"     : f"{klzdat}/processed-for-kamnet/file_lists/XeLS_I124_FV157.dat",
    "I130"     : f"{klzdat}/processed-for-kamnet/file_lists/XeLS_I130_FV157.dat",
    "K40m"     : f"{klzdat}/processed-for-kamnet/file_lists/XeLS_K40m_FV157.dat",
    "Sb118"    : f"{klzdat}/processed-for-kamnet/file_lists/XeLS_Sb118_FV157.dat",
    "Bi210m"   : f"{klzdat}/processed-for-kamnet/file_lists/XeLS_Bi210m_FV157.dat",
    "SingleGamma-2225keV" : f"{klzdat}/processed-for-kamnet/file_lists/XeLS_SingleGamma-2225keV_FV157.dat",
    "2nu_1st0p_Xe136": f"{klzdat}/processed-for-kamnet/file_lists/XeLS_2nu_1st0p_Xe136-set1-2_FV157.dat",
}

if __name__ == "__main__" :

    for bkg_name in ["I122", "I124", "I130", "Sb118", "K40m", "Bi210m", "SolarB8ES"]:
        # ======================================================================
        # Job Parameters
        TIME = "1:00:00"

        homedir = "/projectnb/snoplus/SoYoung/klz"

        input_files = {
            "2nu_1st0p_Xe136": data_dirs["2nu_1st0p_Xe136"],
            bkg_name         : data_dirs[bkg_name],
        }

        toml_config = {
            "DSIZE"    : 20000,
            "SIGNAL"   : "2nu_1st0p_Xe136",
            "ELOW"     : 1.5,
            "EHIGH"    : 2.3,
            "MAXNFILES": 200,
            "MODEL"    : f"{homedir}/kamnet-results/training_2nu1st0p-vs-2nu/trial1_50k/model/KamNet_model_epoch30.pt",
            "OUTDIR"   : f"{homedir}/kamnet-results/testing_2nu1st0p-vs-2nu/{bkg_name}",
        }
        jobname = f"{bkg_name}_{toml_config['ELOW']}_{toml_config['EHIGH']}"
        # ======================================================================

        # directories and files to use
        qsub_script = "/project/snoplus/SoYoung/klz/KamNet/scc_scripts/qsub/qsub_template_pytorch.sh"
        toml_script = "/project/snoplus/SoYoung/klz/KamNet/scc_scripts/toml/settings_template_test.toml"
        pyfile      = "/project/snoplus/SoYoung/klz/KamNet/model/run_KamNet.py"
        dumpdir     = "/projectnb/snoplus/SoYoung/dumpfiles"
        finalizeDir(dumpdir)
        checkFile(qsub_script, toml_script, pyfile)
        
        tomlfilepath = os.path.join(dumpdir, f'{jobname}.toml')
        qsubfilepath = os.path.join(dumpdir, f'{jobname}.sh')

        qsub_config = {
            "JOBNAME" : f"KamNet_{jobname}",
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