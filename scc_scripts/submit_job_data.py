#=====================================================================================
#  Author: Aobo Li
#  Contact: liaobo77@gmail.com
#  
#  Last Modified: Sep. 5, 2021
#  
#  * Batch processing script for KamNet
#  * Converting the 
#=====================================================================================
#!/usr/bin/python
import json
import time
import datetime
import sys
import argparse
import os
import re
import string
import signal
import subprocess
import shutil
import numpy as np
from settings import OUT_DIR, INPUT_DIR, MACRO_DIR, TIME, PROCESSOR

def finalizeDir(*paths):
    '''
    Make directories if they don't exist
    '''
    for path in paths:
        try:
            os.makedirs(path)
        except:
            pass

def getFolderName(path):
    '''
    Get name of lowest level folder of path
    '''
    if os.path.isfile(path):
        path = os.path.dirname(path)
    return os.path.basename(os.path.normpath(path))

def main(argv):
   # Setting the output directory if it does not exist.
   finalizeDir(OUT_DIR, MACRO_DIR)

   inputfiles = []
   # Reading out isotopes from the input directory, and add their addresses into a list
   inputfiles += [(ifile) for ifile in os.listdir(INPUT_DIR) if ".root" in ifile]

   foldername = getFolderName(INPUT_DIR)

   for rootfile in inputfiles:
      # Loading the template to run on the Boston University SCC cluster batch queue
      # In order to run on your cluster batch system, please modify this .sh template and its inputs
      macrotemplate = string.Template(open('process_kamland.sh', 'r').read())
      with cd(MACRO_DIR):
         outputstring = str(OUT_DIR)
         timestring = str(TIME)
         inputstring = os.path.join(INPUT_DIR, rootfile)
         macrostring = macrotemplate.substitute(TIME=timestring, INPUT=inputstring, OUTPUT=outputstring, PROCESSOR=PROCESSOR, PROCESSING_UNIT=-1)
         macrofilename = f'shell_{foldername}_{rootfile}.sh'
         macro = open(macrofilename,'w')
         macro.write(macrostring)
         macro.close()
         print(os.path.join(MACRO_DIR, macrofilename))
         try:
            command = ['qsub', macrofilename]
            process = subprocess.call(command)
         except Exception as error:
            return 0
   return 1


class cd:
   '''
   Context manager for changing the current working directory
   '''
   def __init__(self, newPath):
      self.newPath = newPath

   def __enter__(self):
      self.savedPath = os.getcwd()
      os.chdir(self.newPath)

   def __exit__(self, etype, value, traceback):
      os.chdir(self.savedPath)

if __name__=="__main__":

   print(sys.exit(main(sys.argv[1:])))

