
import os
import re

def naturalSort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    natsort_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=natsort_key)
    
def finalizeDir(*paths):
    '''
    Make directories if they don't exist
    '''
    for path in paths:
        try:
            os.makedirs(path)
        except:
            pass

def checkFile(*paths):
    '''
    Check that files exist
    '''
    for path in paths:
        if not os.path.isfile(path):
            raise Exception(f"{path} does not exist.")
    return

def getFileName(path):
    '''
    Get name of file, without file extension or parent folder
    '''
    namepieces = path.split("/")[-1].split(".")
    filename = ".".join(namepieces[:-1])
    return filename

def getFolderName(path):
    '''
    Get name of lowest level folder of path
    '''
    if os.path.isfile(path):
        path = os.path.dirname(path)
    return os.path.basename(os.path.normpath(path))

def getFilesUnderFolder(path, filetype=""):
    '''
    Get list of files under folder
    Will only list files of filetype if specified
    (pass string without leading ".", ex. "root", "pkl")
    '''
    if filetype:
        l = len(filetype)
        return [os.path.join(path,f) for f in os.listdir(path)
                if os.path.isfile(os.path.join(path,f))
                and f[-(l+1):] == f".{filetype}"]
    else:
        return [os.path.join(path,f) for f in os.listdir(path)
                if os.path.isfile(os.path.join(path,f))]
    
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