import os
import re
import argparse

def naturalSort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    natsort_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=natsort_key)

def makeList(outfile, pfiles):
    # Reads out all .pickle file addresses
    inputfiles = naturalSort(pfiles)

    # Generate the .dat pickle list
    datfile = open(outfile,"w")
    for filepath in inputfiles:
        if os.stat(filepath).st_size == 0:
            # Skip file with 0 size
            continue
        datfile.write(filepath + '\n')
    datfile.close()

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('outfile', type=str, action="store", help="Path for output .dat file")
    parser.add_argument('pfiles', type=str, action="store", nargs="+", help="List of pickle files to add to list")
    args = parser.parse_args()

    makeList(args.outfile, args.pfiles)