#=====================================================================================
#  Author: Aobo Li
#  Contact: liaobo77@gmail.com
#  
#  Last Modified: Sep. 5, 2021
#  
#  * setting files for data batch processing scripts
#=====================================================================================
INPUT_DIR =            "/projectnb/snoplus/SoYoung/klz/root_files2/XeLS_C11p"      # Location of the input .root files
OUT_DIR =    "/projectnb/snoplus/SoYoung/klz/processed-for-kamnet2/XeLS_C11p"      # Location to store the .pickle files
MACRO_DIR = "/projectnb/snoplus/SoYoung/dumpfiles"                                  # A place to store all the generated shell scripts
TIME = "0:10:00"                                                                    # Processing time of each shell script
PROCESSOR = "/project/snoplus/SoYoung/klz/KamNet/data/processing_kamland_new_mc.py" # The processor we'd like to use

# Define the size of each hit maps
COLS = 38
ROWS = COLS

# Define the fiducial volume cutting threshold for event selection
FV_CUT_LOW = 0.0
FV_CUT_HI = 167.0

# Define the energy range for event selection
ELOW = 0.5
EHI  = 5.0

good_hit = True        # If true, use only good PMT hits, otherwise use all PMT hits
only_17inch = False    # If true, use only 17-inch PMTs, otherwise use both 17 and 20 inch pmts
use_charge = False     # If true, register the corresponding charge of each PMT to hit map for each hit, otherwise register 1.0 for each hit
PLOT_HITMAP=False      # Plot flag. If true, plot hit maps of a input event. Note that if this flag is set to True, then the processing script won't process any file.