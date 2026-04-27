#=====================================================================================
#  Author: Aobo Li
#  Contact: liaobo77@gmail.com
#  
#  Last Modified:
#     Apr. 26th, 2026, by So Young Jeon (jeonsy@bu.edu)
#  
#  * The PyTorch dataset classes for KamNet
#=====================================================================================
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.data import Dataset

def loadPickledItems(file):
    '''
    Generator for pickled items in pickle file
    '''
    with open(file, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break

def readEventsFromFiles(files_dict, vars_to_load, var_for_cut="energy", low=2.0, high=3.0):
    '''
    Read in events from pickle files; returns pandas dataframe
    * vars_to_load : names of variables to load (array of str)
    * var_for_cut : name of variables to use for selection (str)
    * low, high : lower/higher bound of selection (int/float)
    '''
    event_dict = {el:[] for el in vars_to_load}
    event_dict['isotope'] = []

    for isotope, file_list in files_dict.items():
        print(f" * Loading data for {isotope}")

        for file in tqdm(file_list):
            for event in loadPickledItems(file):
                if (event[var_for_cut] > high) or (event[var_for_cut] < low):
                    continue

                for var in vars_to_load:
                    if var != "isotope":
                        event_dict[var].append(event[var])
                event_dict['isotope'].append(isotope)

    return pd.DataFrame(event_dict)

class KamNetDataset(Dataset):

    def __init__(self, files_dict, signal_isotope, vars_to_output=None, elow=2.0, ehigh=3.0):
        '''
        Base class for KamNet datasets
        '''
        if vars_to_output:
            self.vars_to_output = vars_to_output
        else:
            self.vars_to_output = []

        vars_to_load = np.unique(["event","Nhit","energy"] + list(self.vars_to_output))

        # read in data as panda dataframe
        self.data = readEventsFromFiles(files_dict, vars_to_load, low=elow, high=ehigh)
        # convert images to ndarray
        self.data['event'] = self.data['event'].apply(lambda x: np.array(x, dtype=object))
        # Add labels to events; 1 if signal, 0 if background
        self.data['label'] = np.where(self.data.isotope == signal_isotope, 1, 0)

        # something like (T, H, W)
        example_image = self.data['event'][0] # list of sparse matrices
        example_sparse_matrix = example_image[0]
        self.image_shape = (example_image.shape[0], *example_sparse_matrix.shape)
        print(f"Image Shape: {self.image_shape}")

    def __len__(self):
        return len(self.data.index)

    def __getitem__(self, idx):
        # un-sparse matrices
        image = np.zeros(self.image_shape, dtype=np.float32)
        for time_index, time in enumerate(self.data.iloc[idx]['event']):
            image[time_index] = time.todense()
        # aggregate other variables to save
        other_vars = self.data[self.vars_to_output].iloc[idx].to_dict()
        return image, self.data.iloc[idx]['label'], other_vars

    def printSize(self):
        '''
        Print dataset size information for each isotope
        '''
        all_isotopes = self.data.isotope.unique()
        name_len = max([len(i) for i in all_isotopes])
        isotope_groups = self.data.groupby('isotope', sort=False).size().reset_index(name='counts')
        print(isotope_groups)

    def downsize(self, dsize):
        '''
        downsize each dataset (by isotope) to maximum dsize
        '''
        print(f" * Downsizing datasets to dsize={dsize}")
        shuffled = self.data.sample(frac=1)
        downsized = shuffled.groupby('isotope', sort=False).head(dsize)
        self.data = downsized.reset_index(drop=True)
        self.printSize()
    
    def matchSBNhit(self):
        '''
        Perform Nhit matching between signal and all backgrounds
        '''
        print(" * Nhit Matching for signal and background")
        grouped = self.data.groupby(['label','Nhit'], sort=False)
        all_matched_groups = []
        for nhit in self.data.Nhit.unique():
            try:
                len_bkg = len(grouped.get_group((0, nhit)).index)
                len_sig = len(grouped.get_group((1, nhit)).index)
            except KeyError:
                continue
            max_sample = min(len_bkg, len_sig)
            for label in [0, 1]:
                all_matched_groups.append(grouped.get_group((label, nhit)).sample(max_sample))
        self.data = pd.concat(all_matched_groups, ignore_index=True)
        self.printSize()
    
    def matchIsotopeNhit(self):
        '''
        Perform Nhit matching between all isotopes in dataset
        This is identical to matchSBNhit() if there are only two isotopes in the dataset,
        where one is signal, one is background.
        '''
        print(">>> Nhit Matching for each isotope...")
        grouped = self.data.groupby(['isotope','Nhit'], sort=False)
        all_isotopes = self.data.isotope.unique()
        all_matched_groups = []
        for nhit in self.data.Nhit.unique():
            len_per_isotope = []
            try:
                for isotope in all_isotopes:
                    len_per_isotope.append(len(grouped.get_group((isotope, nhit)).index))
            except KeyError:
                continue
            max_sample = min(len_per_isotope)
            for isotope in  all_isotopes:
                all_matched_groups.append(grouped.get_group((isotope, nhit)).sample(max_sample))
        self.data = pd.concat(all_matched_groups, ignore_index=True)
        self.printSize()

    def getInputDimension(self):
        '''
        This method returns the time channel and one hit map dimension of input
        E.g. If it returns (28,38), this means the input has 28 time channel, where
        each channel contains a 38*38 hitmap
        '''
        return (self.image_shape[0], self.image_shape[1])