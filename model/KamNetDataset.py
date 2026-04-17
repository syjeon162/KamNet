#=====================================================================================
#  Author: Aobo Li
#  Contact: liaobo77@gmail.com
#  
#  Last Modified: Aug. 29, 2021
#  
#  * The PyTorch dataset classes for KamNet
#=====================================================================================
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm
import pickle

def readEventsFromFiles(file_list, load_strings, low=2.0, high=3.0):
  event_dict = {el:[] for el in load_strings}
  for file in tqdm(file_list):
    with open(file, 'rb') as f:
      try:
        event = pickle.load(f, encoding='latin1')
        if (event["energy"] > high) or (event["energy"] < low):
          continue
        for load in load_strings:
          event_dict[load].append(event[load])
      except EOFError:
        break
  return event_dict

class KamNetDataset(Dataset):

    def __init__(self, json_name):
        """
        Base class for all KamNet datasets
        """
        self.json_name = json_name

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image = np.zeros(self.image_shape, dtype=np.float32)
        for time_index, time in enumerate(self.trainX[idx]):
            image[time_index] = time.todense()
        return image, self.trainY[idx], self.energy[idx]

    def return_time_channel(self):
        '''
        This method returns the time channel and one hit map dimension of input
        E.g. If it returns (28,38), this means the input has 28 time channel, where
        each channel contains a 38*38 hitmap
        '''
        return (self.__getitem__(0)[0].shape[0], self.image_shape[1])

    def getRandomSamples(self, input, maxNum):
        '''
        This method randomly resamples part of the dataset,
        and returns indices for the downsized sample
        '''
        if input.shape[0] < maxNum:
            return np.arange(input.shape[0])
        sig_samples = np.random.choice(np.arange(input.shape[0]), maxNum, replace=False)
        return sig_samples

    def getSparseNhit(self, sparse_dict):
        '''
        This method gets nhit as a list of given event dict
        It reads out the Nhit directly if Nhit is stored in the dict
        Otherwise it calculates Nhit from the sparse matrices
        '''
        if "Nhit" in sparse_dict.keys():
            return np.array(sparse_dict["Nhit"], dtype=int).flatten()
        else:
            sparsem = np.array(sparse_dict[self.json_name], dtype=object)
            sparse_nhit = []
            for sparsem_i in tqdm(sparsem):
                sparse_nhit.append(np.sum([len(slice.nonzero()[0]) for slice in sparsem_i]))
            return np.array(sparse_nhit)

    def match_nhit(self, sig_dict, bkg_dict, multiplier=1.0):
        '''
        Perform Nhit matching between input sig and output bkg
        '''
        sig_img = np.array(sig_dict[self.json_name], dtype=object)
        bkg_img = np.array(bkg_dict[self.json_name], dtype=object)
        nhit_range = np.arange(0, 2000, 1)
        sig_nhit = np.array(self.getSparseNhit(sig_dict))
        bkg_nhit = np.array(self.getSparseNhit(bkg_dict))
        sig_energy = np.array(sig_dict['energy'], dtype=object)
        bkg_energy = np.array(bkg_dict['energy'], dtype=object)

        sig_list = []
        bkg_list = []

        for (nlow, nhi) in tqdm(zip(list(nhit_range[:-1]), list(nhit_range[1:])),0):
            sig_index = np.where((sig_nhit >= nlow) & (sig_nhit <nhi))[0]
            bkg_index = np.where((bkg_nhit >= nlow) & (bkg_nhit <nhi))[0]
            if (len(sig_index) != 0) and (len(bkg_index) != 0):
                sampled_amount = min(len(sig_index), len(bkg_index))
                sig_list += list(np.random.choice(list(sig_index), sampled_amount, replace=False))
                bkg_list += list(np.random.choice(list(bkg_index), min(len(bkg_index), int(sampled_amount*multiplier)), replace=False))
        # save nhit before and after?

        return sig_img[sig_list], bkg_img[bkg_list], sig_energy[sig_list], bkg_energy[bkg_list]


class KamNetDataset_Nhit(KamNetDataset):

    def __init__(self, sig_img_list, bkg_img_list, json_name, dsize=-1, elow=2.0, ehi=3.0):
        super(KamNetDataset_Nhit, self).__init__(json_name)
        """
        KamNet dataset with Nhit matching. Nhit matching removes Nhit dependency of sig/bkg events
        Used for training the neural network
        elow and ehi indicates the min/max energy of events we'd like to read out
        """
        sig_dict = readEventsFromFiles(sig_img_list, (json_name, 'Nhit','energy','zpos'), low=elow, high=ehi)
        bkg_dict = readEventsFromFiles(bkg_img_list, (json_name, 'Nhit','energy','zpos'), low=elow, high=ehi)

        sig_img, bkg_img, sig_energy, bkg_energy = self.match_nhit(sig_dict, bkg_dict)

        if dsize != -1:
          sig_indices = self.getRandomSamples(sig_img, dsize)
          sig_img = sig_img[sig_indices]
          sig_energy = sig_energy[sig_indices]
          bkg_indices = self.getRandomSamples(bkg_img, dsize)
          bkg_img = bkg_img[bkg_indices]
          bkg_energy = bkg_energy[bkg_indices]

        sig_labels = np.ones(len(sig_img), dtype=np.float32)
        bkg_labels = np.zeros(len(bkg_img), dtype=np.float32)

        self.trainX = np.concatenate((sig_img, bkg_img), axis=0)
        self.trainY = np.concatenate((sig_labels, bkg_labels), axis=0)
        self.energy = np.concatenate((sig_energy, bkg_energy), axis=0)

        self.size = len(sig_img) + len(bkg_img)
        self.image_shape = (trainX.shape[-1], *trainX[0,0].shape)


class KamNetDataset_NonUniform(KamNetDataset):

    def __init__(self, sig_img_list, bkg_img_list, json_name, elow=2.0,ehi=3.0):
        super(KamNetDataset_NonUniform, self).__init__(json_name)
        """
        KamNet dataset which do not require the sig/bkg dataset to follow the same size
        """
        sig_dict = readEventsFromFiles(sig_img_list, (json_name, 'Nhit','energy','zpos'), low=elow, high=ehi)
        bkg_dict = readEventsFromFiles(bkg_img_list, (json_name, 'Nhit','energy','zpos'), low=elow, high=ehi)

        sig_img = np.array(sig_dict[json_name], dtype=object)
        bkg_img = np.array(bkg_dict[json_name], dtype=object)

        sig_indices = self.getRandomSamples(sig_img, dsize)
        sig_img = sig_img[sig_indices]
        bkg_indices = self.getRandomSamples(bkg_img,dsize)
        bkg_img = bkg_img[bkg_indices]

        sig_labels = np.ones(len(sig_img), dtype=np.float32)
        bkg_labels = np.zeros(len(bkg_img), dtype=np.float32)

        self.trainX = np.concatenate((sig_img, bkg_img), axis=0)
        self.size = self.trainX.shape[0]
        self.trainY = np.concatenate((sig_labels, bkg_labels), axis=0)
        self.image_shape = (self.trainX.shape[-1], *self.trainX[0,0].shape)

        sig_energy = np.array(sig_dict["energy"]).flatten()
        bkg_energy = np.array(bkg_dict["energy"]).flatten()
        self.energy = np.concatenate((sig_energy, bkg_energy), axis=0)

    def __getitem__(self, idx):
        image = np.ndarray(self.image_shape, dtype=np.float32)
        for time_index, time in enumerate(self.trainX[idx]):
            image[time_index] = time.todense()
        return image, self.trainY[idx], self.energy[idx]


class KamNetDatasetRep(KamNetDataset):

    def __init__(self, sig_img_list, bkg_image_dict, json_name, dsize = -1, elow=2.0,ehi=3.0):
        super(KamNetDatasetRep, self).__init__(json_name)
        """
        KamNet dataset outputing multiple isotopes for validation purpose
        """

        self.trainX = []
        self.trainY = []
        self.energy = []

        sig_dict = readEventsFromFiles(sig_img_list, (json_name, "Nhit", "energy"), low=elow, high=ehi)
        sig_img = np.array(sig_dict[json_name], dtype=object)
        sig_img_indices = self.getRandomSamples(sig_img, 2000)
        sig_img = sig_img[sig_img_indices]
        
        self.trainX.append(sig_img)
        self.trainY += ["sig"] * len(sig_img)
        sig_energy = np.array(sig_dict["energy"]).flatten()
        self.energy.append(sig_energy[sig_img_indices])

        for bkgn,bkglist in bkg_image_dict.items():
            bkgev = readEventsFromFiles(bkglist, (json_name, "id", "energy"), low=elow, high=ehi)
            sig_img = np.array(bkgev[json_name], dtype=object)
            if len(sig_img) == 0:
                continue
            sig_img_indices = self.getRandomSamples(sig_img, 2000)
            sig_img = sig_img[sig_img_indices]
            self.trainX.append(sig_img)
            self.trainY += [bkgn] * len(sig_img)
            sig_energy = np.array(bkgev["energy"]).flatten()
            self.energy.append(sig_energy[sig_img_indices])

        self.trainX = np.concatenate(self.trainX, axis=0)
        self.trainY = np.array(self.trainY)
        self.energy = np.concatenate(self.energy, axis=0)
        self.image_shape = (self.trainX.shape[-1], *self.trainX[0,0].shape)
        self.size = len(self.trainY)
