import numpy as np
import os
import argparse
import time
import math
import random
from sklearn.metrics import roc_auc_score, roc_curve
import torch.nn as nn
import torch.optim as optim
from torch.nn.parameter import Parameter
from torch.nn import init
from s2cnn import SO3Convolution
from s2cnn import S2Convolution
from s2cnn import so3_integrate
from s2cnn import so3_near_identity_grid, so3_equatorial_grid
from s2cnn import s2_near_identity_grid, s2_equatorial_grid
import torch.nn.functional as F
import torch
import torch.utils.data as data_utils
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import pickle
import numpy as np
import copy
from torch.autograd import Variable
from scipy import sparse
from torch.amp import autocast
from tqdm import tqdm

from AttentionConvLSTM import ConvLSTM
from KamNetDataset import KamNetDataset, KamNetDataset_Nhit, KamNetDataset_NonUniform, KamNetDatasetRep
from settings import SEED, NUM_EPOCHS, BATCH_SIZE, KAMNET_PARAMS, LEARNING_RATE, DSIZE
from tool import get_roc, get_rej, roc_nhit, cd

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(DEVICE)


if SEED:
    '''
    Setting reproducability. If SEED=True, then training the neural network with
    the same configuration will result in exactly the same output
    '''
    manualSeed = 7

    np.random.seed(manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)
    # if you are using GPU
    torch.cuda.manual_seed(manualSeed)
    torch.cuda.manual_seed_all(manualSeed)


    torch.backends.cudnn.enabled = False 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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

def load_data(batch_size, signal_dir, bkg_dir_dict, elow, ehi, file_upperlim):
    '''
    Load datasets from various pickle list
    '''
    json_name = "event"

    data_list = getFilesUnderFolder(signal_dir, "pickle")
    bkg_dict = {}
    for bkgname, bkg_dir in bkg_dir_dict.items():
        bkg_dict[bkgname] = getFilesUnderFolder(bkg_dir, "pickle")[:file_upperlim]
    dataset = KamNetDatasetRep(data_list[:file_upperlim], bkg_dict,
        str(json_name), dsize=-1, elow=elow, ehi=ehi)
    test_loader = data_utils.DataLoader(dataset, batch_size=batch_size, drop_last=False)

    return test_loader, dataset.return_time_channel()

class KamNet(nn.Module):

    def __init__(self, time_channel):
        super(KamNet, self).__init__()

        param_dict = KAMNET_PARAMS  # Store the hyperparameters for KamNet

        # Initialize the grid for spherical CNN
        grid_dict = {'s2_eq': s2_equatorial_grid, 's2_ni': s2_near_identity_grid, "so3_eq":so3_equatorial_grid, 'so3_ni':so3_near_identity_grid}
        s2_grid_type = param_dict["s2gridtype"]
        so3_grid_type = param_dict["so3gridtype"]
        grid_s2 = grid_dict[s2_grid_type]()
        grid_so3 = grid_dict[so3_grid_type]()

        self.ftype = param_dict["ftype"]

        # Number of neurons in spherical CNN
        s2_1  = param_dict["s2_1"]
        so3_2 = param_dict["so3_2"]
        so3_3 = param_dict["so3_3"]
        so3_4 = param_dict["so3_4"]

        # Number of neurons in fully connected NN
        fc1 = int(param_dict["fc_max"])
        fc2 = int(param_dict["fc_max"] * 0.8)
        fc3 = int(param_dict["fc_max"] * 0.4)
        fc4 = int(param_dict["fc_max"] * 0.2)
        fc5 = int(param_dict["fc_max"] * 0.05)

        do1r = param_dict["do"]
        do2r = param_dict["do"]
        do3r = param_dict["do"]
        do4r = param_dict["do"]
        do5r = param_dict["do"]

        do1r = min(max(do1r,0.0),1.0)
        do2r = min(max(do2r,0.0),1.0)
        do3r = min(max(do3r,0.0),1.0)
        do4r = min(max(do4r,0.0),1.0)
        do5r = min(max(do5r,0.0),1.0)

        # Number of neurons in AttentionConvLSTM
        s1 = param_dict["s1"]
        s2 = param_dict["s2"]

        # Last output of spherical CNN
        last_entry = so3_4

        # Last output of fully connected NN
        last_fc_entry = fc5

        # The spherical CNN bandwidth
        last_bw = int(param_dict["last_bw"])
        bw = np.linspace(int(time_channel[1]/2), last_bw, 5).astype(int)

        #. Spherical CNN part of KamNet
        self.conv1 = S2Convolution(
            nfeature_in=s2,
            nfeature_out=s2_1,
            b_in=bw[0],
            b_out=bw[1],
            grid=grid_s2)

        self.conv2 = SO3Convolution(
            nfeature_in=s2_1,
            nfeature_out=so3_2,
            b_in=bw[1],
            b_out=bw[2],
            grid=grid_so3)

        self.conv3 = SO3Convolution(
            nfeature_in=so3_2,
            nfeature_out=so3_3,
            b_in=bw[2],
            b_out=bw[3],
            grid=grid_so3)

        self.conv4 = SO3Convolution(
            nfeature_in=so3_3,
            nfeature_out=so3_4,
            b_in=bw[3],
            b_out=bw[4],
            grid=grid_so3)

        #. AttentionConvLSTM part of KamNet
        self.convlstm1=ConvLSTM(1, [s1,s2], [(param_dict["first_filter"],param_dict["first_filter"]),(param_dict["second_filter"],param_dict["second_filter"])],2, time_channel,batch_first=True,fill_value=0.1)

        if self.ftype == "SO3I":
            # This means integrating the last spherical CNN output using the Haar measure as provided in the paper
            self.fc_layer = nn.Linear(so3_4, fc1)
        else:
            # This means flattening the last spherical CNN output into a 1D vector (batch_size,flattened_dimension)
            self.fc_layer = nn.Linear(so3_4*(2*last_bw)**3, fc1)

        # Fully connected part of KamNet
        self.fc_layer_2 = nn.Linear(fc1, fc2)
        self.fc_layer_3 = nn.Linear(fc2, fc3)
        self.fc_layer_4 = nn.Linear(fc3, fc4)
        self.fc_layer_5 = nn.Linear(fc4, fc5)

        self.norm_layer_3d_1 = nn.BatchNorm3d(s2_1)
        self.norm_layer_3d_2 = nn.BatchNorm3d(so3_2)
        self.norm_layer_3d_3 = nn.BatchNorm3d(so3_3)
        self.norm_layer_3d_4 = nn.BatchNorm3d(so3_4)


        self.norm_1d_1 = nn.BatchNorm1d(fc1)
        self.norm_1d_2 = nn.BatchNorm1d(fc2)
        self.norm_1d_3 = nn.BatchNorm1d(fc3)
        self.norm_1d_4 = nn.BatchNorm1d(fc4)
        self.norm_1d_5 = nn.BatchNorm1d(fc5)
        self.norm_1d_6 = nn.BatchNorm1d(1)

        self.fc_layer_6 = nn.Linear(fc5, 1)

        self.do1 = nn.Dropout(do1r)
        self.do2 = nn.Dropout(do2r)
        self.do3 = nn.Dropout(do3r)
        self.do4 = nn.Dropout(do4r)
        self.do5 = nn.Dropout(do5r)

        self.sdo1 = nn.Dropout(param_dict["sdo"])
        self.sdo2 = nn.Dropout(param_dict["sdo"])
        self.sdo3 = nn.Dropout(param_dict["sdo"])
        self.sdo4 = nn.Dropout(param_dict["sdo"])


    def forward(self, x):
        x = x.unsqueeze(2)
        with autocast("cuda"):
            x = self.convlstm1(x)

        x = self.conv1(x)
        x = self.norm_layer_3d_1(x)
        x = torch.relu(x)
        x = self.sdo1(x)

        x = self.conv2(x)
        x = self.norm_layer_3d_2(x)
        x = torch.relu(x)
        x = self.sdo2(x)

        x = self.conv3(x)
        x = self.norm_layer_3d_3(x)
        x = torch.relu(x)
        x = self.sdo3(x)


        x = self.conv4(x)
        x = self.norm_layer_3d_4(x)
        x = torch.relu(x)
        x = self.sdo4(x)

        if self.ftype == "SO3I":
            x = so3_integrate(x)
        else:
            x = x.view(x.size(0),-1)
        with autocast("cuda"):
            x = self.fc_layer(x)
            x = self.norm_1d_1(x)
            x = torch.relu(x)
            x = self.do1(x)

            x = self.fc_layer_2(x)
            x = self.norm_1d_2(x)
            x = torch.relu(x)
            x = self.do2(x)

            x = self.fc_layer_3(x)
            x = self.norm_1d_3(x)
            x = torch.relu(x)
            x = self.do3(x)

            x = self.fc_layer_4(x)
            x = self.norm_1d_4(x)
            x = torch.relu(x)
            x = self.do4(x)

            x = self.fc_layer_5(x)
            x = self.norm_1d_5(x)
            x = torch.relu(x)
            x = self.do5(x)

            x = self.fc_layer_6(x)

        return x
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # parser.add_argument('--sig', type=str)
    parser.add_argument('--bkg', type=str)
    # parser.add_argument('--pt', type=str)
    parser.add_argument('--elow', type=float)
    parser.add_argument('--ehi', type=float)
    # parser.add_argument('--outdir', type=str)
    # parser.add_argument('--filemax', type=int)
    args = parser.parse_args()

    '''
    Training KamNet
    '''

    ##### SETTINGS #####
    pt_file = "/projectnb/snoplus/SoYoung/klz/kamnet-training/260319_2nu1st0p_vs_2nu_goodPMTs_trial2/KamNet_epoch18.pt"
    sig_dir = "/projectnb/snoplus/SoYoung/klz/processed-for-kamnet/XeLS_2nu_1st0p_Xe136_goodPMTs"
    elow = args.elow
    ehi = args.ehi
    filemax = 200
    jobname = "test"
    outdir = f"/projectnb/snoplus/SoYoung/dumpdir/{jobname}"
    bkg_dir_dict = {
        "Xe136"    : "/projectnb/snoplus/SoYoung/klz/processed-for-kamnet/XeLS_2nu_Xe136_goodPMTs",
        "Bi210m"   : "/projectnb/snoplus/SoYoung/klz/processed-for-kamnet/XeLS_Bi210m",
        "C11p"     : "/projectnb/snoplus/SoYoung/klz/processed-for-kamnet/XeLS_C11p",
        "Kr85m"    : "/projectnb/snoplus/SoYoung/klz/processed-for-kamnet/XeLS_Kr85m",
        "SolarB8ES": "/projectnb/snoplus/SoYoung/klz/processed-for-kamnet/XeLS_SolarB8ES",
        "Xe137m"   : "/projectnb/snoplus/SoYoung/klz/processed-for-kamnet/XeLS_Xe137m",
        "Cs136m"   : "/projectnb/snoplus/SoYoung/klz/processed-for-kamnet/XeLS_Cs136m",

        "I122"     : "/projectnb/snoplus/SoYoung/klz/processed-for-kamnet2/XeLS_I122",
        "I124"     : "/projectnb/snoplus/SoYoung/klz/processed-for-kamnet2/XeLS_I124",
        "I130"     : "/projectnb/snoplus/SoYoung/klz/processed-for-kamnet2/XeLS_I130",
        "K40m"     : "/projectnb/snoplus/SoYoung/klz/processed-for-kamnet2/XeLS_K40m",
        "Sb118"    : "/projectnb/snoplus/SoYoung/klz/processed-for-kamnet2/XeLS_Sb118",
        "SingleGamma-2225keV":"/projectnb/snoplus/SoYoung/klz/processed-for-kamnet2/XeLS_SingleGamma-2225keV",
    }
    bkg_dir_dict_selected = {}
    for bkg_name, bkg_dir in bkg_dir_dict.items():
        if bkg_name == args.bkg:
            bkg_dir_dict_selected[bkg_name] = bkg_dir
    
    if len(bkg_dir_dict_selected.keys()) < 1:
        raise Exception("No bkgs selected... check your arguments.")

    if not os.path.exists(outdir):
        os.makedirs(outdir)
    result_file_path = f'{outdir}/results_{args.bkg}_{elow}_{ehi}.p'
    
    print("Start!")

    test_loader, time_channel = load_data(BATCH_SIZE, sig_dir, bkg_dir_dict_selected, elow, ehi, filemax)
    # test_loader, time_channel = load_data(BATCH_SIZE, args.sig, bkg_dir_dict, args.elow, args.ehi, args.filemax)

    classifier = KamNet(time_channel)

    #=====================================================================================
    '''
    This part allows the loading of previously trained of KamNet using '.pt' model
    '''
    pretrained_dict = torch.load(pt_file)
    model_dict = classifier.state_dict()
    model_dict.update(pretrained_dict)
    classifier.load_state_dict(pretrained_dict)
    #=====================================================================================

    classifier.to(DEVICE)

    print("#params", sum(x.numel() for x in classifier.parameters()))

    '''
    Define the loss function
    '''
    criterion = nn.BCEWithLogitsLoss()
    criterion = criterion.to(DEVICE)

    param_dict = KAMNET_PARAMS

    #=====================================================================================
    '''
    Set up optimizer with varying learning rate:
      Ramp Up   : Gradually ramp up learning rate in the first 5 epochs, this allows the attention mechanism to learn proper attention score
      Flat    : Fix the learning rate at the nominal value
      Ramp Down : Ramp down the learning rate to 10% of nominal value in the last 10th - 5th epochs
      Flat    : Fix the learning rate at 10% of the nominal value for the last 5 epochs
    '''
    # step_length = len(train_loader)
    # total_step = int(NUM_EPOCHS * step_length)
    # ramp_up = np.linspace(1e-4, 1.0, 5*step_length)
    # ramp_down = list(np.linspace(1.0, 0.1, 5*step_length).flatten()) + [0.1]* 5*step_length
    # ramp_down_start = total_step - len(ramp_down)
    # lmbda = lambda epoch: ramp_up[epoch] if epoch<len(ramp_up) else ramp_down[epoch-ramp_down_start-1] if epoch > ramp_down_start else 1.0
    # optimizer = torch.optim.RMSprop(classifier.parameters(),lr=param_dict["lr"], momentum=param_dict["momentum"])
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda)
    #=====================================================================================

    # dictionary for saving to pickle
    datadict = {}
    for key in ['sigmoid_s', 'sigmoid_b', 'rej_eff', 'auc', 'v_loss', 'energy_s', 'energy_b']:
        datadict[key] = []

    # =========== VALIDATE =============
    sigmoid_s = []
    energy_s = []
    sigmoid_b_dict = {}
    energy_b_dict = {}
    loss_i = 0
    for (images, labels, energies) in test_loader:
        classifier.eval()

        with torch.no_grad():
            outputs = classifier(images.to(DEVICE)).view(-1,1)

            label_data = np.array(labels)
            output_data = outputs.cpu().data.numpy().flatten()
            energy_data = energies.cpu().data.numpy().flatten()

            # loss = criterion(outputs, label_data)
            # loss_i += loss.item()

            signal = np.argwhere(label_data == "signal")
            sigmoid_s += list(output_data[signal].flatten())
            energy_s += list(energy_data[signal].flatten())

            bkg_name_list = np.unique(label_data[label_data != "signal"])
            for bkg_name in bkg_name_list:
                if bkg_name not in sigmoid_b_dict:
                    sigmoid_b_dict[bkg_name] = []
                    energy_b_dict[bkg_name] = []
                sigmoid_b_dict[bkg_name] += list(output_data[label_data == bkg_name].flatten())
                energy_b_dict[bkg_name] += list(energy_data[label_data == bkg_name].flatten())
    datadict['v_loss'] = loss_i/len(test_loader)

    # Validation rejection
    cut = np.nanpercentile(sigmoid_s, 10)
    rej_eff_dict, auc_dict = {}, {}
    for bkg_name, sigmoid_b in sigmoid_b_dict.items():
        print(bkg_name)
        reject = 0
        for i in sigmoid_b:
            if i < cut:
                reject += 1
        rej_eff = 100*reject/len(sigmoid_b)
        rej_eff_dict[bkg_name] = rej_eff
        print(f"validation rejection efficiency: {rej_eff}")

        # Area Under ROC Curve
        auc_labels = np.concatenate((np.ones(len(sigmoid_s)), np.zeros(len(sigmoid_b))))
        auc_scores = np.concatenate((sigmoid_s, sigmoid_b))
        auc_i = roc_auc_score(auc_labels, auc_scores)
        auc_dict[bkg_name] = auc_i
        print('AUC:', auc_i)

    datadict['sigmoid_s'] = sigmoid_s
    datadict['sigmoid_b'] = sigmoid_b_dict
    datadict['energy_s'] = energy_s
    datadict['energy_b'] = energy_b_dict
    datadict['rej_eff'] = rej_eff_dict
    datadict['auc'] = auc_dict

    del images
    torch.cuda.empty_cache()

    # save results
    with open(result_file_path, 'wb') as pfile:
        pickle.dump(datadict, pfile)