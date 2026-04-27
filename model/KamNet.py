#=====================================================================================
#  Author: Aobo Li
#  Contact: liaobo77@gmail.com
#  
#  Last Modified: Aug. 29, 2021
#  
#  * KamNet is a deep learning model developed for KamLAND-Zen and 
#    other spherical liquid scintillator detectors.
#  * It attempts to harness all of the inherent symmetries to produce a
#    state-of-the-art algorithms for a spherical liquid scintillator detector.
#=====================================================================================
import numpy as np

from s2cnn import SO3Convolution
from s2cnn import S2Convolution
from s2cnn import so3_integrate
from s2cnn import so3_near_identity_grid, so3_equatorial_grid
from s2cnn import s2_near_identity_grid, s2_equatorial_grid

import torch
import torch.nn as nn
from torch.amp import autocast
from AttentionConvLSTM import ConvLSTM

class KamNet(nn.Module):

    def __init__(self, time_channel, param_dict):
        super(KamNet, self).__init__()

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
        self.convlstm1=ConvLSTM(1, [s1,s2], [(param_dict["first_filter"],param_dict["first_filter"]),(param_dict["second_filter"],param_dict["second_filter"])],2, time_channel, batch_first=True, fill_value=0.1)

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
