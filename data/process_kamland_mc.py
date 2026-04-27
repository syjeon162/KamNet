#=====================================================================================
#    Author: Aobo Li
#    Contact: liaobo77@gmail.com
#    
#    Last Modified:
#       Apr. 26th, 2026, by So Young Jeon (jeonsy@bu.edu)
#    
#    * This code is used to convert MC simulated .root file into a 2D square grid
#    * Save each event and other variables as a CSR sparse matrix in .pickle format.
#    * Only applicable to the KLGSim simulation by the KamLAND-Zen group. To use this on your
#      own experiment, please modify this code to adapt to your own MC data structures.
#=====================================================================================
import argparse
import os
import math
import pickle
import tomllib
from scipy import sparse

import numpy as np
from ROOT import TFile
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from clock import clock

def PMT_setup(pmt_file_with_index):
    '''
    Reads in PMT positional information according to PMT location file.
    This file is internal to KamLAND collaboration thus cannot be provided here,
    it should follow the format of:

    PMTID  PMT_X[cm]  PMT_Y[cm]  PMT_Z[cm]

    Each line represents a PMT and each field is separated by blank space
    '''
    PMT_POSITION = {}
    for pmt in np.loadtxt(pmt_file_with_index):
        current_pmt_pos = pmt[1:] / 100.0
        PMT_POSITION[int(pmt[0])] = current_pmt_pos
    return PMT_POSITION

def xyz_to_phi_theta(x, y, z):
    phi = math.atan2(y, x)
    r = (x**2 + y**2 + z**2)**.5
    theta = math.acos(z / r)
    return phi, theta

# Convert the phi theta information to row and column index in 2D grid
def phi_theta_to_row_col(phi, theta, rows, cols):
   # phi is in [-pi, pi], theta is in [0, pi]
   row = max(0, min(rows / 2 + (math.floor((rows / 2) * phi / math.pi)), rows - 1))
   col = max(0, min(math.floor(cols * theta / math.pi), cols - 1))
   return int(row), int(col)

# Converting Cartesian position to 2D Grid
def xyz_to_row_col(pmt_index, PMT_POSITION, rows=38, cols=38):
    x, y, z = tuple(PMT_POSITION[pmt_index])
    return phi_theta_to_row_col(*xyz_to_phi_theta(x, y, z), rows, cols)

def plotHitMap(t_hist, current_clock, event_map, idx_pool=[5,11,14,18], figpath="./event.png"):
    '''
    This is the plot method for given dataset, it plots a few selected hit maps for
    demonstration purpose
    '''
    plt.figure(figsize=(15,15))
    spec = gridspec.GridSpec(ncols=4, nrows=2, height_ratios=[1,2])
    plt.subplot(spec[1,:])
    plt.hist(t_hist, bins=np.arange(-20,40,1.5), density=True, color=colormap_normal(0.2))
    plt.axvline(x=-20, color="red", label="KamNet Window")
    plt.axvline(x=22, color="red")
    for idxc in idx_pool:
        begin, end = current_clock.get_range_from_tick(idxc)
        plt.axvspan(xmin=begin, xmax=end, color=colormap_normal(0.7), alpha=0.5)
    plt.ylim(0,0.08)
    plt.legend(frameon=False)
    plt.xlabel("Proper Hit Time [ns]", fontsize=25, labelpad=20)
    plt.ylabel("Normalized Amplitude", fontsize=25, labelpad=20)

    for event_data in event_map:
        subplot_index = 0
        for idx in idx_pool:
            maps = event_data['event'][idx]
            ax = plt.subplot(spec[0, subplot_index])
            begin, end = current_clock.get_range_from_tick(idx)
            if begin == -9999:
                plt.title("(Past, %.1f ns)"%(end), fontsize=30)
            else:
                plt.title("(%s ns, %.1f ns)"%(begin,end), fontsize=30)
            ax.axes.get_xaxis().set_visible(False)
            ax.axes.get_yaxis().set_visible(False)
            ax.imshow(maps, cmap=colormap_normal, norm=matplotlib.colors.LogNorm(vmin=0.3, vmax=10.0))
            subplot_index += 1
            if subplot_index > 49:
                break
            plt.tight_layout()
            plt.savefig(figpath, dpi=600)

def main(CONFIG):
    output_parent_dir = os.path.dirname(CONFIG['output'])
    if not os.path.exists(output_parent_dir):
        os.makedirs(output_parent_dir)

    PMT_POSITION = PMT_setup(CONFIG['pmt_file_index'])
    current_clock = clock(0)

    f = TFile(CONFIG['input'])
    tree = f.Get("nt")  # Read the ROOT tree

    event_map = []
    if CONFIG['plot_hitmap']:
        t_hist = []

    for evt_index in tqdm(range(tree.GetEntries())):
        tree.GetEntry(evt_index)

        # FV / ROI cut
        try:
            energy = tree.EnergyA2
            position = tree.r
            if (energy < CONFIG['elow']) or (energy > CONFIG['ehigh']):
                continue
            if (position < CONFIG['fv_cut_low']) or (position > CONFIG['fv_cut_high']):
                continue
        except:
            print("Error when making event selection. Skip event.")
            continue

        '''
        Read out PMT hitlist, time and charge
        '''
        # get PMT information for a event
        if CONFIG['good_pmthit']:
            pmt_list = np.array(tree.pmtlist_good)
            pmt_time_list = np.array(tree.pmtt_good)
            pmt_charge_list = np.array(tree.pmtq_good)
        else:
            pmt_list = np.array(tree.pmtlist)
            pmt_time_list = np.array(tree.pmtt)
            pmt_charge_list = np.array(tree.pmtq)

        # Read out only 17 inch PMTs (that is, PMT index < 1325)
        if CONFIG['only_17inch']:
            good_index = pmt_list < 1325
            pmt_list = pmt_list[good_index]
            pmt_time_list = pmt_time_list[good_index]
            pmt_charge_list = pmt_charge_list[good_index]

        # Calculate time of flight.
        # In KLZ simulation the TOF is already subtracted, so just set it to 0
        pmt_tof = np.zeros(len(pmt_list))

        stacked_pmt_info = np.dstack((pmt_list, pmt_time_list, pmt_charge_list, pmt_tof))[0]

        t0 = tree.T0
        event = np.zeros((current_clock.clock_size(), CONFIG['rows'], CONFIG['cols']))
        for pmtinfo in stacked_pmt_info:
            if pmtinfo[-2] == 0.0:
                # Skip PMT with 0 charge
                continue

            col, row = xyz_to_row_col(pmtinfo[0], PMT_POSITION, CONFIG['rows'], CONFIG['cols'])
            t_center = pmtinfo[1] - t0
            if CONFIG['plot_hitmap']:
                t_hist.append(t_center)
            time = current_clock.tick(t_center)

            if CONFIG['use_charge']:
                event[time][row][col] += pmtinfo[-2]
            else:
                event[time][row][col] += 1.0

        event_dict = {}
        event_dict['id'] = tree.EventNumber
        event_dict['run'] = tree.run
        event_dict['Nhit'] = np.count_nonzero(event)
        event_dict['energy'] = energy
        event_dict['vertex'] = tree.r
        event_dict['zpos'] = tree.z
        event_dict['event'] = event

        event_map.append(event_dict)
        
    if CONFIG['plot_hitmap']:
        plotHitMap(t_hist, current_clock, event_map)

    with open(CONFIG['output'], 'wb') as handle:
        numev = 0
        print(" * Number of Events total: ", len(event_map))

        for event_data in event_map:
            numev += 1
            time_sequence = []
            for idx, maps in enumerate(event_data['event']):
                # Save each event as a CSR sparse matrix
                time_sequence.append(sparse.csr_matrix(maps))
            event_data['event'] = time_sequence

            # dump event into .pickle file
            pickle.dump(event_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

        print(" * Number of Events dumped: ", numev)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('configfile', type=str, action="store", help="Pass in .TOML file that contains KamNet settings to use.")
    args = parser.parse_args()

    # read config from TOML file
    with open(args.configfile, "rb") as f:
        CONFIG = tomllib.load(f)

    main(CONFIG)