'''
Wrapper for running KamNet

Last Modified:
    Apr. 26th, 2026 by So Young Jeon (jeonsy@bu.edu)
'''
import os
import sys
import shutil
import argparse
import random
import pickle
import tomllib
from tqdm import tqdm

import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

from KamNet import KamNet
from KamNetDataset import KamNetDataset

def useSeed(seed=7):
    '''
    Setting reproducability. If used, training the neural network with
    the same configuration will result in exactly the same output

    FIXME : should be double-checked if everything has been accounted for
    '''
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    # if you are using GPU
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = False 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    return

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

def plotNhit(dataset, figpath):
    image, label, other_vars = dataset[0]
    try:
        temp = other_vars['Nhit']
    except:
        print("Nhit won't be plotted because information was not saved.")
        return # didn't save Nhit. can't plot.
    try:
        temp = other_vars['isotope']
        mode = 'isotope'
    except:
        mode = 'label'

    nhits = {}
    for idx in range(len(dataset)):
        image, label, other_vars = dataset[idx]
        if mode == 'isotope': key = other_vars['isotope']
        elif mode == 'label': key = "Signal" if label == 1 else "Background"
        try:
            nhits[key].append(other_vars['Nhit'])
        except:
            nhits[key] = [other_vars['Nhit']]
    
    fig, ax = plt.subplots(figsize=[8,6], tight_layout=True)
    bins = np.arange(0, 700+5, 5)
    for key, val in nhits.items():
        plt.hist(val, label=key, bins=bins, histtype="stepfilled", edgecolor="black", alpha=0.4)
    plt.xlim(bins[0], bins[-2])
    plt.xlabel("Nhit")
    plt.legend()
    plt.grid()
    plt.savefig(figpath)
    plt.close()

    print(f"Plot saved to {figpath}")

def addMetrics(results, sig_accept_rate=0.7, plot=False, plot_dir="./"):
    '''
    Add a few metrics to the results
        - Overall loss, ROC curve, AUC
        - Rejection efficiency (keeping <sig_accept_rate> of signal)
    '''
    # Get overall loss
    results['loss'] = float(np.mean(results['loss_per_batch']))
    print(f"Loss | {results['loss']:.4f}")

    # Get ROC curve
    fpr, tpr, thresholds = metrics.roc_curve(results['label'], results['score'])
    results['roc_fpr'] = fpr.tolist()
    results['roc_tpr'] = tpr.tolist()
    results['roc_thresholds'] = thresholds.tolist()

    # Get AUC
    results['auc'] = float(metrics.auc(fpr, tpr))
    print(f"AUC  | {results['auc']:.4f}")

    # Get Rejection Efficiency
    rej_eff = 1 - np.interp(sig_accept_rate, tpr, fpr)
    print(f"BG Rejection Efficiency (keep {round(sig_accept_rate*100)}% signal) | {rej_eff*100:5.2f}%")

    if plot:
        # FIXME : plot stuff (roc curve, ...)
        pass

    return results

def trainKamNet(train_loader, val_loader, kamnet_params, DEVICE,
        learning_rate    = 0.000018675460538381732,
        num_epochs       = 30,
        output_vars      = [],
        result_dir_path  = "./",
        make_plots       = False,
        plot_dir_path    = "./"
        ):
    '''
    Train KamNet
    '''

    # set up result file paths
    t_result_path = os.path.join(result_dir_path, f"result_training.pickle")
    v_result_path = os.path.join(result_dir_path, f"result_validation.pickle")
    try: os.remove(t_result_path) # delete if already exists; will be appending to file!
    except: pass
    try: os.remove(v_result_path)
    except: pass

    # ================== KAMNET INITIATION =====================
    classifier = KamNet(train_loader.dataset.getInputDimension(), kamnet_params)
    classifier.to(DEVICE)

    print(f"# of Parameters : {sum(x.numel() for x in classifier.parameters())}")

    # Define the loss function
    criterion = nn.BCEWithLogitsLoss()
    criterion = criterion.to(DEVICE)

    # Set up optimizer with varying learning rate
    '''
    Ramp Up   : Gradually ramp up learning rate in the first 5 epochs, this allows the attention mechanism to learn proper attention score
    Flat      : Fix the learning rate at the nominal value
    Ramp Down : Ramp down the learning rate to 10% of nominal value in the last 10th - 5th epochs
    Flat      : Fix the learning rate at 10% of the nominal value for the last 5 epochs
    '''
    step_size  = len(train_loader)
    step_total = int(num_epochs * step_size)
    len_up, len_down, len_end = 5 * step_size, 5 * step_size, 5 * step_size
    len_mid    = step_total - (len_up + len_down + len_end)
    flat_mid, flat_end = 1.0, 0.1

    def lr_rate_fcn(epoch):
        if epoch < len_up:
            return 1e-4 + (epoch / len_up) * (flat_mid - 1e-4)
        elif epoch < len_up + len_mid:
            return flat_mid
        elif epoch < len_up + len_mid + len_down:
            return flat_mid - ((epoch - (len_up + len_mid)) / len_down) * (flat_mid - flat_end)
        else:
            return flat_end
    
    optimizer = torch.optim.RMSprop(classifier.parameters(), lr=kamnet_params["lr"], momentum=kamnet_params["momentum"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_rate_fcn)

    # ================== RUN TRAINING ==================
    print("\n>>> Running KamNet...")
    result_params = np.unique(['score','label','loss_per_batch'] + output_vars).tolist()

    for epoch in tqdm(range(num_epochs), ncols=0):
        print("\n\n==================================================")
        print(f"  [ EPOCH {epoch + 1:02d}/{num_epochs} ]   Learning Rate : {scheduler.get_last_lr()[0]:.10f}")
        print(    "==================================================")

        # ================== TRAINING ==================
        print("\n>>> Training...")
        training_results = {el:[] for el in result_params}

        for i, (images, labels, other_vars) in enumerate(train_loader):
            classifier.train()

            outputs = classifier(images.to(DEVICE)).view(-1,1)
            labels_float = labels.to(DEVICE).view(-1,1).float()

            loss = criterion(outputs, labels_float)
            print(f"  - Iter {i:0{len(str(step_size))}d}/{step_size} | Loss : {loss.item():.5f}")

            loss.backward()         # optimize the net
            optimizer.step()        # update parameters of net
            optimizer.zero_grad()   # reset gradient
            scheduler.step()

            # save results
            label_array = labels_float.cpu().flatten().tolist()
            score_array = outputs.cpu().data.flatten().tolist()

            training_results['loss_per_batch'].append(float(loss.item()))
            training_results['score'] += score_array
            training_results['label'] += label_array

            for key, item in other_vars.items():
                training_results[key] += list(item)
        
        # save training results
        training_results = addMetrics(training_results, plot=make_plots, plot_dir=plot_dir_path)

        with open(t_result_path, 'ab') as pfile:
            pickle.dump(training_results, pfile)

        # Save KamNet parameters to .pt file
        torch.save(classifier.state_dict(), os.path.join(result_dir_path, f'KamNet_model_epoch{epoch}.pt'))

        # ================== VALIDATION ==================
        print("\n>>> Validating...")
        validation_results = {el:[] for el in result_params}

        for (images, labels, other_vars) in val_loader:
            classifier.eval()

            with torch.no_grad():
                outputs = classifier(images.to(DEVICE)).view(-1,1)
                labels_float = labels.to(DEVICE).view(-1,1).float()

                loss = criterion(outputs, labels_float)

                # save results
                label_array = labels_float.cpu().flatten().tolist()
                score_array = outputs.cpu().detach().flatten().tolist()

                validation_results['loss_per_batch'].append(float(loss.item()))
                validation_results['score'] += score_array
                validation_results['label'] += label_array

                for key, item in other_vars.items():
                    validation_results[key] += list(item)

        validation_results = addMetrics(validation_results, plot=make_plots, plot_dir=plot_dir_path)

        # save validation results to pickle file
        with open(v_result_path, 'ab') as pfile:
            pickle.dump(validation_results, pfile)
        
        # clean up
        del images
        torch.cuda.empty_cache()
    # END of EPOCH loop

    return 0

def testKamNet(test_loader, trained_model, kamnet_params, DEVICE,
        output_vars      = [],
        result_dir_path  = "./test_results",
        make_plots       = False,
        plot_dir_path    = "./"
        ):
    '''
    Test KamNet on a dataset with a previously trained KamNet model
    '''
    result_file_path = os.path.join(result_dir_path, f'result_test.pickle')
    plot_dir_path = os.path.join(result_dir_path, f'plots/')

    # ================== KAMNET INITIATION =====================
    classifier = KamNet(test_loader.dataset.getInputDimension(), kamnet_params)

    # Load previously trained model of KamNet using '.pt' model
    print(f"\n>>> Loading previously trained model at {trained_model}")
    pretrained_dict = torch.load(trained_model)
    model_dict = classifier.state_dict()
    model_dict.update(pretrained_dict)
    classifier.load_state_dict(pretrained_dict)
    classifier.to(DEVICE)

    print(f"# of Parameters : {sum(x.numel() for x in classifier.parameters())}")

    # Define the loss function
    criterion = nn.BCEWithLogitsLoss()
    criterion = criterion.to(DEVICE)

    # ================== RUN VALIDATION ==================
    print("\n>>> Running KamNet...")

    default_results = ['score','label','loss_per_batch']
    results = {el:[] for el in np.unique(default_results + output_vars).tolist()}
    for (images, labels, other_vars) in tqdm(test_loader):
        classifier.eval()

        with torch.no_grad():
            outputs = classifier(images.to(DEVICE)).view(-1,1)
            labels_float = labels.to(DEVICE).view(-1,1).float()

            loss = criterion(outputs, labels_float)

            # save results
            label_array = labels_float.cpu().flatten().tolist()
            score_array = outputs.cpu().detach().flatten().tolist()
            
            results['loss_per_batch'].append(float(loss.item()))
            results['score'] += score_array
            results['label'] += label_array

            for key, item in other_vars.items():
                try:
                    results[key] += item.tolist()
                except:
                    results[key] += list(item)

    del images
    torch.cuda.empty_cache()

    results = addMetrics(results, plot=True, plot_dir=plot_dir_path)

    # save to pickle file
    with open(result_file_path, 'wb') as pfile:
        pickle.dump(results, pfile)
        print(f"Results saved to {result_file_path}")

    return 0

def main(CONFIG, DEVICE):
    '''
    main function
    '''
    # set up output directories
    if not os.path.exists(CONFIG['output_dir']):
        os.makedirs(CONFIG['output_dir'])
    plot_dir_path = os.path.join(CONFIG['output_dir'], "plots/")
    if not os.path.exists(plot_dir_path):
        os.makedirs(plot_dir_path)

    # copy settings.toml over to result directory for record-keeping
    shutil.copy(args.configfile, os.path.join(CONFIG['output_dir'], f"settings.toml"))

    # use seed?
    if CONFIG['use_seed']:
        useSeed(seed=CONFIG['seed_value'])

    # get dictionary of list of input files
    files_dict = {}
    for isotope, folder in CONFIG['input'].items():
        files_dict[isotope] = getFilesUnderFolder(folder, filetype="pickle")[:CONFIG['max_num_files']]

    # ============= TRAIN ==============
    if CONFIG['run_mode'] == "train":
        # load dataset
        print(">>> Loading data...")
        dataset = KamNetDataset(files_dict, CONFIG['signal_isotope'],
            vars_to_output=CONFIG['output_vars'], elow=CONFIG['elow'], ehigh=CONFIG['ehigh'])
        dataset.printSize()

        # run Nhit matching
        if CONFIG['make_plots']:
            plotNhit(dataset, os.path.join(plot_dir_path, "Nhit_before_matching.png"))
        dataset.matchSBNhit()
        if CONFIG['make_plots']:
            plotNhit(dataset, os.path.join(plot_dir_path, "Nhit_after_matching.png"))

        if CONFIG['max_dataset_size']:
            dataset.downsize(CONFIG['max_dataset_size'])

        # split into training and validation datasets
        dataset_size = len(dataset)
        dataset_indices = np.array(range(dataset_size))
        np.random.shuffle(dataset_indices)
        split_index = int(np.floor(CONFIG['validation_split'] * dataset_size))

        train_sampler = SubsetRandomSampler(dataset_indices[split_index:])
        val_sampler = SubsetRandomSampler(dataset_indices[:split_index])
        train_loader = data_utils.DataLoader(dataset, sampler=train_sampler, batch_size=CONFIG['batch_size'], drop_last=True)
        va_loader = data_utils.DataLoader(dataset, sampler=val_sampler, batch_size=CONFIG['batch_size'], drop_last=True)

        print(f"Training DataLoader size : {len(train_loader)} | Batch size : {CONFIG['batch_size']}")
        print(f"Validation DataLoader size : {len(va_loader)} | Batch size : {CONFIG['batch_size']}")

        # run training!
        trainKamNet(
            train_loader,
            va_loader,
            CONFIG['kamnet_params'],
            DEVICE,
            learning_rate    = CONFIG['learning_rate'],
            num_epochs       = CONFIG['num_epochs'],
            output_vars      = CONFIG['output_vars'],
            result_dir_path  = CONFIG['output_dir'],
            make_plots       = CONFIG['make_plots'],
            plot_dir_path    = plot_dir_path,
            )

    # ============ TEST =============
    elif CONFIG['run_mode'] == "test":
        print(">>> Loading data...")
        # load dataset & dataloader
        dataset = KamNetDataset(files_dict, CONFIG['signal_isotope'],
            vars_to_output=CONFIG['output_vars'], elow=CONFIG['elow'], ehigh=CONFIG['ehigh'])
        dataset.printSize()
        if CONFIG['max_dataset_size']:
            dataset.downsize(CONFIG['max_dataset_size'])
        test_loader = data_utils.DataLoader(dataset, batch_size=CONFIG['batch_size'], drop_last=True)
        print(f"DataLoader size : {len(test_loader)} | Batch size : {CONFIG['batch_size']}")

        testKamNet(
            test_loader,
            CONFIG['trained_model'],
            CONFIG['kamnet_params'],
            DEVICE,
            output_vars      = CONFIG['output_vars'],
            result_dir_path  = CONFIG['output_dir'],
            make_plots       = CONFIG['make_plots'],
            plot_dir_path    = plot_dir_path,
            )

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('configfile', type=str, action="store", help="Pass in .TOML file that contains KamNet settings to use.")
    args = parser.parse_args()

    # read config from TOML file
    with open(args.configfile, "rb") as f:
        CONFIG = tomllib.load(f)
    
    # initiate GPU device
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    try:
        assert DEVICE.type == "cuda"
        torch.zeros(1).cuda()
    except RuntimeError as e:
        print("************************************************")
        print("* GPU is not available.                        *")
        print("* Make sure you have access to a GPU.          *")
        print("* Try running `nvidia-smi` and `kill -9 <PID>` *")
        print("* to kill unwanted processes on GPU.           *")
        print("************************************************")
        sys.exit(e)

    
    main(CONFIG, DEVICE)