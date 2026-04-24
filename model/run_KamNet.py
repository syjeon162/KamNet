'''
Wrapper for running KamNet
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

def load_data(files_dict, signal_isotope,
        vars_to_output=[], batch_size=32, elow=0.5, ehigh=5.0, dsize=None, match_nhit=False):
    '''
    Load datasets from input
        * files_dict: dictionary of list of paths to pickle files, keyed by isotope name
        * signal_isotope: name of isotope that should be designated as the signal
    '''
    print("\n>>> Loading data...")
    
    dataset = KamNetDataset(files_dict, signal_isotope, vars_to_output=vars_to_output,
                            elow=elow, ehigh=ehigh)

    dataset.printSize()

    if match_nhit:
        # match nhit between signal and all backgrounds aggregated together
        if match_nhit == "sig-bkg":
            dataset.matchSBNhit()
        # match nhit between each and all isotopes
        elif match_nhit == "isotope":
            dataset.matchIsotopeNhit()
        else:
            raise Exception("Pass in 'sig-bkg' or 'isotope'")

    if dsize:
        dataset.downsize(dsize)

    dataloader = data_utils.DataLoader(dataset, batch_size=batch_size, drop_last=True)
    
    print(f"DataLoader size: {len(dataloader)} / Batch size: {batch_size}")

    return dataloader

def addMetrics(results, sig_accept_rate=0.7, plot=False, plot_dir="./"):
    '''
    Add a few metrics to the results
        - Overall loss, ROC curve, AUC
        - Rejection efficiency (keeping <sig_accept_rate> of signal)
    '''
    # Get overall loss
    results['loss'] = np.mean(results['loss_per_batch'])
    print(f"Loss | {results['loss']:.4f}")

    # Get ROC curve
    fpr, tpr, thresholds = metrics.roc_curve(results['label'], results['score'])
    results['roc_fpr'] = fpr
    results['roc_tpr'] = tpr
    results['roc_thresholds'] = thresholds

    # Get AUC
    results['auc'] = metrics.auc(fpr, tpr)
    print(f"AUC  | {results['auc']:.4f}")

    # Get Rejection Efficiency
    rej_eff = 1 - np.interp(sig_accept_rate, tpr, fpr)
    print(f"BG Rejection Efficiency (keep {round(sig_accept_rate*100)}% signal) | {rej_eff*100:5.2f}%")

    if plot:
        # FIXME : plot stuff (roc curve, ...)
        print("")

    return results

def trainKamNet(data_loader, kamnet_params, DEVICE,
        learning_rate    = 0.000018675460538381732,
        num_epochs       = 30,
        validation_split = 0.3,
        output_vars      = [],
        result_dir_path  = "./training_results",
        result_file_name = "test"):
    '''
    Train KamNet
    '''

    # set up result file paths
    t_result_path = os.path.join(result_dir_path, f"{result_file_name}_training_result.pickle")
    v_result_path = os.path.join(result_dir_path, f"{result_file_name}_validation_result.pickle")
    # FIXME : delete if already exists; will be appending to file!

    # FIXME : split into training and validation dataset
    train_loader = ""
    val_loader = ""

    # ================== KAMNET INITIATION =====================
    classifier = KamNet(data_loader.dataset.getInputDimension(), kamnet_params)
    classifier.to(DEVICE)

    print(f"# of Parameters : {sum(x.numel() for x in classifier.parameters())}")

    # Define the loss function
    criterion = nn.BCEWithLogitsLoss()
    criterion = criterion.to(DEVICE)

    # Set up optimizer with varying learning rate
    '''
    # Ramp Up   : Gradually ramp up learning rate in the first 5 epochs, this allows the attention mechanism to learn proper attention score
    # Flat      : Fix the learning rate at the nominal value
    # Ramp Down : Ramp down the learning rate to 10% of nominal value in the last 10th - 5th epochs
    # Flat      : Fix the learning rate at 10% of the nominal value for the last 5 epochs
    '''
    step_size  = len(train_loader)
    step_total = int(num_epochs * step_size)
    len_up, len_down, len_end = 5 * step_size, 5 * step_size, 5 * step_size
    len_mid    = step_total - sum(len_up, len_down, len_end)

    ramp_up    = list(np.linspace(1e-4, 1.0, len_up))
    flat_mid   = [1.0] * len_mid
    ramp_down  = list(np.linspace(1.0, 0.1, len_down))
    flat_end   = [0.1] * len_end
    lr_list    = ramp_up + flat_mid + ramp_down + flat_end
    
    optimizer = torch.optim.RMSprop(classifier.parameters(), lr=kamnet_params["lr"], momentum=kamnet_params["momentum"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda epoch: lr_list[epoch])

    # ================== RUN TRAINING ==================
    print("\n>>> Running KamNet...")
    result_params = np.unique(['score','label','loss_per_batch'] + output_vars)

    for epoch in range(num_epochs):
        print(f"\n[ EPOCH {epoch + 1}/{num_epochs} ]   Learning Rate : {scheduler.get_lr()}")

        # ================== TRAINING ==================
        print("\nTraining...")
        training_results = {el:[] for el in result_params}

        for i, (images, labels, other_vars) in enumerate(train_loader):
            classifier.train()

            outputs = classifier(images.to(DEVICE)).view(-1,1)
            labels_float = labels.to(DEVICE).view(-1,1).float()

            loss = criterion(outputs, labels_float)
            print(f"   - Iter {i}/{len(train_loader)} | Loss : {loss.item()}")

            loss.backward()         # optimize the net
            optimizer.step()        # update parameters of net
            optimizer.zero_grad()   # reset gradient
            scheduler.step()

            # save results
            label_array = labels_float.cpu().numpy().flatten()
            score_array = outputs.cpu().data.numpy().flatten()

            training_results['loss_per_batch'].append(loss.item())
            training_results['score'] += list(score_array)
            training_results['label'] += list(label_array)

            for key, item in other_vars.items():
                training_results[key] += list(item)
        
        # save training results
        training_results = addMetrics(training_results, plot=True, plot_dir=result_dir_path)

        with open(t_result_path, 'ab') as pfile:
            pickle.dump(training_results, pfile)

        # Save KamNet parameters to .pt file
        torch.save(classifier.state_dict(), f'KamNet_{ev_suffix}.pt')

        # ================== VALIDATION ==================
        print("\nValidating...")
        validation_results = {el:[] for el in result_params}

        for (images, labels, other_vars) in val_loader:
            classifier.eval()

            with torch.no_grad():
                outputs = classifier(images.to(DEVICE)).view(-1,1)
                labels_float = labels.to(DEVICE).view(-1,1).float()

                loss = criterion(outputs, labels_float)

                # save results
                label_array = labels_float.cpu().numpy().flatten()
                score_array = outputs.cpu().data.numpy().flatten()

                validation_results['loss_per_batch'].append(loss.item())
                validation_results['score'] += list(score_array)
                validation_results['label'] += list(label_array)

                for key, item in other_vars.items():
                    validation_results[key] += list(item)

        validation_results = addMetrics(validation_results, plot=True, plot_dir=result_dir_path)

        # save validation results to pickle file
        with open(v_result_path, 'ab') as pfile:
            pickle.dump(validation_results, pfile)
        
        # clean up
        del images
        torch.cuda.empty_cache()
    # END of EPOCH loop

    return 0

def testKamNet(test_loader, kamnet_params, trained_model, DEVICE,
        output_vars      = [],
        result_dir_path  = "./test_results",
        result_file_name = "test"):
    '''
    Test KamNet on a dataset with a previously trained KamNet model
    '''
    result_file_path = os.path.join(result_dir_path, f'{result_file_name}_test_result.pickle')

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
    results = {el:[] for el in np.unique(default_results + output_vars)}
    for (images, labels, other_vars) in tqdm(test_loader):
        classifier.eval()

        with torch.no_grad():
            outputs = classifier(images.to(DEVICE)).view(-1,1)
            labels_float = labels.to(DEVICE).view(-1,1).float()

            loss = criterion(outputs, labels_float)

            # save results
            label_array = labels_float.cpu().numpy().flatten()
            score_array = outputs.cpu().data.numpy().flatten()
            
            results['loss_per_batch'].append(loss.item())
            results['score'] += list(score_array)
            results['label'] += list(label_array)

            for key, item in other_vars.items():
                results[key] += list(item)

    del images
    torch.cuda.empty_cache()

    results = addMetrics(results, plot=True, plot_dir=result_dir_path)

    # save to pickle file
    with open(result_file_path, 'wb') as pfile:
        pickle.dump(results, pfile)
        print(f"Results saved to {result_file_path}")

    return 0

def main(CONFIG, DEVICE):
    '''
    main function
    '''
    # set up output directory
    if not os.path.exists(CONFIG['output_dir']):
        os.makedirs(CONFIG['output_dir'])

    # copy settings.toml over to result directory
    shutil.copy(args.configfile, os.path.join(CONFIG['output_dir'], f"settings_{CONFIG['output_file_name']}.toml"))

    # seed?
    if CONFIG['use_seed']:
        useSeed(seed=CONFIG['seed_value'])

    # get list of input files
    files_dict = {}
    for isotope, folder in CONFIG['input'].items():
        files_dict[isotope] = getFilesUnderFolder(folder, filetype="pickle")[:CONFIG['max_num_files']]

    # ============= TRAIN ==============
    if CONFIG['run_mode'] == "train":
        data_loader = load_data(
            files_dict,
            CONFIG['signal_isotope'],
            vars_to_output = CONFIG['output_vars'],
            batch_size     = CONFIG['batch_size'],
            elow           = CONFIG['elow'],
            ehigh          = CONFIG['ehigh'],
            dsize          = CONFIG['max_dataset_size'],
            match_nhit     = "sig-bkg",
            )

        trainKamNet(
            data_loader,
            CONFIG['kamnet_params'],
            CONFIG['trained_model'],
            DEVICE,
            learning_rate    = CONFIG['learning_rate'],
            num_epochs       = CONFIG['num_epochs'],
            validation_split = CONFIG['validation_split'],
            output_vars      = CONFIG['output_vars'],
            result_dir_path  = CONFIG['output_dir'],
            result_file_name = CONFIG['output_file_name'],
            )

    # ============ TEST =============
    elif CONFIG['run_mode'] == "test":

        test_loader = load_data(
            files_dict,
            CONFIG['signal_isotope'],
            vars_to_output = CONFIG['output_vars'],
            batch_size     = CONFIG['batch_size'],
            elow           = CONFIG['elow'],
            ehigh          = CONFIG['ehigh'],
            dsize          = CONFIG['max_dataset_size'],
            match_nhit     = False,
            )

        testKamNet(
            test_loader,
            CONFIG['kamnet_params'],
            CONFIG['trained_model'],
            DEVICE,
            output_vars      = CONFIG['output_vars'],
            result_dir_path  = CONFIG['output_dir'],
            result_file_name = CONFIG['output_file_name'],
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