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
import pandas as pd
from sklearn import metrics

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data_utils
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

from KamNet import KamNet
from KamNetDataset import KamNetDataset
import plotting_tool

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

def getSummaryMetrics(event_data, sig_accept_rate=0.7):
    '''
    Print and add some metrics to the results
        - Overall loss, ROC curve, AUC
        - Rejection efficiency (keeping <sig_accept_rate> of signal)
    '''

    # save full list of isotopes in dataset
    result_summary = {}
    try:
        result_summary['isotopes'] = np.unique(event_data['isotope']).tolist()
    except:
        pass

    # Get overall loss
    result_summary['loss'] = float(np.mean(event_data['loss']))

    # Get ROC curve
    fpr, tpr, thresholds = metrics.roc_curve(event_data['label'], event_data['score'])
    result_summary['roc_fpr'] = fpr.tolist()
    result_summary['roc_tpr'] = tpr.tolist()
    result_summary['roc_thresholds'] = thresholds.tolist()

    # Get AUC
    result_summary['auc'] = float(metrics.auc(fpr, tpr))
    print(f"AUC  | {result_summary['auc']:.4f}")

    # Get Rejection Efficiency
    rej_eff = 1 - np.interp(sig_accept_rate, tpr, fpr)
    print(f"BG Rejection Efficiency (keep {round(sig_accept_rate*100)}% signal) | {rej_eff*100:5.2f}%")

    return result_summary

def trainKamNet(train_loader, val_loader, kamnet_params, DEVICE,
        learning_rate    = 0.000018675460538381732,
        num_epochs       = 30,
        output_vars      = [],
        result_dir_path  = "./",
        make_plots       = False,
        ):
    '''
    Train KamNet
    '''
    # set up result directory paths
    model_dir_path = os.path.join(result_dir_path, "model/")
    event_data_dir_path = os.path.join(result_dir_path, "event_data/")
    plot_dir_path = os.path.join(result_dir_path, "plots/")
    for p in [model_dir_path, event_data_dir_path, plot_dir_path]:
        if not os.path.exists(p):
            os.makedirs(p)

    # ================== KAMNET INITIATION =====================
    classifier = KamNet(train_loader.dataset.getInputDimension(), kamnet_params)
    classifier.to(DEVICE)

    print(f"# of Parameters : {sum(x.numel() for x in classifier.parameters())}")

    # Define the loss function
    criterion = nn.BCEWithLogitsLoss(reduction="none")
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
    
    optimizer = torch.optim.RMSprop(classifier.parameters(), lr=kamnet_params["lr"], momentum=kamnet_params["momentum"], weight_decay=kamnet_params["l2reg"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_rate_fcn)

    # ================== RUN TRAINING ==================
    print("\n>>> Running KamNet...")

    result_params = np.unique(['score','label','loss'] + output_vars).tolist()
    training_results, validation_results = [], []

    for i_epoch in tqdm(range(num_epochs), ncols=0):
        epoch = i_epoch + 1
        print("\n\n==================================================")
        print(f"  [ EPOCH {epoch:02d}/{num_epochs} ]   Learning Rate : {scheduler.get_last_lr()[0]:.10f}")
        print(    "==================================================")

        # ================== TRAINING ==================
        print("\n>>> Training...")
        training_event_data = {el:[] for el in result_params}

        for i, (images, labels, other_vars) in enumerate(train_loader):
            classifier.train()

            outputs = classifier(images.to(DEVICE)).view(-1,1)
            labels_float = labels.to(DEVICE).view(-1,1).float()

            per_sample_loss = criterion(outputs, labels_float)
            loss = per_sample_loss.mean()
            print(f"  - Iter {i:0{len(str(step_size))}d}/{step_size} | Loss : {loss:.5f}")

            loss.backward()         # optimize the net
            optimizer.step()        # update parameters of net
            optimizer.zero_grad()   # reset gradient
            scheduler.step()

            # save results
            label_array = labels_float.cpu().flatten().tolist()
            score_array = outputs.cpu().data.flatten().tolist()
            loss_array  = per_sample_loss.cpu().flatten().tolist()

            training_event_data['score'] += score_array
            training_event_data['label'] += label_array
            training_event_data['loss'] += loss_array

            for key, item in other_vars.items():
                try:
                    training_event_data[key] += item.tolist() # do this if possible to convert tensor to list of floats
                except:
                    training_event_data[key] += list(item)
        
        # make summary of results
        t_result_epoch = getSummaryMetrics(training_event_data)
        training_results.append(t_result_epoch)

        # save event-by-event data as csv
        event_df = pd.DataFrame(training_event_data)
        event_df.to_csv(os.path.join(event_data_dir_path, f"training_epoch{epoch:02}.csv"), index=False)

        # save KamNet parameters to .pt file
        torch.save(classifier.state_dict(), os.path.join(model_dir_path, f'KamNet_model_epoch{epoch:02}.pt'))

        # plot
        if make_plots:
            plotting_tool.plotROC(t_result_epoch['roc_fpr'], t_result_epoch['roc_tpr'],
                            os.path.join(plot_dir_path, f"training_epoch{epoch:02}_roc_curve.png"))
            plotting_tool.plotScoresFromDF(event_df,
                            os.path.join(plot_dir_path, f"training_epoch{epoch:02}_score.png"))


        # ================== VALIDATION ==================
        print("\n>>> Validating...")
        validation_event_data = {el:[] for el in result_params}

        for (images, labels, other_vars) in val_loader:
            classifier.eval()

            with torch.no_grad():
                outputs = classifier(images.to(DEVICE)).view(-1,1)
                labels_float = labels.to(DEVICE).view(-1,1).float()

                per_sample_loss = criterion(outputs, labels_float)

                # save results
                label_array = labels_float.cpu().flatten().tolist()
                score_array = outputs.cpu().detach().flatten().tolist()
                loss_array  = per_sample_loss.cpu().flatten().tolist()

                validation_event_data['score'] += score_array
                validation_event_data['label'] += label_array
                validation_event_data['loss'] += loss_array

                for key, item in other_vars.items():
                    try:
                        validation_event_data[key] += item.tolist() # do this if possible to convert tensor to list of floats
                    except:
                        validation_event_data[key] += list(item)

        # make summary of results
        v_result_epoch = getSummaryMetrics(validation_event_data)
        validation_results.append(v_result_epoch)
        
        # save event-by-event data as csv
        event_df = pd.DataFrame(validation_event_data)
        event_df.to_csv(os.path.join(event_data_dir_path, f"validation_epoch{epoch:02}.csv"), index=False)

        # plot
        if make_plots:
            plotting_tool.plotROC(v_result_epoch['roc_fpr'], v_result_epoch['roc_tpr'],
                            os.path.join(plot_dir_path, f"validation_epoch{epoch:02}_roc_curve.png"))
            plotting_tool.plotScoresFromDF(event_df,
                            os.path.join(plot_dir_path, f"validation_epoch{epoch:02}_score.png"))

        # clean up
        del images
        torch.cuda.empty_cache()
    # END of EPOCH loop

    # save summary results to pickle file
    with open(os.path.join(result_dir_path, f"result_training.pickle"), 'wb') as pfile:
        pickle.dump(training_results, pfile)
    with open(os.path.join(result_dir_path, f"result_validation.pickle"), 'wb') as pfile:
        pickle.dump(validation_results, pfile)
    return 0

def testKamNet(test_loader, trained_model, kamnet_params, DEVICE,
        output_vars      = [],
        result_dir_path  = "./test_results",
        make_plots       = False,
        ):
    '''
    Test KamNet on a dataset with a previously trained KamNet model
    '''
    plot_dir_path = os.path.join(result_dir_path, f'plots/')
    if not os.path.exists(plot_dir_path):
        os.makedirs(plot_dir_path)

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
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    criterion = criterion.to(DEVICE)

    # ================== RUN VALIDATION ==================
    print("\n>>> Running KamNet...")

    event_data = {el:[] for el in np.unique(['score','label','loss'] + output_vars).tolist()}
    for (images, labels, other_vars) in tqdm(test_loader):
        classifier.eval()

        with torch.no_grad():
            outputs = classifier(images.to(DEVICE)).view(-1,1)
            labels_float = labels.to(DEVICE).view(-1,1).float()

            per_sample_loss = criterion(outputs, labels_float)

            # save results
            label_array = labels_float.cpu().flatten().tolist()
            score_array = outputs.cpu().detach().flatten().tolist()
            loss_array  = per_sample_loss.cpu().flatten().tolist()
            
            event_data['score'] += score_array
            event_data['label'] += label_array
            event_data['loss'] += loss_array

            for key, item in other_vars.items():
                try:
                    event_data[key] += item.tolist() # do this if possible to convert tensor to list of floats
                except:
                    event_data[key] += list(item)

    del images
    torch.cuda.empty_cache()

    result_summary = getSummaryMetrics(event_data)
    
    # save to summary to pickle file
    result_file_path = os.path.join(result_dir_path, f'result_test.pickle')
    with open(result_file_path, 'wb') as pfile:
        pickle.dump(result_summary, pfile)
        print(f"Results saved to {result_file_path}")

    # save event data to csv file
    event_data_path = os.path.join(result_dir_path, f'event_data.csv')
    event_df = pd.DataFrame(event_data)
    event_df.to_csv(event_data_path, index=False)
    print(f"Event data saved to {event_data_path}")

    # plot
    if make_plots:
        plotting_tool.plotROC(result_summary['roc_fpr'], result_summary['roc_tpr'],
                        os.path.join(plot_dir_path, f"roc_curve.png"))
        plotting_tool.plotScoresFromDF(event_df,
                        os.path.join(plot_dir_path, f"score.png"))
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
    for isotope, pfilelist in CONFIG['input'].items():
        files_dict[isotope] = [str(filename.strip()) for filename in list(open(pfilelist, 'r')) if filename != ''][:CONFIG['max_num_files']]

    # ============= TRAIN ==============
    if CONFIG['run_mode'] == "train":
        # load dataset
        print(">>> Loading data...")
        dataset = KamNetDataset(files_dict, CONFIG['signal_isotope'],
            vars_to_output=CONFIG['output_vars'], elow=CONFIG['elow'], ehigh=CONFIG['ehigh'])
        dataset.printSize()

        # run Nhit matching
        if CONFIG['make_plots']:
            plotting_tool.plotNhitFromDataset(dataset, os.path.join(plot_dir_path, "Nhit_before_matching.png"))
        dataset.matchSBNhit()
        if CONFIG['make_plots']:
            plotting_tool.plotNhitFromDataset(dataset, os.path.join(plot_dir_path, "Nhit_after_matching.png"))

        if CONFIG['max_dataset_size']:
            dataset.downsize(CONFIG['max_dataset_size'])

        # split into training and validation datasets
        # ensure fair split of sig/bkg...
        sig_indices = list(np.random.permutation(dataset.getSignalIndices()))
        bkg_indices = list(np.random.permutation(dataset.getBackgroundIndices()))
        dataset_size = len(dataset) / 2 # should be the same for sig and bkg after nhit matching.
        split_index = int(np.floor(CONFIG['validation_split'] * dataset_size))

        train_sampler = SubsetRandomSampler(sig_indices[split_index:] + bkg_indices[split_index:])
        val_sampler = SubsetRandomSampler(sig_indices[:split_index] + bkg_indices[:split_index])
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