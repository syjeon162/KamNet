'''
Wrapper for running KamNet
Use to train or validate
'''
import os
import argparse
import random
import pickle
import tomllib
from tqdm import tqdm

from sklearn.metrics import roc_auc_score, roc_curve
from tool import get_roc, get_rej, roc_nhit, cd

import torch.optim as optim
import torch.utils.data as data_utils
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

from KamNetDataset import KamNetDataset, KamNetDataset_Nhit, KamNetDatasetRep

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def useSeed(seed=7):
    '''
    Setting reproducability. If SEED=True, then training the neural network with
    the same configuration will result in exactly the same output
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

def load_data(batch_size, signal_dir, bkg_dir, elow, ehi, file_upperlim):
    '''
    Load datasets from various pickle list
    '''
    json_name = "event"

    data_list = getFilesUnderFolder(signal_dir, "pickle")
    datab_list = getFilesUnderFolder(bkg_dir, "pickle")
    dataset = KamNetDataset_Nhit(data_list[:file_upperlim], datab_list[:file_upperlim],
        str(json_name), dsize=-1, bootstrap=False, elow=elow, ehi=ehi)
    validation_split = .3
    shuffle_dataset = True
    random_seed= 42222

    division = 2
    dataset_size = int(len(dataset)/division)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        # Shuffle the dataset
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_indices += list(division*dataset_size - 1 - np.array(train_indices))
    val_indices += list(division*dataset_size - 1 - np.array(val_indices))

    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)

    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    # Convert dataset to data loader
    train_loader = data_utils.DataLoader(dataset, batch_size=batch_size, sampler=train_sampler,  drop_last=True)
    valid_loader = data_utils.DataLoader(dataset, batch_size=batch_size, sampler=valid_sampler,  drop_last=True)
    print(len(train_loader), len(val_loader))
    print(f"time_channel: {dataset.return_time_channel()}")

    return train_loader, valid_loader, dataset.return_time_channel()

def trainKamNet():
    return

def validateKamNet():
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('settings', type=str, description="Pass in .TOML file that contains KamNet settings to use.")
    args = parser.parse_args()

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    '''
    Training KamNet
    '''

    train_loader, val_loader, time_channel = load_data(BATCH_SIZE, args.sig, args.bkg, args.elow, args.ehi, args.filemax)

    classifier = KamNet(time_channel, param_dict)

    #=====================================================================================
    '''
    This part allows the loading of previously trained of KamNet using '.pt' model
    '''
    # pretrained_dict = torch.load('pretrain_data.pt')
    # model_dict = classifier.state_dict()
    # model_dict.update(pretrained_dict) 
    # classifier.load_state_dict(pretrained_dict)
    #=====================================================================================

    classifier.to(DEVICE)

    print("#params", sum(x.numel() for x in classifier.parameters()))

    '''
    Define the loss function
    '''
    criterion = nn.BCEWithLogitsLoss()
    criterion = criterion.to(DEVICE)

    #=====================================================================================
    '''
    Set up optimizer with varying learning rate:
      Ramp Up   : Gradually ramp up learning rate in the first 5 epochs, this allows the attention mechanism to learn proper attention score
      Flat    : Fix the learning rate at the nominal value
      Ramp Down : Ramp down the learning rate to 10% of nominal value in the last 10th - 5th epochs
      Flat    : Fix the learning rate at 10% of the nominal value for the last 5 epochs
    '''
    step_length = len(train_loader)
    total_step = int(NUM_EPOCHS * step_length)
    ramp_up = np.linspace(1e-4, 1.0, 5*step_length)
    ramp_down = list(np.linspace(1.0, 0.1, 5*step_length).flatten()) + [0.1]* 5*step_length
    ramp_down_start = total_step - len(ramp_down)
    lmbda = lambda epoch: ramp_up[epoch] if epoch<len(ramp_up) else ramp_down[epoch-ramp_down_start-1] if epoch > ramp_down_start else 1.0
    optimizer = torch.optim.RMSprop(classifier.parameters(),lr=param_dict["lr"], momentum=param_dict["momentum"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lmbda)
    #=====================================================================================

    # dictionary for saving to pickle
    datadict = {}
    for key in ['tsigmoid_s', 'tsigmoid_b', 'sigmoid_s', 'sigmoid_b',
                't_rej_eff', 'rej_eff', 'auc', 't_loss', 'v_loss']:
        datadict[key] = []
        
    for epoch in tqdm(range(NUM_EPOCHS)):
        loss_i = 0
        tsigmoid_s = []
        tsigmoid_b = []
        print(scheduler.get_lr())

        # =========== TRAIN =============
        for i, (images, labels) in tqdm(enumerate(train_loader)):
            classifier.train()
            images = images.to(DEVICE)
            labels = labels.view(-1,1)
            labels = labels.to(DEVICE).float()

            outputs  = classifier(images)
            loss = criterion(outputs,labels)
            
            lb_data = labels.cpu().data.numpy().flatten()
            outpt_data = outputs.cpu().data.numpy().flatten()
            signal = np.argwhere(lb_data == 1.0)
            bkg = np.argwhere(lb_data == 0.0)
            tsigmoid_s += list(outpt_data[signal].flatten())
            tsigmoid_b += list(outpt_data[bkg].flatten())

            loss.backward()   # optimizer the net
            optimizer.step()        # update parameters of net
            optimizer.zero_grad()   # reset gradient
            scheduler.step()
            loss_i += loss.item()
            # print('\rEpoch [{0}/{1}], Iter [{2}/{3}] Loss: {4:.4f}'.format(
            #     epoch+1, NUM_EPOCHS, i+1, len(train_loader),
            #     loss.item(), end=""))
        datadict['t_loss'].append(loss_i/len(train_loader))
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {datadict['t_loss'][-1]:.4f}")

        # =========== VALIDATE =============
        sigmoid_s = []
        sigmoid_b = []
        loss_i = 0
        for (images, labels) in val_loader:
            classifier.eval()

            with torch.no_grad():
                images = images.to(DEVICE)
                labels = labels.view(-1,1)
                labels = labels.to(DEVICE).float()

                outputs  = classifier(images).view(-1,1)

                loss = criterion(outputs, labels)
                loss_i += loss.item()

                image_data = images.cpu().data.numpy().reshape(BATCH_SIZE,-1)
                lb_data = labels.cpu().data.numpy().flatten()
                outpt_data = outputs.cpu().data.numpy().flatten()

                signal = np.argwhere(lb_data == 1)
                bkg = np.argwhere(lb_data == 0)
                sigmoid_s += list(outpt_data[signal].flatten())
                sigmoid_b += list(outpt_data[bkg].flatten())
        datadict['v_loss'].append(loss_i/len(val_loader))

        # Rejection Efficiency calculation Training
        cut = np.percentile(tsigmoid_s, 10)
        reject = 0
        for i in tsigmoid_b:
            if i<cut:
                reject += 1
        rej_eff = 100*reject/len(tsigmoid_b)
        datadict['t_rej_eff'].append(rej_eff)
        print(f"training rejection efficiency: {rej_eff}")

        # Validation rejection
        cut = np.percentile(sigmoid_s, 10)
        reject = 0
        for i in sigmoid_b:
            if i<cut:
                reject += 1
        rej_eff = 100*reject/len(sigmoid_b)
        datadict['rej_eff'].append(rej_eff)
        print(f"validation rejection efficiency: {rej_eff}")

        # Area Under ROC Curve
        auc_labels = np.concatenate((np.ones(len(sigmoid_s)), np.zeros(len(sigmoid_b))))
        auc_scores = np.concatenate((sigmoid_s, sigmoid_b))
        auc_i = roc_auc_score(auc_labels, auc_scores)
        datadict['auc'].append(auc_i)
        print('AUC:', auc_i)

        datadict['tsigmoid_s'].append(tsigmoid_s)
        datadict['tsigmoid_b'].append(tsigmoid_b)
        datadict['sigmoid_s'].append(sigmoid_s)
        datadict['sigmoid_b'].append(sigmoid_b)
        
        del images
        torch.cuda.empty_cache()
        torch.save(classifier.state_dict(), f'{args.outdir}/KamNet_epoch{epoch}.pt')    # Save KamNet parameters in KamNet.pt file
    torch.save(classifier.state_dict(), f'{args.outdir}/KamNet_final.pt')

    # save results
    with open(f'{args.outdir}/results.p', 'wb') as pfile:
        pickle.dump(datadict, pfile)