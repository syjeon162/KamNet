import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

plt.rcParams['font.size'] = 15
plt.rcParams["axes.prop_cycle"] = cycler('color', ["#0077BB", "#CC3311", "#009988", "#EE7733", "#33BBEE", "#EE3377", "#BBBBBB"])

def plotNhit(nhit_dict, figpath):
    fig, ax = plt.subplots(figsize=[8,6])
    bins = np.arange(0, 700+5, 5)

    for key, val in nhit_dict.items():
        plt.hist(val, label=key, bins=bins, histtype="step")
    
    plt.xlim(bins[0], bins[-2])
    plt.xlabel("Nhit")
    plt.grid()
    leg = plt.legend()
    leg.get_frame().set_linewidth(0.0)

    plt.savefig(figpath)
    plt.close()

    print(f"Plot saved to {figpath}")

def plotNhitFromDataset(dataset, figpath):
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
    
    plotNhit(nhits, figpath)
    
def plotROC(fpr, tpr, figpath):
    fig, ax = plt.subplots(figsize=[6,6])

    plt.plot(fpr, tpr)
    plt.axline((0, 0), slope=1, color='grey', linestyle='--')

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.grid()

    plt.savefig(figpath)
    plt.close()

    print(f"Plot saved to {figpath}")

def plotScores(scores_dict, figpath):
    fig, ax = plt.subplots(figsize=[8,6])
    bins = np.arange(-10, 10, 0.2)

    for label, score in scores_dict.items():
        plt.hist(score, label=label, bins=bins, histtype="step", density=True)
    
    plt.xlim(-10, 10)
    plt.xlabel("KamNet Scores")
    plt.grid()
    leg = plt.legend()
    leg.get_frame().set_linewidth(0.0)

    plt.savefig(figpath)
    plt.yscale("log")
    plt.savefig(figpath.replace(".png", "_log.png"))
    plt.close()

def plotScoresFromDF(df, figpath):
    scores_dict = {}
    try:
        labels = np.sort(np.unique(df['isotope'])).tolist()
        for l in labels:
            scores_dict[l] = df.loc[df['isotope'] == l, 'score'].to_numpy().flatten()
    except:
        scores_dict["Signal"] = df.loc[df['label'] == 1, 'score'].to_numpy().flatten()
        scores_dict["Background"] = df.loc[df['label'] == 0, 'score'].to_numpy().flatten()

    plotScores(scores_dict, figpath)