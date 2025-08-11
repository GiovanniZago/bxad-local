import numpy as np
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 16})
import mplhep as hep
hep.style.use("CMS")

import config

def plot_multiplicity_counts(counts_dict: dict, **kwargs):
    kind = kwargs.get("kind", "bar")
    assert kind in ["bar", "step"], "Unknow value of kind, expected either 'bar' or 'step'"
    rate_norm = kwargs.get("rate_norm", None)
    xlabel = kwargs.get("xlabel", "x quantity")
    ylabel = kwargs.get("ylabel", "y quantity")
    yscale = kwargs.get("yscale", "linear")
    name = kwargs.get("name", "foo")

    xvalues = np.array(list(counts_dict.keys()))
    yvalues = np.array(list(counts_dict.values()))
    
    if rate_norm:
        yvalues = yvalues / rate_norm


    plt.figure(figsize=(12,10))
    hep.cms.label(label="Private Work", data=True, rlabel="Level-1 Trigger Scouring 2024 (13.6 TeV)", fontsize=18)

    match kind:
        case "bar":
            plt.bar(xvalues, yvalues)

        case "step":
            plt.step(xvalues, yvalues)
            

    plt.yscale(yscale)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(config.PLOT_OUTPUT_DIR + name + ".png")

def plot_bin_counts(counts_dict: dict, **kwargs):
    norm = kwargs.get("norm", None)
    rate_norm = kwargs.get("rate_norm", None)
    xlabel = kwargs.get("xlabel", "x quantity")
    ylabel = kwargs.get("ylabel", "y quantity")
    yscale = kwargs.get("yscale", "linear")
    name = kwargs.get("name", "foo")

    xvalues = np.array(list(counts_dict.keys()))
    yvalues = np.array(list(counts_dict.values()))

    match norm:
        case "density":
            raise NotImplementedError("Still to be done")
    
        case "rate": 
            assert rate_norm is not None, "You must provide rate_norm in case of rate normalization"
            yvalues = yvalues / rate_norm

    plt.figure(figsize=(12,10))
    hep.cms.label(label="Private Work", data=True, rlabel="Level-1 Trigger Scouring 2024 (13.6 TeV)", fontsize=18)

    width = np.min(np.diff(xvalues))
    plt.bar(x=xvalues, height=yvalues, width=width, align="edge")
    plt.yscale(yscale)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.savefig(config.PLOT_OUTPUT_DIR + name + ".png")