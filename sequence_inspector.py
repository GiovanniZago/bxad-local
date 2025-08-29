import uproot
import awkward as ak

import numpy as np
import pandas as pd
from tqdm import tqdm
import pickle

import pprint

from bxad_lo import sequences, processors, plots, utils
import config

if __name__ == "__main__":
    length = 3
    config.set_data_path(f"/eos/user/g/gizago/bxad-local/output_1000_seq{length}.root")
    config.set_data_ttree("L1BMTFStubSequences")
    config.set_plot_output_dir("/eos/user/g/gizago/php-plots/bxad-local-plots/")

    sequences_ttree = uproot.open(config.DATA_PATH + ":" + config.DATA_TTREE)
    pprint.pprint(sequences_ttree.show())

    # sequences_arrays = sequences_ttree.arrays()

    # counts = processors.get_multiplicity_counts(ak.ravel(sequences_arrays["L1BMTFStub_wheel"]))
    # plot_options = {
    #     "xlabel": "BMTF Stubs Wheel Multiplicity", 
    #     "ylabel": "Counts",
    #     "yscale": "log", 
    #     "name": f"{length}seq_wheel_mulcounts"
    # }

    # plots.plot_multiplicity_counts(counts, **plot_options)