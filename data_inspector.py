import config

import uproot
import numpy as np
import awkward as ak
import pandas as pd

from pprint import pprint

if __name__ == "__main__":
    config.set_data_path("output_1000_seq3s.root")
    config.set_plot_output_dir("/eos/user/g/gizago/php-plots/bxad-local-plots/")

    data = uproot.open(config.DATA_PATH + ":L1BMTFStubSequences").arrays()
    df = ak.to_dataframe(data)

    df["L1BMTFStub_hwQEta"].plot(
        kind="hist", 
        bins=np.arange(128),
        ylabel="Counts",
        logy=True
        ).get_figure().savefig(config.PLOT_OUTPUT_DIR + "seq3s_hwQEta.png")
