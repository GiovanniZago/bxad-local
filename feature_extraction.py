from bxad_lo import sequences, processors, plots, utils
import config

import uproot
import numpy as np
import awkward as ak

import os
import psutil

if __name__ == "__main__":
    """
    Setup global variables and read data
    """
    process = psutil.Process(os.getpid())

    config.set_data_path("/eos/cms/store/cmst3/group/daql1scout/run3/ntuples/zb/run383996/output_1000.root")
    config.set_plot_output_dir("/eos/user/g/gizago/php-plots/bxad-local-plots/")

    events = uproot.open(config.DATA_PATH + ":" + config.DATA_TTREE)
    events_metadata = uproot.open(config.DATA_PATH + ":" + config.METADATA_TTREE)

    data = events.arrays(
        ["orbit", "bx", "num_stubs", "quality", "phi_rel", "phi_bend", "eta_hits", "eta_qual", "wheel", "sector", "station"], 
        aliases={
            "orbit": "orbitNumber", 
            "bx": "bunchCrossing", 
            "num_stubs": "nL1BMTFStub", 
            "quality": "L1BMTFStub_hwQual", 
            "phi_rel": "L1BMTFStub_hwPhi", 
            "phi_bend": "L1BMTFStub_hwPhiB", 
            "eta_hits": "L1BMTFStub_hwEta", 
            "eta_qual": "L1BMTFStub_hwQual", 
            "wheel": "L1BMTFStub_wheel", 
            "sector": "L1BMTFStub_sector", 
            "station": "L1BMTFStub_station"
            }
    )
    print(f"Current memory usage: {process.memory_info().rss / 1e6:.3f} MB")

    """
    Global data info
    """
    num_tot_orbits = np.ravel(events_metadata["nOrbits"].arrays())[0]

    t_daq = num_tot_orbits * 25e-9 * 3564
    print(f"T daq: {t_daq:.3f} s")

    num_tot_bx = ak.num(data.bx, axis=0)
    print(f"Tot bx: {num_tot_bx}")

    effective_rate = num_tot_bx / t_daq
    print(f"Effective rate: {(effective_rate / 1e6):.2f} MHz")

    """
    Auxiliary data on colliding bunches
    """
    cbx_mask = utils.get_fill_data("25ns_2352b_2340_2004_2133_108bpi_24inj.json")
    cbx_array = np.flatnonzero(cbx_mask)

    """
    Filter data: at least one BMTF stub and bx is a colliding bunch
    """
    data_filtered = data[data.num_stubs >= 1]
    mask = np.isin(data_filtered.bx, cbx_array)
    data_filtered = data_filtered[mask]

    """
    Restructure data
    """
    data_gpo = sequences.orbit_aggregator(data_filtered)
    print(f"Current memory usage: {process.memory_info().rss / 1e6:.3f} MB")

    """
    Find sequences
    """
    data_gpo = sequences.get_bx_sequences(data_gpo, length=3)
    print(f"Current memory usage: {process.memory_info().rss / 1e6:.3f} MB")

    """
    Restructure sequences, making them lists and not tuples
    """
    seq = ak.to_list(data_gpo.seq)
    seq_lists = [list(map(lambda tp: list(tp), ls)) for ls in seq]
    seq_array = ak.Array(seq_lists)
    data_gpo = ak.with_field(data_gpo, seq_array, where="seq_array")

    print(f"Current memory usage: {process.memory_info().rss / 1e6:.3f} MB") 


    """ 
    Flatten sequences and keep for each orbit only the bxs
    that belong to seq_flatten
    """
    bx = ak.to_list(data_gpo.bx)
    seq_flatten = ak.to_list(ak.flatten(data_gpo.seq_array, axis=2))

    # remove fields to allow for slicing
    data_gpo = ak.without_field(data_gpo, ["seq"])
    data_gpo = ak.without_field(data_gpo, ["seq_array"])
    data_gpo = ak.without_field(data_gpo, ["orbit"])
    data_gpo = ak.without_field(data_gpo, ["num_stubs"])

    bx_seq_mask = [[True if el in seqls else False for el in bxls] for bxls, seqls in zip(bx, seq_flatten)]
    bx_seq_mask_array = ak.Array(bx_seq_mask)
    data_gpo = data_gpo[bx_seq_mask_array]

    print(f"Current memory usage: {process.memory_info().rss / 1e6:.3f} MB") 

    for field in data_gpo.fields:
        print(f"{field}: ", data_gpo[0][field])
  