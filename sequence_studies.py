from bxad_lo import sequences, processors, plots, utils
import config

import uproot
import numpy as np
import awkward as ak

import os
import psutil

if __name__ == "__main__":
    process = psutil.Process(os.getpid())

    config.set_data_path("/eos/cms/store/cmst3/group/daql1scout/run3/ntuples/zb/run383996/output_1000.root")
    config.set_plot_output_dir("/eos/user/g/gizago/php-plots/bxad-local-plots/")

    events = uproot.open(config.DATA_PATH + ":" + config.DATA_TTREE)
    events_metadata = uproot.open(config.DATA_PATH + ":" + config.METADATA_TTREE)

    data = events.arrays(
        ["orbit", "bx", "num_stubs"], 
        aliases={"orbit": "orbitNumber", "bx": "bunchCrossing", "num_stubs": "nL1BMTFStub"}
    )
    print(f"Current memory usage: {process.memory_info().rss / 1e6:.3f} MB")

    num_tot_orbits = np.ravel(events_metadata["nOrbits"].arrays())[0]

    t_daq = num_tot_orbits * 25e-9 * 3564
    print(f"T daq: {t_daq:.3f} s")

    num_tot_bx = ak.num(data.bx, axis=0)
    print(f"Tot bx: {num_tot_bx}")

    effective_rate = num_tot_bx / t_daq
    print(f"Effective rate: {(effective_rate / 1e6):.2f} MHz")

    """ 
    Stubs multiplicity
    """
    stub_mulcounts = processors.get_multiplicity_counts(data.num_stubs)

    plot_options = {
        "xlabel": "BMTF Stubs Multiplicity", 
        "ylabel": "Counts",
        "rate_norm": t_daq, 
        "yscale": "log", 
        "name": "stubs_mulcounts"
    }
    plots.plot_multiplicity_counts(stub_mulcounts, **plot_options)
    print(f"Current memory usage: {process.memory_info().rss / 1e6:.3f} MB")

    """
    Sequences
    """
    data_gpo = sequences.orbit_aggregator(data)
    data_seq = sequences.get_bx_sequences(data_gpo, length=2)
    print(f"Current memory usage: {process.memory_info().rss / 1e6:.3f} MB")
    
    num_seq_per_orbit = ak.num(data_seq, axis=1)
    num_seq_mulcounts = processors.get_multiplicity_counts(num_seq_per_orbit)
    print(f"Current memory usage: {process.memory_info().rss / 1e6:.3f} MB")

    plot_options = {
        "kind": "step",
        "xlabel": "2-sequence multiplicity",
        "ylabel": "Orbit counts",
        "yscale": "log", 
        "name": "2seq_mulcounts"
    }
    plots.plot_bin_counts(num_seq_mulcounts, **plot_options)
    

    """ 
    Sequences in colliding bunches with at least one BMTF stub
    """
    cbx_mask = utils.get_fill_data("25ns_2352b_2340_2004_2133_108bpi_24inj.json")
    cbx_array = np.flatnonzero(cbx_mask)

    mask = np.isin(data.bx, cbx_array)
    data_cbx = data[mask]
    data_cbx_stubs = data_cbx[data_cbx.num_stubs >= 1]
    print(f"Current memory usage: {process.memory_info().rss / 1e6:.3f} MB")

    data_gpo = sequences.orbit_aggregator(data_cbx_stubs)
    seq_tot_counts = {}
    
    # 2-sequences
    data_seq = sequences.get_bx_sequences(data_gpo, length=2)
    print(f"Current memory usage: {process.memory_info().rss / 1e6:.3f} MB")
    
    num_seq_per_orbit = ak.num(data_seq, axis=1)
    num_seq_mulcounts = processors.get_multiplicity_counts(num_seq_per_orbit)
    seq_tot_counts["2"] = sum([key * num_seq_mulcounts[key] for key in num_seq_mulcounts])
    print(f"Current memory usage: {process.memory_info().rss / 1e6:.3f} MB")

    plot_options = {
        "kind": "step",
        "xlabel": "2-sequence multiplicity",
        "ylabel": "Orbit counts",
        "yscale": "log", 
        "name": "2seq_mulcounts_cb_stubs"
    }
    plots.plot_bin_counts(num_seq_mulcounts, **plot_options)

    # 3-sequences
    data_seq = sequences.get_bx_sequences(data_gpo, length=3)
    print(f"Current memory usage: {process.memory_info().rss / 1e6:.3f} MB")
    
    num_seq_per_orbit = ak.num(data_seq, axis=1)
    num_seq_mulcounts = processors.get_multiplicity_counts(num_seq_per_orbit)
    seq_tot_counts["3"] = sum([key * num_seq_mulcounts[key] for key in num_seq_mulcounts])
    print(f"Current memory usage: {process.memory_info().rss / 1e6:.3f} MB")

    plot_options = {
        "kind": "step",
        "xlabel": "3-sequence multiplicity",
        "ylabel": "Orbit counts",
        "yscale": "log", 
        "name": "3seq_mulcounts_cb_stubs"
    }
    plots.plot_bin_counts(num_seq_mulcounts, **plot_options)

    # 4-sequences
    data_seq = sequences.get_bx_sequences(data_gpo, length=4)
    print(f"Current memory usage: {process.memory_info().rss / 1e6:.3f} MB")
    
    num_seq_per_orbit = ak.num(data_seq, axis=1)
    num_seq_mulcounts = processors.get_multiplicity_counts(num_seq_per_orbit)
    seq_tot_counts["4"] = sum([key * num_seq_mulcounts[key] for key in num_seq_mulcounts])
    print(f"Current memory usage: {process.memory_info().rss / 1e6:.3f} MB")

    plot_options = {
        "kind": "step",
        "xlabel": "4-sequence multiplicity",
        "ylabel": "Orbit counts",
        "yscale": "log", 
        "name": "4seq_mulcounts_cb_stubs"
    }
    plots.plot_bin_counts(num_seq_mulcounts, **plot_options)

    # 5-sequences
    data_seq = sequences.get_bx_sequences(data_gpo, length=5)
    print(f"Current memory usage: {process.memory_info().rss / 1e6:.3f} MB")
    
    num_seq_per_orbit = ak.num(data_seq, axis=1)
    num_seq_mulcounts = processors.get_multiplicity_counts(num_seq_per_orbit)
    seq_tot_counts["5"] = sum([key * num_seq_mulcounts[key] for key in num_seq_mulcounts])
    print(f"Current memory usage: {process.memory_info().rss / 1e6:.3f} MB")

    plot_options = {
        "kind": "step",
        "xlabel": "5-sequence multiplicity",
        "ylabel": "Orbit counts",
        "yscale": "log", 
        "name": "5seq_mulcounts_cb_stubs"
    }
    plots.plot_bin_counts(num_seq_mulcounts, **plot_options)

    # 6-sequences
    data_seq = sequences.get_bx_sequences(data_gpo, length=6)
    print(f"Current memory usage: {process.memory_info().rss / 1e6:.3f} MB")
    
    num_seq_per_orbit = ak.num(data_seq, axis=1)
    num_seq_mulcounts = processors.get_multiplicity_counts(num_seq_per_orbit)
    seq_tot_counts["6"] = sum([key * num_seq_mulcounts[key] for key in num_seq_mulcounts])
    print(f"Current memory usage: {process.memory_info().rss / 1e6:.3f} MB")

    plot_options = {
        "kind": "step",
        "xlabel": "6-sequence multiplicity",
        "ylabel": "Orbit counts",
        "yscale": "log", 
        "name": "6seq_mulcounts_cb_stubs"
    }
    plots.plot_bin_counts(num_seq_mulcounts, **plot_options)

    # 7-sequences
    data_seq = sequences.get_bx_sequences(data_gpo, length=7)
    print(f"Current memory usage: {process.memory_info().rss / 1e6:.3f} MB")
    
    num_seq_per_orbit = ak.num(data_seq, axis=1)
    num_seq_mulcounts = processors.get_multiplicity_counts(num_seq_per_orbit)
    seq_tot_counts["7"] = sum([key * num_seq_mulcounts[key] for key in num_seq_mulcounts])
    print(f"Current memory usage: {process.memory_info().rss / 1e6:.3f} MB")

    plot_options = {
        "kind": "step",
        "xlabel": "7-sequence multiplicity",
        "ylabel": "Orbit counts",
        "yscale": "log", 
        "name": "7seq_mulcounts_cb_stubs"
    }
    plots.plot_bin_counts(num_seq_mulcounts, **plot_options)

    # 8-sequences
    data_seq = sequences.get_bx_sequences(data_gpo, length=8)
    print(f"Current memory usage: {process.memory_info().rss / 1e6:.3f} MB")
    
    num_seq_per_orbit = ak.num(data_seq, axis=1)
    num_seq_mulcounts = processors.get_multiplicity_counts(num_seq_per_orbit)
    seq_tot_counts["8"] = sum([key * num_seq_mulcounts[key] for key in num_seq_mulcounts])
    print(f"Current memory usage: {process.memory_info().rss / 1e6:.3f} MB")

    plot_options = {
        "kind": "step",
        "xlabel": "8-sequence multiplicity",
        "ylabel": "Orbit counts",
        "yscale": "log", 
        "name": "8seq_mulcounts_cb_stubs"
    }
    plots.plot_bin_counts(num_seq_mulcounts, **plot_options)

    # 9-sequences
    data_seq = sequences.get_bx_sequences(data_gpo, length=9)
    print(f"Current memory usage: {process.memory_info().rss / 1e6:.3f} MB")
    
    num_seq_per_orbit = ak.num(data_seq, axis=1)
    num_seq_mulcounts = processors.get_multiplicity_counts(num_seq_per_orbit)
    seq_tot_counts["9"] = sum([key * num_seq_mulcounts[key] for key in num_seq_mulcounts])
    print(f"Current memory usage: {process.memory_info().rss / 1e6:.3f} MB")

    plot_options = {
        "kind": "step",
        "xlabel": "9-sequence multiplicity",
        "ylabel": "Orbit counts",
        "yscale": "log", 
        "name": "9seq_mulcounts_cb_stubs"
    }
    plots.plot_bin_counts(num_seq_mulcounts, **plot_options)

    # 10-sequences
    data_seq = sequences.get_bx_sequences(data_gpo, length=10)
    print(f"Current memory usage: {process.memory_info().rss / 1e6:.3f} MB")
    
    num_seq_per_orbit = ak.num(data_seq, axis=1)
    num_seq_mulcounts = processors.get_multiplicity_counts(num_seq_per_orbit)
    seq_tot_counts["10"] = sum([key * num_seq_mulcounts[key] for key in num_seq_mulcounts])
    print(f"Current memory usage: {process.memory_info().rss / 1e6:.3f} MB")

    plot_options = {
        "kind": "step",
        "xlabel": "10-sequence multiplicity",
        "ylabel": "Orbit counts",
        "yscale": "log", 
        "name": "10seq_mulcounts_cb_stubs"
    }
    plots.plot_bin_counts(num_seq_mulcounts, **plot_options)

    # plot total multiplicity per sequence length
    plot_options = {
        "kind": "bar",
        "xlabel": "sequence length",
        "ylabel": "Counts",
        "yscale": "log", 
        "name": "seq_tot_counts_cb_stubs"
    }
    plots.plot_multiplicity_counts(seq_tot_counts, **plot_options)
    
