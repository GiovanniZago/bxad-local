from bxad_lo import sequences, processors, plots, utils
import config

import uproot
import numpy as np
import awkward as ak

import os
import psutil
import pprint

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
        ["orbit", "bx", "num_stubs", "L1BMTFStub_hwQual", "L1BMTFStub_hwPhi", 
         "L1BMTFStub_hwPhiB", "L1BMTFStub_hwEta", "L1BMTFStub_hwQEta", 
         "L1BMTFStub_wheel", "L1BMTFStub_sector", "L1BMTFStub_station"], 
        aliases={
            "orbit": "orbitNumber", 
            "bx": "bunchCrossing", 
            "num_stubs": "nL1BMTFStub", 
            "L1BMTFStub_hwQual": "L1BMTFStub_hwQual", 
            "L1BMTFStub_hwPhi": "L1BMTFStub_hwPhi", 
            "L1BMTFStub_hwPhiB": "L1BMTFStub_hwPhiB", 
            "L1BMTFStub_hwEta": "L1BMTFStub_hwEta", 
            "L1BMTFStub_hwQEta": "L1BMTFStub_hwQEta", 
            "L1BMTFStub_wheel": "L1BMTFStub_wheel", 
            "L1BMTFStub_sector": "L1BMTFStub_sector", 
            "L1BMTFStub_station": "L1BMTFStub_station"
        }
    )
    print(f"Current memory usage: {process.memory_info().rss / 1e6:.3f} MB")

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
    length = 5
    seq_record_array = sequences.get_bx_sequences(data_gpo, length=length)
    print(f"Current memory usage: {process.memory_info().rss / 1e6:.3f} MB")

    """
    Restructure sequences, making them lists and not tuples
    """
    seq_tuple_lists = ak.to_list(seq_record_array)
    seq = [list(map(lambda tp: list(tp), ls)) for ls in seq_tuple_lists]

    print(f"Current memory usage: {process.memory_info().rss / 1e6:.3f} MB")

    """
    Now for each bx sequence we want to extract, 
    on a per-orbit basis, the indexes of their 
    bxs in the full bx array. 

    Eg:
    bx_in_orbit = [1, 2, 3, 5, 7, 8, 9, 10]

    seq_in_orbit = [[1, 2, 3], [8, 9, 10]]

    -> we want to get
    [[0, 1, 2], [5, 6, 7]]

    Such list, for each orbit, is stored
    inside seq_indexes
    """
    bx = ak.to_list(data_gpo.bx)

    def find_indexes(array, array_elements):
        result = []
        for target in array_elements:
            length = len(target)
            for i in range(len(array) - length + 1):
                if array[i:i + length] == target:
                    result.append(list(range(i, i + length)))
                    break  
        return result
    
    seq_indexes = [
        find_indexes(bx_list, bx_seq_list) 
        for bx_list, bx_seq_list 
        in zip(bx, seq)
    ]

    """
    Now that we have seq_indexes we want 
    to slice each feature vector, on a per-orbit
    basis, in order to keep the values of the
    features only for the bxs contained inside
    the sequences
    """
    data_gpo_dict = data_gpo.to_list()

    last_seq_idx = 0
    for indexes, orbit_dict in zip(seq_indexes, data_gpo_dict):
        index_array = ak.Array(indexes)
        num_seq_in_orbit = ak.num(index_array, axis=0).item()

        for key, values in orbit_dict.items():
            if key == "orbit":
                continue

            arr = ak.Array(values)
            orbit_dict[key] = [ak.to_list(arr[idxs]) for idxs in index_array]

        orbit_dict["seq_idx"] = list(range(last_seq_idx, last_seq_idx + num_seq_in_orbit))
        last_seq_idx += num_seq_in_orbit

    data_gpo = ak.Array(data_gpo_dict)
    data_gpo["orbit"] = ak.broadcast_arrays(data_gpo["bx"], data_gpo["orbit"])[-1]
    data_gpo["seq_idx"] = ak.broadcast_arrays(data_gpo["bx"], data_gpo["seq_idx"])[-1]

    print("DATA_GPO")
    for record in data_gpo[0:5]:
        pprint.pprint(record.to_list())

    selected_fields = ["bx", "num_stubs", "orbit", "seq_idx"]

    data_sequences = ak.zip(
        {
            field: ak.flatten(data_gpo[field], axis=-1) for field in selected_fields
        }
    )

    for field in data_gpo.fields:
        if field in selected_fields:
            continue

        data_sequences[field] = ak.flatten(data_gpo[field], axis=-2)

    data_sequences = ak.flatten(data_sequences)

    print("\n\n\nDATA_SEQUENCES")
    for record in data_sequences[0:5]:
        pprint.pprint(record.to_list())

    """
    Fix the data format to match 
    the silly Uproot requirements
    """

    L1BMTFStub = ak.zip({name[11:]: array for name, array in zip(ak.fields(data_sequences), ak.unzip(data_sequences)) if name.startswith("L1BMTFStub_")})

    file_out = uproot.recreate(f"output_1000_seq{length}.root")
    file_out["L1BMTFStubSequences"] = {
        "L1BMTFStub": L1BMTFStub, 
        "orbitNumber": data_sequences["orbit"], 
        "bunchCrossing": data_sequences["bx"], 
        "sequenceIndex": data_sequences["seq_idx"]
    }