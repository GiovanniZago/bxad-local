import config
from bxad_lo import utils
from bxad_lo.sequences.utils import orbit_aggregator

import uproot
import numpy as np
import awkward as ak

import os 
import pprint

def shuffle_orbit_sequences(orbit_dict):
    shuffle_pattern = np.random.permutation(len(orbit_dict["bunchCrossing"]))
    shuffled_bxs = ak.Array(orbit_dict["bunchCrossing"])[shuffle_pattern]

    while np.any(np.diff(shuffled_bxs) == 1):
        shuffle_pattern = np.random.permutation(len(orbit_dict["bunchCrossing"]))
        shuffled_bxs = ak.Array(orbit_dict["bunchCrossing"])[shuffle_pattern]
        
    shuffled_orbit_dict = {}

    for key in orbit_dict:
        if key == "orbitNumber":
            continue
        
        shuffled_orbit_dict[key] = ak.Array(orbit_dict[key])[shuffle_pattern].to_list()

    shuffled_orbit_dict["orbitNumber"] = orbit_dict["orbitNumber"]
    return shuffled_orbit_dict

def extract_orbit_sequences(orbit_dict, sequence_length=3):
    nbx = len(orbit_dict["bunchCrossing"])
    nseq = nbx // sequence_length
    mask = np.arange(nbx).reshape((nseq, sequence_length))

    sequence_dict = {}

    for key, values in orbit_dict.items():
        if key == "orbitNumber":
            continue

        values_ak = ak.Array(values)
        sequence_dict[key] = values_ak[mask].to_list()

    orbit_number = orbit_dict["orbitNumber"] * np.ones((nseq, sequence_length), dtype=np.int32)
    sequence_dict["orbitNumber"] = orbit_number.tolist()
    return sequence_dict 

def bx_filter(sequence_dict, train_length=24):
    bx_diff = np.abs(np.diff(sequence_dict["bunchCrossing"]))

    return np.all(bx_diff >= train_length)

if __name__ == "__main__":
    config.set_data_path("output_1000_seq3.root")
    config.set_data_ttree("L1BMTFStubSequences")
    config.set_plot_output_dir("/eos/user/g/gizago/php-plots/bxad-local-plots/")

    sequences = uproot.open(config.DATA_PATH + ":" + config.DATA_TTREE).arrays()    

    sequence_length = 3
    train_length = 24

    # aggregate by orbit
    sequences_gpo = orbit_aggregator(sequences, orbit_field="orbitNumber")
    sequences_gpo = sequences_gpo.to_list()


    # shuffle bxs within the same orbit
    sequences_shuffled_gpo = list(
        map(
            shuffle_orbit_sequences, 
            sequences_gpo
        )
    )

    # rearrange shuffled sequences always keeping the data grouped by orbit
    sequences_shuffled_gpo = list(
        map(
            lambda x: extract_orbit_sequences(x, sequence_length=sequence_length), 
            sequences_shuffled_gpo
        )
    )

    # ungroup data in order to obtain an array of records where each record corresponds to a sequence
    sequences_shuffled = ak.concatenate([ak.Array(d) for d in sequences_shuffled_gpo]).to_list()

    # filter sequences by looking at the absolute difference between consecutive bxs
    # if the difference is smaller than the train length, discard the sequence, otherwise keep it
    sequences_shuffled = list(
        filter(
            lambda x: bx_filter(x, train_length=train_length),
            sequences_shuffled
        )
    )

    # update sequenceIndex
    for index, sequence_dict in enumerate(sequences_shuffled):
        sequence_dict["sequenceIndex"] = sequence_length * [index]

    # flatten sequences to single bx in order to ntuplize
    sequences_shuffled_flat = ak.concatenate([ak.Array(d) for d in sequences_shuffled])

    # create root file
    L1BMTFStub = ak.zip(
        {
            name[11:]: array for name, array in zip(ak.fields(sequences_shuffled_flat), ak.unzip(sequences_shuffled_flat)) if name.startswith("L1BMTFStub_")
        }
    )

    file_out = uproot.recreate(f"output_1000_seq{sequence_length}s.root")
    file_out["L1BMTFStubSequences"] = {
        "L1BMTFStub": L1BMTFStub, 
        "orbitNumber": sequences_shuffled_flat["orbitNumber"], 
        "bunchCrossing": sequences_shuffled_flat["bunchCrossing"], 
        "sequenceIndex": sequences_shuffled_flat["sequenceIndex"]
    }