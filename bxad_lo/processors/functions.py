import numpy as np
import awkward as ak
import hist

def get_multiplicity_counts(data: ak.Array):
    mults, counts = np.unique(data, return_counts=True)

    return dict(zip(mults, counts))

def get_bin_counts(data: ak.Array, field: str, nbins: int = 50):
    assert field in data.fields, "Specified field not found in data.fields"

    min_data = ak.min(data[field]).astype(np.float32)
    max_data = ak.max(data[field]).astype(np.float32)

    h = hist.Hist(
            hist.axis.Regular(nbins, min_data, max_data, flow=False)
        ).fill(data[field])

    counts_dict = dict(zip(h.axes[0].edges[:-1], h.values()))

    return counts_dict