import numpy as np
import awkward as ak
import hist

def get_multiplicity_counts(data: ak.Array):
    mults, counts = np.unique(data, return_counts=True)

    return dict(zip(mults, counts))

def get_bin_counts(data: ak.Array, nbins: int = 50):
    min_data = ak.min(data).astype(np.float32)
    max_data = ak.max(data).astype(np.float32)

    h = hist.Hist(
            hist.axis.Regular(nbins, min_data, max_data, flow=False)
        ).fill(data)

    counts_dict = dict(zip(h.axes[0].edges[:-1], h.values()))

    return counts_dict