import awkward as ak

from .utils import reduce_and_tuple_array

def get_bx_sequences(data: ak.Array, length: int = 2):
    """
    expects orbit-aggregated data
    """

    assert "orbit" in data.fields
    assert "bx" in data.fields
    assert length >= 2
    
    bxs_shifted = [data.bx[:, i:(-length + i + 1)] for i in range(length - 1)]
    bxs_shifted.append(data.bx[:, (length - 1):])
    sequences = ak.zip(bxs_shifted)
    
    bxs_diff = [bxs_shifted[i + 1] - bxs_shifted[i] for i in range(length - 1)]
    bxs_masks = [diff == 1 for diff in bxs_diff]
    masks = ak.zip(bxs_masks) 
    
    """
    The structure of masks is like
    [
        [(True, False, False), (True, False, True), ...]
        [(...), ...]
        ...
    ]

    because zipping a list generates an array of tuples (...) which are
    actually records with automatic field naming "0", "1", ... up to the
    number of elements (-1) of each tuple. We have to and-reduce each tuple.
    """
    masks_reduced = reduce_and_tuple_array(masks)

    return sequences[masks_reduced]