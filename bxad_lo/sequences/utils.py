import awkward as ak
import numpy as np

import operator
from functools import reduce

def orbit_aggregator(data: ak.Array, **kwargs):
    orbit_field = kwargs.get("orbit_field", "orbit")

    data_sorted = data[ak.argsort(data[orbit_field])]
    orbit_reps = ak.run_lengths(data_sorted[orbit_field])

    """
    unflatten data
    """
    data_uf = ak.unflatten(
        data_sorted, 
        orbit_reps
    )
    
    """
    group by orbit (gpo)
    """
    zip_dict = {key: data_uf[key] for key in data_uf.fields if key != orbit_field}
    zip_dict[orbit_field] = data_uf[orbit_field][:,0]
    data_gpo = ak.zip(zip_dict, depth_limit=1)

    return data_gpo

def reduce_and_tuple_array(arr: ak.Array):
    assert "0" in arr.fields, "Expected '0' within array fields. Make sure it is an array of tuples"
    
    return reduce(operator.and_, [arr[field] for field in arr.fields])