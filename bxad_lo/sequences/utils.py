import awkward as ak
import numpy as np

import operator
from functools import reduce

def orbit_aggregator(data: ak.Array):
    assert "orbit" in data.fields
    assert "bx" in data.fields
    
    data_sorted = data[ak.argsort(data.orbit)]
    orbit_reps = ak.run_lengths(data_sorted.orbit)

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
    zip_dict = {key: data_uf[key] for key in data_uf.fields if key != "orbit"}
    zip_dict["orbit"] = data_uf["orbit"][:,0]
    data_gpo = ak.zip(zip_dict, depth_limit=1)

    return data_gpo

def reduce_and_tuple_array(arr: ak.Array):
    assert "0" in arr.fields, "Expected '0' within array fields. Make sure it is an array of tuples"
    
    return reduce(operator.and_, [arr[field] for field in arr.fields])