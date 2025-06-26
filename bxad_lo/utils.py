import json
import numpy as np

def get_fill_data(json_file: str):
    with open(json_file) as f:
        fill_data = json.load(f)

    assert "schemebeam1" in fill_data, "'schemebeam1' is expected as key in the provided json data."

    return np.array(fill_data["schemebeam1"], dtype=np.uint32)