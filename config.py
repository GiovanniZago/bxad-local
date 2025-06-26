# data
DATA_PATH = ""
DATA_TTREE = "Events"
METADATA_TTREE = "LuminosityBlocks"

def set_data_path(path: str):
    global DATA_PATH
    DATA_PATH = path

def set_data_ttree(ttree: str):
    global DATA_TTREE
    DATA_TTREE = ttree 

def set_metadata_ttree(ttree: str):
    global METADATA_TTREE
    METADATA_TTREE = ttree 

# bx
T_DAQ = 22.33 # seconds

def set_t_daq(time: int):
    global T_DAQ
    T_DAQ = time

# plots
PLOT_OUTPUT_DIR = "plots/"

def set_plot_output_dir(path: str):
    global PLOT_OUTPUT_DIR
    PLOT_OUTPUT_DIR = path

