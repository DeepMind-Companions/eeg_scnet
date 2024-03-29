import mne
import numpy as np
from .channels import CHANNELS

def reduce_channels(raw):
    """
        Reducing the number of channels to the 21 channels in use
        Takes in raw data in mne format
        Returns raw data in mne format with 21 channels only
    """
    return raw.pick(CHANNELS)

def clip_data(raw, absclip):
    """
        Responsible for Clipping the data inside a fixed voltage range
        Inputs: raw EEG data in MNE format
        Outputs: raw EEG data clipped between -absclipx10^-6 and absclipx10^-6
    """
    return raw.apply_function(lambda data: np.clip(data, -0.000001*absclip, 0.000001*absclip))

def return_data(file):
    """
        Returns the data in mne format after preprocessing
        Inputs: file path
        Outputs: raw EEG data in mne format
    """
    # Starts by reading the file into the MNE format
    raw = mne.io.read_raw_edf(file, preload=True, verbose='CRITICAL')

    # Continues to reduce the channels to the specific ones
    raw = reduce_channels(raw)

    # Clipping the data to +-100 microvolts
    raw = clip_data(raw, 100)
    
    # Resampling the data to 100 Hz
    raw.resample(100)

    # Removing the first 60 seconds of the data
    raw.crop(tmin=60, tmax=480)

    return raw 
