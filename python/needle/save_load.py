"""Save and load Needle state dict to/from a file."""

import numpy as np
from . import init

def save_state_dict(module, filename):
    """
    Save the state dict of a module to a file.

    Args:
        module (ndl.nn.Module): The module to save.
        filename (str): The file to save to.
    """
    # Create a dictionary to store the state dict and signature of the module
    save_dict = {}
    save_dict['state_dict'] = module.state_dict()

    # Convert the state dict to numpy arrays and store them in the dictionary
    for k, v in save_dict['state_dict'].items():
        save_dict['state_dict'][k] = v.numpy().astype(np.float32)
    # Store the signature of the module in the dictionary
    save_dict['signature'] = str(module.id())

    # Store the device of the module in the dictionary
    save_dict['device'] = str(module.device)

    # Save the dictionary to the specified file using numpy
    np.save(filename, save_dict)
    print("Saved state dict to file: {}".format(filename))


def load_state_dict(module, filename):
    """
    Load the state dict from a file.

    Args:
        module: (ndl.nn.Module): The module to load the state dict to.
        filename (str): The file to load state dict from.
    """
    # Load the dictionary containing the state dict and signature of the module from the file
    save_dict = np.load(filename, allow_pickle=True).item()

    # Check if the signature of the loaded module matches the saved state dict
    assert save_dict['signature'] == str(module.id()), "Module signature does not match saved state dict"

    # Check if the device of the loaded module matches the saved state dict
    assert save_dict['device'] == str(module.device), "Module device does not match saved state dict"

    # Load the state dict to the module
    module.load_state_dict(save_dict['state_dict'])

    # Print a message to indicate that the state dict has been loaded from the file
    print("Loaded state dict from file: {}".format(filename))
