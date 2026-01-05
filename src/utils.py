# src/utils.py
import sys
import os

def get_resource_path(relative_path):
    """
    Resolves the absolute path to an internal resource for both dev and production.

    This function handles the path discrepancy created when bundling a Python app into 
    a single-file executable. In production (PyInstaller), assets are extracted to 
    a temporary folder. In development, assets reside in the project's root directory.

    Args:
        relative_path (str): The path to the asset relative to the project root 
            (e.g., 'models/ball_detection/model.xml').

    Returns:
        str: The absolute filesystem path to the requested resource.
    """
    try:
        # PyInstaller creates a temporary folder and stores the path in _MEIPASS 
        # when the bundled executable is launched.
        base_path = sys._MEIPASS
    except AttributeError:
        # DEVELOPMENT MODE:
        # If _MEIPASS doesn't exist, we are running as a standard script.
        # We calculate the project root by going two levels up from this file 
        # (assuming this utility is located in project_root/src/utils.py).
        base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    return os.path.join(base_path, relative_path)