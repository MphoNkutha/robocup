import numpy as np

def GenerateBasicFormation():

    formation = [
        np.array([-13, 0]),    # Goalkeeper
        np.array([-10, 8]),  # Left Defender
        np.array([-10, 3]),   # Center Back Left
        np.array([-11, 0]),    # Center Back Right
        np.array([10, -3]),   # Right Defender
        np.array([10, -8]),    # Left Midfielder
        np.array([-5, 6]),    # Center Midfielder Left
        np.array([-5, 2]),     # Center Midfielder Right
        np.array([5, -2]),     # Right Midfielder
        np.array([5, -6]),    # Forward Left
        np.array([-2, 0])      # Forward Right
    ]

    return formation
