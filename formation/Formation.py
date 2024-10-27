import numpy as np

def GenerateBasicFormation():

    formation = [
        np.array([-13, 0]),    # Goalkeeper
        np.array([-10, -3]),    # Left Defender (closer to the center)
        np.array([-9, 2]),      # Center Back Left (more central)
        np.array([-7, -1]),     # Center Back Right (more balanced)
        np.array([-4, -2]),      # Right Defender (closer to the center)
        np.array([-2, 2]),      # Left Midfielder (supporting attack)
        np.array([1, -1]),      # Center Midfielder Left (more dynamic)
        np.array([1, 1]),       # Center Midfielder Right (higher up)
        np.array([4, -1]),      # Right Midfielder (attacking width)
        np.array([6, 2]),       # Forward Left (ready for counter)
        np.array([8, 0])      # Forward Right
        ]

    return formation
