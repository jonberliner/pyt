import matplotlib.pyplot as plt
import numpy as np


def discrete_cmap(N, base_cmap='Set2'):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    cmap = base(np.linspace(0, 1, N))
    return cmap

