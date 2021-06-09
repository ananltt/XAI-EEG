import copy
from matplotlib import pyplot as plt
from functions_dataset import *


def features_variation(signal, n_segments=8, s=2):

    ca_all, cd_all = pywt.dwt(signal, 'db1')

    indexes = extract_indexes_segments(len(signal), n_segments)
    i = s-1
    start, end = indexes[i]

    signal_zero = copy.deepcopy(signal)
    signal_zero[start:end] = np.zeros(end - start)
    ca_zero, cd = pywt.dwt(signal_zero, 'db1')

    signal_linear = copy.deepcopy(signal)
    signal_linear[start:end] = np.linspace(signal_linear[start], signal_linear[end - 1], num=end - start)
    ca_linear, cd = pywt.dwt(signal_zero, 'db1')

    # Plot results
    fig, axes = plt.subplots(3, 2)
    fig.tight_layout(h_pad=2)

    axes[0, 0].set_title("Signal")
    axes[0, 0].plot(signal)
    axes[0, 1].set_title("Signal Wavelet")
    axes[0, 1].plot(ca_all)

    axes[1, 0].set_title("Zero-Ablation")
    axes[1, 0].plot(signal_zero)
    axes[1, 1].set_title("Zero-Ablation Wavelet")
    axes[1, 1].plot(ca_zero)

    axes[2, 0].set_title("Interpolation-Ablation")
    axes[2, 0].plot(signal_linear)
    axes[2, 1].set_title("Interpolation-Ablation Wavelet")
    axes[2, 1].plot(ca_linear)
    plt.savefig('output/signal-wavelet.png', bbox_inches='tight')
    plt.show()
    # fig.clear()
