from os.path import abspath, dirname
from pathlib import Path

import matplotlib.pyplot as plt

from ntn_neurocomputing.experiment_util import curve_from_dir

def comparison(first_dir, second_dir, title, xlabel, ylabel, first_label, second_label, resample=128, radius=10):
    first_x,   first_y,   first_std   = curve_from_dir(first_dir, resample, radius, use_evaluation=False)
    second_x, second_y, second_std = curve_from_dir(second_dir, resample, radius, use_evaluation=False)

    fig, ax = plt.subplots()

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    
    ax.plot(first_x, first_y, label=first_label)
    ax.fill_between(first_x, first_y - first_std, first_y + first_std, alpha=.2)

    ax.plot(second_x, second_y, label=second_label)
    ax.fill_between(second_x, second_y - second_std, second_y + second_std, alpha=.2)
    ax.legend(loc='lower right')

    return fig


def generate_encoding_comparison(thermometer_dir, circular_dir, title, resample=False, radius=1):
    return comparison(
      thermometer_dir,
      circular_dir,
      title,
      "Steps",
      "Cumulative Reward",
      "Thermometer Encoding",
      "Circular Encoding",
      resample,
      radius
    )

def generate_arch_comparison(wnn_dir, mlp_dir, title, resample=False, radius=1):
    return comparison(
      wnn_dir,
      mlp_dir,
      title,
      "Steps",
      "Cumulative Reward",
      "n-tuple network",
      "MLP",
      resample,
      radius
    )

if __name__ == "__main__":
    filedir = abspath(dirname(__file__))
    plotdir = filedir + "/plots"
    enc_dir = plotdir + "/encodings"
    com_dir = plotdir + "/comparison"


    # Ensure the plots directories exists
    Path(enc_dir).mkdir(parents=True, exist_ok=True)
    Path(com_dir).mkdir(parents=True, exist_ok=True)

    # Encoding comparisons
    fig = generate_encoding_comparison(
        f"{filedir}/wnn/cartpole/thermo/individual_run/experiment",
        f"{filedir}/wnn/cartpole/circular/individual_run/experiment",
        title="Cartpole"
    )
    fig.savefig(f"{enc_dir}/cartpole.png", dpi=600)

    fig = generate_encoding_comparison(
        f"{filedir}/wnn/lunar/thermo/individual_run/experiment",
        f"{filedir}/wnn/lunar/circular/individual_run/experiment",
        title="Lunar Lander"
    )
    fig.savefig(f"{enc_dir}/lunar.png", dpi=600)

    fig = generate_encoding_comparison(
        f"{filedir}/wnn/reacher/thermo/individual_run/experiment",
        f"{filedir}/wnn/reacher/circular/individual_run/experiment",
        title="Reacher"
    )
    fig.savefig(f"{enc_dir}/reacher.png", dpi=600)

    fig = generate_encoding_comparison(
        f"{filedir}/wnn/hopper/thermo/individual_run/experiment",
        f"{filedir}/wnn/hopper/circular/individual_run/experiment",
        title="Hopper",
        resample=128
    )
    fig.savefig(f"{enc_dir}/hopper.png", dpi=600)

    #
    # Architecture comparisons
    #
    fig = generate_arch_comparison(
        f"{filedir}/wnn/cartpole/thermo/individual_run/cartpole-wnn-experiment",
        f"{filedir}/ann/cartpole/individual_run/cartpole-experiment",
        title="Cartpole"
    )
    fig.savefig(f"{com_dir}/cartpole.png", dpi=600)

    fig = generate_arch_comparison(
        f"{filedir}/wnn/lunar/thermo/individual_run/experiment",
        f"{filedir}/ann/lunar/individual_run/experiment",
        title="Lunar Lander"
    )
    fig.savefig(f"{com_dir}/lunar.png", dpi=600)

    fig = generate_arch_comparison(
        f"{filedir}/wnn/reacher/circular/individual_run/experiment",
        f"{filedir}/ann/reacher/individual_run/experiment",
        title="Reacher"
    )
    fig.savefig(f"{com_dir}/reacher.png", dpi=600)

    fig = generate_arch_comparison(
        f"{filedir}/wnn/hopper/thermo/individual_run/experiment",
        f"{filedir}/ann/hopper/individual_run/experiment",
        title="Hopper",
        resample=128
    )
    fig.savefig(f"{com_dir}/hopper.png", dpi=600)

