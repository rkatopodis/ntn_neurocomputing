import numpy as np

from os.path import abspath, expanduser
from pathlib import Path
from glob import glob
from functools import partial
from itertools import chain

from operator import attrgetter

from tensorflow.python.summary.summary_iterator import summary_iterator

import matplotlib.pyplot as plt

from ray.tune import ExperimentAnalysis

# This next line might have broken things
from ntn_neurocomputing.plot_util import smooth, symmetric_ema

def trial_evaluation_from_logdir(logdir, use_evaluation=True):
    events_file_paths = sorted(glob(abspath(expanduser(logdir)) + "/events*"))
    
    source = "/evaluation" if use_evaluation else ""
    filtered_events = filter(
        lambda s: len(s.summary.value) != 0 and s.summary.value[0].tag == f"ray/tune{source}/episode_reward_mean",
        chain(*(summary_iterator(str(p)) for p in events_file_paths))
    )
    
    step_value_tuples = map(lambda s: (s.step, s.summary.value[0].simple_value), filtered_events)
    
    return zip(*step_value_tuples)


def experiment_evaluation(analysis, use_evaluation=True):
    logdirs = map(attrgetter('logdir'), analysis.trials)
    # import pdb; pdb.set_trace()
    trials_steps = []
    trials_evals = []
    for logdir in logdirs:
        steps, values = trial_evaluation_from_logdir(logdir, use_evaluation)
        trials_steps.append(steps)
        trials_evals.append(values)
        
    return trials_steps, trials_evals  # np.array(trials_evals)


def curve(steps, evals, resample=128, radius=10):
    # y = np.apply_along_axis(smooth, 1, evals, radius=radius)
    # import pdb; pdb.set_trace()
    minsteplen = min(map(len, steps))
    def allequal(qs):
        return all((q==qs[0]).all() for q in qs[1:])
    if resample:
        low  = max(x[0] for x in steps)
        high = min(x[-1] for x in steps)
        usex = np.linspace(low, high, resample)
        ys = []
        for step, row in zip(steps, evals):
            row = smooth(row, radius) if radius is not None else row
            ys.append(symmetric_ema(np.asarray(step), np.asarray(row), low, high, resample, decay_steps=1.0)[1])
    else:
        # assert allequal([x[:minsteplen] for x in steps]),\
        #   'If you want to average unevenly sampled data, set resample=<number of samples you want>'
        usex = np.asarray(steps[0])
        ys = np.asarray(evals)
    ymean = np.mean(ys, axis=0)
    ystd = np.std(ys, axis=0)

    return usex, ymean, ystd

def curve_from_dir(experiment_dir, resample=128, radius=10, use_evaluation=True):
    # import pdb; pdb.set_trace()
    analysis = ExperimentAnalysis(
        sorted(glob(abspath(expanduser(experiment_dir)) + "/experiment_state*.json"))[-1]
    )

    steps, evals = experiment_evaluation(analysis, use_evaluation)

    return curve(steps, evals, resample=resample, radius=radius)


def plot_experiment(experiment_dir, resample=None, radius=1, use_evaluation=True):
    analysis = ExperimentAnalysis(
        sorted(glob(abspath(expanduser(experiment_dir)) + "/experiment_state*.json"))[-1]
    )

    x, y, std = curve(*experiment_evaluation(analysis, use_evaluation), resample=resample, radius=radius)
    fig, ax = plt.subplots()

    ax.set_xlabel("Steps")
    ax.set_ylabel("Cumulative Reward")

    ax.plot(x, y, label="Learning curve")
    ax.fill_between(x, y - std, y + std, alpha=.2)
    
    return fig