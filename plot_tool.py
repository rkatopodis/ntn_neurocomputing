import argparse
from os.path import abspath, expanduser, exists

from ntn_neurcomputing.experiment_util import plot_experiment

parser = argparse.ArgumentParser()
parser.add_argument("experimentdir", help="Path to the directory with the Ray Tune experiments")
parser.add_argument(
    "--use-eval",
    help="Use data produced in isolated evaluations (not used in training) to generate the plot",
    action="store_true"
)
parser.add_argument("-s", "--resample", type=int)
parser.add_argument("-r", "--radius", type=int)

if __name__ == "__main__":
    args = parser.parse_args()

    experimentdir = abspath(expanduser(args.experimentdir))

    if not exists(experimentdir):
        print("Experiment directory not found")
        exit()
        
    use_eval = args.use_eval
    # import pdb; pdb.set_trace()
    fig = plot_experiment(experimentdir, resample=args.resample, radius=args.radius, use_evaluation=use_eval)
    name = "eval" if use_eval else "train"
    fig.savefig(f"{experimentdir}/learning_curve_{name}.png", dpi=300)