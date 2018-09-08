#!/usr/bin/python3
from ghost import hyperparameters

import numpy as np
from hyperopt import fmin, tpe, hp, STATUS_OK
import signal
import sys
import pickle
from hyperopt.mongoexp import MongoTrials


# The key in the space must match a variable name in HyperParameters
space = {
    'learning_rate': hp.uniform('learning_rate', 0.001, 0.05),
    'size_of_hidden_layer': hp.uniform('size_of_hidden_layer', 1, 10),
}

num_trials = 10
trials = MongoTrials('mongo://localhost:1234/hp/jobs', exp_key='img1')


def summarizeTrials():
    print()
    print()
    print("Trials is:", np.sort(np.array([x for x in trials.losses() if x is not None])))

def main():

    def objective(args):
        for key, value in args.items():
            if int(value) == value:
                value = int(value)
            setattr(hyperparameters, key, value)
        score = hyperparameters.runAndGetError()
        #print("Score:", score, "for", args)
        return {'loss': score, 'status': STATUS_OK}

    for i in range(num_trials):
        best = fmin(objective,
                space=space,
                algo=tpe.suggest,
                max_evals=(i+1),
                trials=trials)
    summarizeTrials()
    print(i, "Best result was: ", best)


def signal_handler(signal, frame):
    summarizeTrials()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    main()
