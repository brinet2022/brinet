import numpy as np
from utils import readDataset
from NNs import BRInet, Baseline
import argparse
import sys, os
from utils import create_if_not_exist
import vis_graph
from explain import main as explain

class argsP():
    def __init__(self, baseline, exp_num, fold_id, config_id, dataset_index):
        self.baseline=baseline
        self.experiment_number= exp_num
        self.fold_id= fold_id
        self.training_root = "./Training/"
        self.input_size = 10000
        self.output_root = "./explanations/"
        self.dataset_config_id = config_id
        self.dataset_index= dataset_index
        self.dataset_root= "./synth_ds/"


def main():
    exp_num=600
    for config_id in range(11,17):
        for method in [0,1]:
            for dataset_index in range(0,20):
                args = argsP(method, exp_num, 6, config_id, dataset_index)
                explain(args)
                exp_num+=1

main()

