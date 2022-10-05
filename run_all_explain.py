import numpy as np
from utils import readDataset
from NNs import BRInet, Baseline
import argparse
import sys, os
from utils import create_if_not_exist
import vis_graph
from explain import main as explain
import datetime


class argsP():
    def __init__(self, method, exp_num, fold_id, config_id, dataset_index):
        self.method= method
        self.experiment_number= exp_num
        self.fold_id= fold_id
        self.training_root = "./Training/"
        self.input_size = 8641
        self.output_root = "./explanations/"
        self.dataset_config_id = config_id
        self.dataset_index= dataset_index
        self.dataset_root= "./synth_ds/"


def main():
    exp_num=18000
    for config_id in [7000]:
        for method in ["brinet","gennet"]:
            for dataset_index in range(0,100):
                print ("exp: {}".format(exp_num))
                print (datetime.datetime.now())
                args = argsP(method, exp_num, 1, config_id, 0)
                explain(args)
                exp_num+=1
main()

