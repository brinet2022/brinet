import os
import numpy as np
from itertools import islice

def main():
    root_path = "./visualization/training/"
    vals = np.zeros((6,2,20))
    exp_num=300
    for config_id in range(5,11):
        for method in [0,1]:
            for dataset_index in range(0,20):
                stats_file = os.path.join(root_path,"Experiment_{}".format(exp_num), "stats.csv")
                with open(stats_file) as f:
                    for line in islice(f, 1, 2):
                        auc= float(line.split()[4])
                        vals[config_id-5, method, dataset_index]=auc
                exp_num+=1


    vals_causal = np.zeros((6,2,20))
    exp_num=600
    for config_id in range(11,17):
        for method in [0,1]:
            for dataset_index in range(0,20):
                stats_file = os.path.join(root_path,"Experiment_{}".format(exp_num), "stats.csv")
                with open(stats_file) as f:
                    for line in islice(f, 1, 2):
                        auc= float(line.split()[4])
                        vals_causal[config_id-11, method, dataset_index]=auc
                exp_num+=1

    for config_id in range(0,6):
        for method in [0,1]:
            avg_auc= vals[config_id, method,:].mean()
            print("config_id: {}, method: {}, auc: {}".format(config_id, method, avg_auc))

    for config_id in range(0,6):
        for method in [0,1]:
            avg_auc= vals_causal[config_id, method,:].mean()
            print("config_id: {}, method: {}, auc: {}".format(config_id, method, avg_auc))



main()

