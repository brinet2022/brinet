import numpy as np
import os


def create_if_not_exist(folder_dir):
    if not os.path.exists(folder_dir):
        os.makedirs(folder_dir)

class Logger():
    def __init__(self, file_path="log.txt"):
        self.file_path= file_path

    def log(self, msg):
        file_handler = open(self.file_path, "a")
        file_handler.write(msg)
        file_handler.write("\n")
        file_handler.close()


def readDataset(ds_path):
    data,Y, real_causal_snps, confounded, causal_snps_id_value_map, confounders, confounders_input = np.load(ds_path, allow_pickle=True)
    return data, Y, confounders

