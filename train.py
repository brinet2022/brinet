from NNs import BRInet, Baseline
import sys
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
import argparse
from sklearn.utils import class_weight
from utils import readDataset


def main(args):
    epochs = args.epochs
    bs = args.batch_size
    lr = args.learning_rate
    lr_decay = args.learning_rate_decay
    l1_reg = args.l1_reg
    l2_reg = args.l2_reg
    #wpc = args.weight_positive_class
    #wnc = args.weight_negative_class
    experiment_number = args.experiment_number

    ## SET MODEL WEIGHT LOCATION

    folder_dir = args.output_dir + 'Experiment_' + str(experiment_number)   
    if not os.path.exists(folder_dir):
        os.makedirs(folder_dir)  
    
    dataset_path = os.path.join(args.dataset_root, str(args.dataset_config_id), str(args.dataset_index)+".npy")
    x, y, cf_reg = readDataset(dataset_path)
    input_size = x.shape[1]
    print(input_size)
    print("input size: {}".format(input_size))
    topology_path = os.path.join(args.dataset_root, str(args.dataset_config_id), "masks.npy")
    ## PERFORM 5-FOLD CROSS VALIDATION ON TWO SETS: TRAINING SET AND TEST SET

    print (y)
    wnc, wpc = class_weight.compute_class_weight(class_weight="balanced", classes=[0,1], y=y)
    wnc, wpc=1, 1    
    # SETUP TRAINING VARIABLES
    #np.random.seed(42)
    skf = StratifiedKFold(n_splits=5,shuffle=True, random_state = experiment_number)
    fold = 1
    
    exp_info_path = os.path.join(folder_dir,"exp_info.txt")
    with open(exp_info_path, "w") as f:
        f.write("config_id:{}\n".format(args.dataset_config_id))
        f.write("index_id:{}\n".format(args.dataset_index))
        f.write("method: {}\n".format(args.method))
        f.write("input_size: {}\n".format(input_size))
        f.write("l1_reg: {}\n".format(args.l1_reg))
        f.write("l2_reg: {}\n".format(args.l2_reg))
        f.write("noise_radius: {}\n".format(args.noise_radius))
        f.write("grad_threshold: {}\n".format(args.grad_threshold))
        f.write("lr: {}\n".format(args.learning_rate))
        f.write("lr_decay: {}\n".format(lr_decay))
        f.write("batch_sizey: {}".format(bs))
    counter_fold = 0
    for train_idx, test_idx in skf.split(x, y):
        print ("fold : {}".format(fold))
        if counter_fold>0:
            break
        counter_fold+=1
        # TRAINING DATA
        train_data_x = np.array(x[train_idx])
        train_data_y = np.array(y[train_idx])
        train_data_cf_reg = np.array(cf_reg[train_idx])  

        # TEST DATA
        test_data_x = np.array(x[test_idx])
        test_data_y = np.array(y[test_idx])
        
        # BRI-NET TRAINING
        num_cf_reg = train_data_cf_reg.shape[1]
        input_size = train_data_x.shape[1]


        if args.method=="brinet":
            model = BRInet(epochs, bs, lr, lr_decay, l1_reg, l2_reg, num_cf_reg, wpc, wnc, folder_dir, topology_path, input_size, grad_threshold=args.grad_threshold, noise_radius=args.noise_radius)
        elif args.method=="gennet":
            model = Baseline(epochs, bs, lr, lr_decay, l1_reg, l2_reg, wpc, wnc, folder_dir, topology_path, input_size, grad_threshold=args.grad_threshold, noise_radius=args.noise_radius)
        else:
            print("method invalid: {}, {}".format(args.method, type(args.method)))
        model.train(train_data_x, train_data_y, train_data_cf_reg,  test_data_x, test_data_y, fold)
        #print (brinet.test_acc[-1])
        
        fold = fold + 1

def parse_args():
    parser = argparse.ArgumentParser(sys.argv[1:])
    parser.add_argument("--method", type=str, default="brinet")
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--experiment_number', type=int, default=100000)
    parser.add_argument('--dataset_id', type=int, default=0)
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--learning_rate_decay', type=float, default=0.001)
    parser.add_argument('--l1_reg', type=float, default=0.055)
    parser.add_argument('--l2_reg', type=float, default=0.055)
    parser.add_argument('--noise_radius', type=float, default=0.01)
    parser.add_argument('--grad_threshold', type=float, default=0.02)
    parser.add_argument('--weight_positive_class', type=float, default=1)
    parser.add_argument('--weight_negative_class', type=float, default=1)
    parser.add_argument('--output_dir', type=str, default="./Training/")
    parser.add_argument('--dataset_config_id', type=int, default=0)
    parser.add_argument('--dataset_index', type=int, default=0)
    parser.add_argument('--dataset_root', type=str, default="./synth_ds/")
    args = parser.parse_args()
    return args
    
if __name__ == "__main__":
    args = parse_args()
    main(args)

#python train.py --epochs 5000 --batch_size 64 --experiment_number 1004  --dataset_config_id 10  --dataset_index 0 --l1_reg 0.01 --l2_reg 0.01 
