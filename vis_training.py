from numpy import genfromtxt
import os
import csv
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import sys

def readData_fold(exp_path, fold, error_type="train"):
    #meta_path = os.path.join(exp_path, "metrics_names.txt")
    #f=open(meta_path, "r")
    #r=csv.reader(f)
    #columns = next(r)
    columns = ["epoch","loss","accuracy","auc","precision","recall","TP","TN","FP","FN"]
    file_name = "log_{}_fold_{}.txt".format(error_type, fold)
    data_path = os.path.join(exp_path, file_name)
    data = genfromtxt(data_path, delimiter=',') 
    df = pd.DataFrame(data, columns=columns)
    return df
        

def readData_by_path(exp_path, fold_num, error_type):
    dfs = []
    for i in range(1, fold_num+1):
        df = readData_fold(exp_path, i, error_type)
        dfs.append(df)
    return dfs



def readData(exp_num, fold_num, error_type, root_path="./Training/"):
    results_path = os.path.join(root_path, "Experiment_{}".format(exp_num))
    dfs = readData_by_path(results_path, fold_num, error_type)
    return dfs



def drawTrainingMeasures(metric, df_train , df_test, output_file):

    max_val = max(df_train[metric].max(), df_test[metric].max())

    fig, ax = plt.subplots(2,1 , squeeze=False)
    #fig.set_size_inches(25, 10)
    
    ax[0,0].plot(df_train['epoch'], df_train[metric],  c='green', label = 'training loss')
    ax[0,0].set_title('{} vs epoch'.format(metric))
    ax[0,0].axis(ymin = 0, ymax = max_val)

    ax[1,0].plot(df_test['epoch'], df_test[metric],  c='red', label = 'test loss')
    ax[1,0].axis(ymin = 0, ymax = max_val)
    #ax[0,0].legend(fontsize=10)
    fig.savefig(output_file)

def draw(metric, exp_num,max_fold, root_path):
    dfs_train=readData(exp_num, max_fold, "train", root_path="./Training/")
    dfs_test=readData(exp_num, max_fold, "test", root_path="./Training/")
    output_folder = os.path.join(root_path, "Experiment_{}".format(exp_num))
    if not os.path.exists(output_folder):
            os.makedirs(output_folder)
    for i in range(max_fold):
        output_file = os.path.join(output_folder, "{}_fold_{}.png".format(metric, i))
        drawTrainingMeasures(metric, dfs_train[i] , dfs_test[i], output_file)

def compute_f1(row):
    precision = row["precision"]
    recall = row["recall"]
    f1 = (2*precision*recall)/(precision+recall+1e-10)
    return f1

def computeStatsFold(df_train, df_test):
    df_train["f1"] = df_train.apply (lambda row: compute_f1(row), axis=1)
    df_test["f1"] = df_test.apply (lambda row: compute_f1(row), axis=1)
    idxmin = df_test["auc"].idxmax()
    values = df_test.iloc[idxmin]
    return values[1:]    


def computeStats(dfs_train, dfs_test):
    folds_num = len(dfs_train)
    max_vals = []
    for i in range(folds_num):
        #print ("fold: {}".format(i))
        vals = computeStatsFold(dfs_train[i], dfs_test[i])
        max_vals.append(vals)
    df = pd.concat(max_vals, axis=1).T
    df.reset_index(inplace=True)
    #print ("before computing id")
    #print (df)
    fold_id = df["loss"].idxmin()
    values = df.loc[fold_id]
    return [df.mean(axis=0), values, fold_id, df.copy()]


def computeStatsMain(exp_num, max_fold, root_path):
    dfs_train=readData(exp_num, max_fold, "train", root_path="./Training")
    dfs_test=readData(exp_num, max_fold, "test", root_path="./Training")
    output_folder = os.path.join(root_path, "Experiment_{}".format(exp_num))
    if not os.path.exists(output_folder):
            os.makedirs(output_folder)
    df_mean, df_max, fold_id, df_detailed = computeStats(dfs_train, dfs_test)
    df = pd.concat([df_mean, df_max], axis=1).reset_index()
    df.rename({0:"mean",82673:"best"}, inplace=True)
    df=df.T
    col_rename_dict = {i:j for i,j in zip(df.columns, df.iloc[0])}
    df.rename(columns=col_rename_dict, inplace=True)
    df = df.iloc[1:].set_index(pd.Index(["mean","best"]))
    print (df)
    print ("fold id: {}".format(fold_id))
    output_file = os.path.join(output_folder, "stats.csv")
    output_file_detailed = os.path.join(output_folder, "stats_detailed.csv")
    f=open(output_file,"w")
    f.write(df.__repr__())
    f.write("\n")
    f.write("fold id: {}\n".format(fold_id))
    #df.to_csv(output_file)
    f.close()
    f=open(output_file_detailed,"w")
    f.write(df_detailed.__repr__())
    f.close()


#exp_num_list = [2,3]
def main(args):
    metrics = ["loss", "auc", "precision", "recall"]
    #if args.exp_num==-1:
    #    for exp_num in exp_num_list:
    #        for metric in metrics:
    #            draw(metric, exp_num, args.max_fold, args.root_path)
    #    return
    if not args.stats_only:
        for metric in metrics:
            draw(metric, args.exp_num, args.max_fold, args.root_path)
    computeStatsMain(args.exp_num, args.max_fold, args.root_path)
    

def parse_args():
    parser = argparse.ArgumentParser(sys.argv[1:])
    parser.add_argument("--stats_only", action='store_true')
    parser.add_argument("--exp_num", type=int)
    parser.add_argument('--max_fold', type=int, default=1)
    parser.add_argument('--root_path', type=str, default="./visualization/training")
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = parse_args()
    main(args)

        

#python vis_training.py --exp_num 21 --max_fold 4 --stats_only
    
