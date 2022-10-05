import vis_training
import os
import numpy as np

def main():
    experiment_counter=6000

def collectStats(exp_ids):
    res = []
    for i in exp_ids[:9]:
        print (i)
        exp_id_string = "{}".format(i)
        input_dir = os.path.join("explanations/{}".format(exp_id_string))
        if not os.path.exists(input_dir):
            print ("skipped")
            continue
        dice_path = os.path.join(input_dir, "weights.txt")
        with open(dice_path, "r") as f:
            first_line = f.readline()
            dice = float(first_line.strip())
        res.append(dice)
    return res 

def main():
    exp_ids = [(7000, 7019, "config 5, brinet"),(7040, 7059, "config 6, brinet"), (7080,7099, "config 7, brinet"), (7120,7139, "config 8, brinet"), (7160,7179, "config 9, brinet"), (7200, 7219, "config 10, brinet"), (7240, 7259, "config 11: brinet"), (7280, 7299, "config 12: brinet"), (7320, 7339, "config 13: brinet"), (7360, 7379, "config 14: brinet"), (7400, 7419, "config 15: brinet"), (7020,7039, "config 5, gennet"), (7060, 7079, "config 6, gennet" ), ( 7100, 7119, "config 7, gennet"), (7140, 7159, "config 8, gennet"), (7180, 7199, "config 9, gennet"), (7220, 7239, "config 10, gennet"), (7260, 7279, "config 11, gennet"), (7300, 7319, "config 12, gennet"), (7340, 7359, "config 13, gennet" ), (7380, 7399, "config 14, gennet" ), (7420, 7439, "config 15, gennet" )]

    for p in exp_ids[-1:]:
        print ("evaluting : {}".format(p[2]))
        res = collectStats(list(range(p[0], p[1]+1)))
        #print (res)
        print ("{}:    {}".format(p[2], np.mean(res)))

main()
