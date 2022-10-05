import os
import numpy as np

def main():
    root_path = "./explanations/"
    brinet_vals = []
    for i in range(200, 220):
        stats_file = os.path.join(root_path, str(i), "stats.txt")
        vals = np.genfromtxt(stats_file, dtype=float)    
        print (vals)
        print (type(vals))
        max_val = np.max(vals)
        brinet_vals.append(max_val)
    print (np.mean(brinet_vals))


    baseline_vals = []
    for i in range(220, 240):
        stats_file = os.path.join(root_path, str(i), "stats.txt")
        vals = np.genfromtxt(stats_file, dtype=float)
        max_val = np.max(vals)
        baseline_vals.append(max_val)

    print (np.mean(baseline_vals))
    

main()

