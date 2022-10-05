import numpy as np
from collections import defaultdict
import stat_test as st

data,Y, real_causal_snps, confounded, causal_snps_id_value_map, confounders, confounders_input = np.load("synth_ds/6000/0.npy", allow_pickle=True)
labels = np.load("synth_ds/6000/labels.npy", allow_pickle=True)


def create_map(start, end):
    res = defaultdict(int)
    for i in range(start, end):
        causes_list = np.load("explanations/{}/confirmed_causal_indices.npy".format(i), allow_pickle=True) 
        for row in causes_list:
            for cause in row:
                res[cause]+=1
    return res

def main():
    brinet_map = create_map(18000, 18100)
    gennet_map = create_map(18100, 18200)
    
    total_map = defaultdict(int)
    total_keys = list(brinet_map.keys()) + list(gennet_map.keys())
    for key in total_keys:
        s = 0
        if key in gennet_map:
            s += gennet_map[key]
        if key in brinet_map:
            s += brinet_map[key]

        total_map [key] = s
    
    total_list = [(key, total_map[key]) for key in total_map]
    total_list = sorted(total_list , key = lambda x: x[1]) 
    
    return total_list, total_map


def find_top_pvals(node_ids):
    res = []
    for node_id in node_ids:
        pvals, ss, vals = st.collectStats([0.01, 0.05, 0.1, 0.2, 0.5 ], [node_id], data, Y, confounders_input)
        min_id = np.argmin(pvals)
        if vals[min_id][0]>0:
            res.append((node_id, np.min(pvals)))
    res = sorted(res, key=lambda x: x[1])
    return res

def extract_info(min_pvals):
    rows = []
    total_list, total_map = main()
    for pair in min_pvals:
        pvals, ss, vals = st.collectStats([0.01, 0.05, 0.1, 0.01], [pair[0]], data, Y, confounders_input )
        min_id = np.argmin(pvals)
        row = [pair[0], total_map[pair[0]], pair[1], ss[min_id][0], vals[min_id][0]]
        rows.append(row)
    return rows

