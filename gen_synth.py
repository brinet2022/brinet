import numpy as np
from scipy.sparse import coo_matrix
import os
import argparse
import sys
import  random

def get_simdata(num_patients=100, num_features=100,
                    ind_linked=[   [[0, 0], [4, 2]],   [[6, 2], [9, 0]], 
                                   [[12, 2], [19, 2]],   [[21, 2], [29, 0]], [[35, 0], [45, 2]] ] # causal SNPs
                    , n=2, p=0.3,
                    random_seed=42):
    '''A function to create some simulated non-linear data.  
    [[0, 0], [4, 2]] means that whenever the first value/SNP is 0 and the 4th has 
    value 2 then there is an effect'''
    
    np.random.seed(random_seed)
    basis = np.zeros([num_patients, num_features])
    effectsize = 1

    for k in range(num_features):
        basis[:, k] = np.random.binomial(n, p, num_patients)

    status = np.zeros(num_patients)
    for patient in range(num_patients):
        for linked in ind_linked:
            temp = np.zeros([len(linked)])
            i = 0
            for element in linked:
                if basis[patient, element[0]] == element[1]:
                    temp[i] = 1
                i += 1
            if np.min(temp) > 0:
                status[patient] = 1



    num_diseased = np.sum(status)
    causal_snps = [[x[0][0] for x in ind_linked],[x[1][0] for x in ind_linked]]
    print(("Created dataset[", num_patients, " x ", num_features, "] with", num_diseased, "diseased"))
    return basis, status,causal_snps




def make_mask_gene_layer(inputsize):
    '''We create a simple mask for this network, the first 5 are connected to the 
    first neuron. SNPs 5 to 10 are connected to the second neuron etc. 
    We save it as a sparse matrix.
    The created network is plotted at the end of the notebook'''
    mask_d  = np.zeros((inputsize,10), np.bool)
    mask_d[0:5,0]= True
    mask_d[5:10,1]= True
    mask_d[10:20,2]=True
    mask_d[20:30,3]=True
    mask_d[30:50,4]=True
    mask_d[50:70,5]= True
    mask_d[70:80,6]= True
    mask_d[80:90,7]=True
    mask_d[90:95,8]=True
    mask_d[95:100,9]=True
    mask =  coo_matrix(mask_d)
    gene_end =[0,5,10,20,30,50,70,80,90,95,100]  
    return mask, gene_end


#def mask_to_topology(mask, col_names):
#    coo_m=coo_matrix(m)
#
#    df = pd.DataFrame(list(zip(coo_m.row, coo_m.col)), columns = col_names)
    
def gen_and_save_masks(layers_size, ds_path):
    topology_path=os.path.join(ds_path, "masks.npy")
    labels_path = os.path.join(ds_path, "labels.npy")
    masks = get_mask(layers_size)
    np.save(topology_path, masks, allow_pickle=True)
    layers_size.append(1)
    labels=[]
    for l in range(len(layers_size)):
        labels_layer = ["{}-".format(l)+str(i) for i in range(layers_size[l])]
        labels.append(labels_layer)
    print (labels)
    np.save(labels_path, labels, allow_pickle=True)

def get_mask(layers_size):
    masks= []
    for i in range(len(layers_size)-1):
        if i==0:
            overlap=False
        else:
            overlap=True
        lhs_size = layers_size[i]
        rhs_size = layers_size[i+1]
        mask_l = get_mask_for_layer(lhs_size, rhs_size, overlap)
        masks.append(mask_l)
    return masks

def get_mask_for_layer(lhs_size, rhs_size, overlap=False):
    mask = np.zeros((lhs_size, rhs_size))
    step = lhs_size/rhs_size
    for i in range(rhs_size):
        start = int(i*step)
        end = int((i+1)*step)
        mask[start:end, i]=1
        if overlap:
            print("rand_indices")
            print(lhs_size)
            rand_indice =  np.random.randint(0, high=lhs_size, size=5)
            print(rand_indice)
            mask[rand_indice, i] = 1
    mask=coo_matrix(mask)
    #return coo_matrix( np.ones((lhs_size, rhs_size)))
    return mask


def gen_data(sample_size, input_size, causal_num, confounders_num, p=0.3):
    data = np.zeros([sample_size, input_size])

    n=1 #possible values for features
    #p=0.3  #probablity of positive instances
    for j in range(input_size):
        data[:, j] = np.random.binomial(n, p, sample_size)
        
    causal_snps_values = np.random.choice([1], causal_num)
    causal_snps = np.random.choice(input_size, causal_num)
    zip_iterator = zip(causal_snps, causal_snps_values)
    causal_snps_id_value_map = dict(zip_iterator)
    confounded = np.random.choice(causal_snps, confounders_num, replace=False)
    confounders = 2* data[:,confounded] 
    real_causal_snps = l3 = [x for x in causal_snps if x not in confounded]
    Y = np.zeros(sample_size)
    for i in range(sample_size):
        affected=False
        for c in range(causal_num):
            if data[i,causal_snps[c]]==causal_snps_values[c]:
                affected = True
        if affected:
            Y[i]=1
    return data,Y, real_causal_snps, confounded, causal_snps_id_value_map, confounders


    
def gen_synth_data(dataset_num=1, sample_size=1000, input_size=20, causal_num=2, confounders_num=1, output_path="./", p=0.3):
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    for i in range(dataset_num):
        data,Y, real_causal_snps, confounded, causal_snps_id_value_map, confounders = gen_data(sample_size, input_size, causal_num, confounders_num, p) 
        array = np.array([data,Y, real_causal_snps, confounded, causal_snps_id_value_map, confounders], dtype=object)
        file_path= os.path.join(output_path, "{}.npy".format(i))
        np.save(file_path, array, allow_pickle=True)            

def compute_layers_size(feature_size):
    if feature_size ==10000:
        return [10000, 1000, 100]
    elif feature_size == 20:
        return [20, 4, 2]
    else:
        print ("undefined feature size: compute_layers_size")
        return []

def parse_args():
    parser = argparse.ArgumentParser(sys.argv[1:])
    parser.add_argument("--config_id", type=int, default=0)
    parser.add_argument("--dataset_num", type=int, default=1)
    parser.add_argument('--sample_size', type=int, default=500)
    parser.add_argument('--feature_size', type=int, default=10000)
    parser.add_argument('--causal_num', type=int, default=2)
    parser.add_argument('--confounders_num', type=int, default=1)
    parser.add_argument('--output_root', type=str, default="./synth_gen")
    parser.add_argument('--p', type=float, default=0.3)
    args = parser.parse_args()
    return args

if __name__== "__main__":
    args = parse_args()
    ds_path = os.path.join(args.output_root, str(args.config_id))
    gen_synth_data(dataset_num = args.dataset_num, sample_size=args.sample_size, input_size=args.feature_size, causal_num=args.causal_num, confounders_num=args.confounders_num, output_path=ds_path,p =args.p)
    layers_size = compute_layers_size(args.feature_size) 
    gen_and_save_masks(layers_size, ds_path=ds_path)
    #gen_and_save_masks([10000, 100, 10], output_path=topology_path)


#python gen_synth.py --config_id 0  --dataset_num 1 --sample_size 500 --feature_size 20 --causal_num 2 --confounders_num 1 --output_root "./synth_ds"  --p 0.3 
