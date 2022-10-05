import numpy as np
from utils import readDataset
from NNs import BRInet, Baseline
import argparse
import sys, os
from scipy.sparse import coo_matrix
from utils import create_if_not_exist
import vis_graph
import importlib
import networkx as nx
importlib.reload(vis_graph)
from tqdm import tqdm
from sklearn import metrics
from sklearn.model_selection import StratifiedKFold
import stat_test as st
import importlib
importlib.reload(st)
import evaluate_priors as ep
#def get_shap_values(x, y, model):
#    # GET SHAP VALUES
#    background = x[np.random.choice(x.shape[0], 20, replace=False)]
#    e = shap.DeepExplainer(model, background)
#    explainer = shap.KernelExplainer(model, x, link='logit')
#    shap_values = e.shap_values(x_test[0:10])
#    return shap_values

def main(args):
    epochs = 100
    bs = 64
    lr = 0.00001
    lr_decay = 0.0001
    l1_reg = 0.
    l2_reg = 0.001
    wpc = 1
    wnc = 1
    experiment_number = args.experiment_number
    topology_path = os.path.join(args.dataset_root, str(args.dataset_config_id), "masks.npy")
    labels_path = os.path.join(args.dataset_root, str(args.dataset_config_id), "labels.npy")
    dataset_path = os.path.join(args.dataset_root, str(args.dataset_config_id), str(args.dataset_index)+".npy")
    output_dir = os.path.join(args.output_root, str(args.experiment_number))
    stats_file_path = os.path.join(output_dir, "weights.txt")
    confirmed_causal_path = os.path.join(output_dir, "confirmed_causal_indices.npy")
    top_prior_path = os.path.join(output_dir, "top_prior_indices.npy")
    create_if_not_exist(output_dir)
    if args.method =="brinet":
        model = BRInet(epochs, bs, lr, lr_decay, l1_reg, l2_reg, 1,  wpc, wnc, "", topology_path, args.input_size, grad_threshold=0, noise_radius=0) #num_cf_reg TODO
    elif args.method =="gennet":
        model = Baseline(epochs, bs, lr, lr_decay, l1_reg, l2_reg,  wpc, wnc, "", topology_path, args.input_size, grad_threshold=0, noise_radius=0)
    classifier = model.workflow

    m = np.load(topology_path, allow_pickle=True)
    labels = np.load(labels_path, allow_pickle=True)
    labels= add_prefix_labels(labels)
    data,Y, real_causal_snps, confounded, causal_snps_id_value_map, confounders, confounders_input = np.load(dataset_path, allow_pickle=True)
    TPMT_snps = ep.getSNPs("data/domain/TPMT.txt")
    TLR4_snps = ep.getSNPs("data/domain/TLR4.txt")
    prior_snps = TPMT_snps + TLR4_snps
    top_snps = []
    with open(stats_file_path, "w") as f:
        for fold_id in range(1, args.fold_id+1):
            model_path = args.training_root+'Experiment_' + str(experiment_number) + '/best_model_fold_' + str(fold_id) + '_.h5'
            subnetwork_graph_path = os.path.join(output_dir, str(fold_id)+"_subnetwork"+".png")
            network_graph_path = os.path.join(output_dir, str(fold_id)+"_network"+".png")
            classifier.load_weights(model_path)
            G, w = build_graph_and_compute_scores(classifier, m, labels, causal=real_causal_snps, confounded=confounded, min_ratio=0.5)
            causal_nodes, causal_scores, pert_scores, l1_nodes, scores = extract_causal_snp_indices(G, m, labels, classifier, data, Y, model.weighted_binary_crossentropy, 0.95)
            top_priors = list(set(causal_nodes)&set(prior_snps))
            pvals, ss, vals = st.collectStats([0.01, 0.05, 0.1, 0.2, 0.5 ], causal_nodes, data, Y, confounders_input )
            confirmed_causal_indices = [[causal_nodes[i] for i in np.where(pvals[j]<0.05)[0]] for j in range(len(pvals)) ]
            np.save(confirmed_causal_path, confirmed_causal_indices, allow_pickle=True)
            np.save(top_prior_path, top_priors, allow_pickle=True)
            #model_causal=set(causal_nodes) 
            #dice = (2*(len(model_causal & set(real_causal_snps))))/(len(model_causal) + len(real_causal_snps))
            #f.write("{:.4f},{:.4f}\n".format(cause_score, conf_score))
            #f.write("{:.2f}".format(dice))
            #f.write("\n")

def get_gene_scores(masks, data, Y, model):
    skf = StratifiedKFold(n_splits=5,shuffle=True, random_state = 42)
    train_idx, test_idx = next(skf.split(data, Y))
    data = data[test_idx]
    Y = Y[test_idx]

    mask = masks[0].todense()
    snsp_num, genes_num = mask.shape
    mapping = {}

    for i in range(genes_num):
        snp_indices = np.where(mask[:,i]==1)[0]
        mapping[i] = list(range(i*100, (i+1)*100)) #snp_indices

    data_perturbed = data.copy()
    base_score =  computeScore_AUC(model, data, Y)
    pert_scores = []
    #for gene_indice in tqdm(range(genes_num)):
    for gene_indice in tqdm(range(84)):
        for ind in mapping[gene_indice]:
            data_perturbed[:, ind] = np.random.permutation(data_perturbed[:, ind])
        new_score = computeScore_AUC(model, data_perturbed, Y)
        pert_scores.append(max(base_score - new_score,0 ))
        print (new_score)
        data_perturbed = data.copy()
    return pert_scores

def compute_rank_string(scores, l1_nodes, real_causal):
        scores=scores*-1
        temp = scores.argsort()
        rank = temp.argsort()
        res = []
        for cause_id in real_causal:
            cause_node = "0-{}".format(cause_id)
            cause_index = l1_nodes.index(cause_node)
            cause_rank = rank[cause_index]
            res.append((cause_id, cause_rank))
        str_pairs= ["{}:{}".format(pair[0], pair[1]) for pair in res]
        out=",".join(str_pairs)
        return out

def computeScore(model, X, Y, loss_func):
    y_pred = model.predict(X)
    loss_val = loss_func(Y, y_pred)
    return loss_val

def computeScore_AUC(model, X, Y):
    y_pred = model.predict(X)
    auc = metrics.roc_auc_score(Y, y_pred)
    return auc

def filter_causal_nodes(causal_nodes, model, data, Y, loss_func):
    causal_nodes = np.array(causal_nodes)
    data_perturbed = data.copy()
    base_score =  computeScore(model, data, Y, loss_func)
    pert_scores = []
    for ind in causal_nodes:
        data_perturbed[:, ind] = np.random.permutation(data_perturbed[:, ind])
        new_score = computeScore(model, data_perturbed, Y, loss_func)
        pert_scores.append(new_score - base_score)
        data_perturbed = data.copy()
    pert_scores = np.array(pert_scores)
    pert_scores = np.abs(pert_scores)
    threshold = np.std(pert_scores) + np.mean(pert_scores)
    filtered_causal_nodes=  causal_nodes[np.where(pert_scores>threshold)]
    filtered_causal_pert_scores=  pert_scores[np.where(pert_scores>threshold)]
    return filtered_causal_nodes, filtered_causal_pert_scores, pert_scores

def add_prefix_labels(labels):
    new_labels = []
    for index, layer_labels in enumerate(labels):
        new_layer_labels = [ "{}_{}".format(index, l) for l in layer_labels]
        new_labels.append(new_layer_labels)
    return new_labels

def extract_causal_snp_indices(G, masks, labels, model, data, Y, loss_func, threshold=0.95):
    l1_nodes = np.array(vis_graph.get_active_nodes_layer_L(masks, labels, 0))
    scores = np.array([G.nodes[node]["score"] for node in l1_nodes])
    ind = np.argpartition(scores, -100)[-100:]
    ind = ind[np.argsort(scores[ind])]
    #slice_point = int(threshold*len(labels))
    #causal_nodes = ind[-slice_point:]
    causal_nodes = ind
    print ("len of causal nodes:{}".format(len(ind)))
    #causal_nodes, causal_pert_scores, pert_scores = filter_causal_nodes(causal_nodes, model, data, Y, loss_func)
    pert_scores=None
    causal_scores = [scores[i] for i in causal_nodes]
    return causal_nodes, causal_scores, pert_scores, l1_nodes, scores

def build_graph_and_compute_scores(classifier, masks, labels, causal=[], confounded=[], min_ratio=0.5):
    w0 = abs(classifier.layers[2].get_weights()[0])
    w1 = abs(classifier.layers[5].get_weights()[0])
    w2 = abs(classifier.layers[9].get_weights()[0])
    output_mask = np.ones((masks[-1].shape[-1],1))
    masks = [masks[0].todense(), masks[1].todense(), output_mask]
    w=[convert_1dWeight_to_2dweight(masks[0], w0), convert_1dWeight_to_2dweight(masks[1], w1),convert_1dWeight_to_2dweight(masks[2], w2)]
    new_masks = []
    print ("converting mask to graph")
    G = vis_graph.mask_to_graph(masks, labels, w, causal, confounded)
    l1_nodes = vis_graph.get_active_nodes_layer_L(masks, labels, 0)
    print ("computing score for layer 1")
    for node in tqdm(l1_nodes):
        paths= nx.all_simple_paths(G, source=node, target=labels[3][0])
        score_sum=0
        path_counter=0
        scores= []
        for path in paths:
            path_counter+=1
            score=1    
            for i in range(len(path)-1):
                edge_weight= nx.Graph.get_edge_data(G, u=path[i], v=path[i+1])["weight"]
                score= score * edge_weight
            score_sum=score_sum+ np.abs(score)
            scores.append(score)
        nx.set_node_attributes(G, {node:{"score":score_sum/path_counter}})
        #nx.set_node_attributes(G, {node:{"score": np.max(scores)}})
    print ("computing scores for the rest of layers...")
    set_node_scores(G, masks, labels, 1)
    set_node_scores(G, masks, labels, 2)
    set_node_scores(G, masks, labels, 3)

    return G, w 


def set_node_scores(G, masks, labels, layer_index):
    l2_nodes = vis_graph.get_active_nodes_layer_L(masks, labels, layer_index)
    for ii, node in enumerate(l2_nodes):
        sum_score=0
        for u, v, data in G.in_edges(node, data=True):
            sum_score += G.nodes[u]["score"]
        nx.set_node_attributes(G, {node:{"score":sum_score}})




def extract_subnetwork(classifier, masks, min_ratio=0.5):
    w0 = abs(classifier.layers[2].get_weights()[0])
    w1 = abs(classifier.layers[5].get_weights()[0])
    w2 = abs(classifier.layers[9].get_weights()[0])
    output_mask = np.ones((masks[-1].shape[-1],1))
    masks = [masks[0].todense(), masks[1].todense(), output_mask]
    w=[convert_1dWeight_to_2dweight(masks[0], w0), convert_1dWeight_to_2dweight(masks[1], w1),convert_1dWeight_to_2dweight(masks[2], w2)]
    new_masks = []
    for i in range(len(masks)-1,-1,-1):
        #print (masks[i])
        max_weight = w[i].max()
        mask = masks[i]
        new_mask = np.zeros(mask.shape)
        lhs_size, rhs_size = mask.shape
        if i== (len(masks)-1):
            rhs_indices = [0]
        else:
            rhs_indices = np.where(previous_mask.sum(axis=1)>0)[0]
        #print ("rhs indices:")
        #print (rhs_indices)
        for rhs in rhs_indices:
            lhs_indices = np.where(mask[:,rhs]==1)[0]
            for lhs in lhs_indices:
                #edge_weight = extract_weight(mask, w[i], lhs, rhs)
                edge_weight = w[i][lhs, rhs]
                #print ("edge weight:")
                #print (edge_weight)
                if edge_weight/max_weight> min_ratio:
                    #print ("edge {}-{} removed".format(lhs,rhs))
                    new_mask[lhs, rhs]=1
        previous_mask = new_mask
        #print ("final mask")
        #print (new_mask)
        new_masks.append(new_mask)
    return [ list(reversed(new_masks)), w]



def extract_weight(mask, weights, lhs, rhs):
    i,j=np.where(mask==1)
    c=coo_matrix((weights.squeeze(), (i,j)), shape=mask.shape)
    c=c.tocsr()
    return c[lhs, rhs]

def convert_1dWeight_to_2dweight(mask, weights):
    i,j=np.where(mask==1)
    c=coo_matrix((weights.squeeze(), (i,j)), shape=mask.shape)
    c=c.tocsr()
    return c


def parse_args():
    parser = argparse.ArgumentParser(sys.argv[1:])
    parser.add_argument("--method", type=str , default="brinet")
    parser.add_argument('--experiment_number', type=int, default=100000)
    parser.add_argument('--fold_id', type=int, default=1)
    parser.add_argument('--training_root', type=str, default="./Training/")
    parser.add_argument('--input_size', type=int, default=20)
    parser.add_argument('--output_root', type=str, default="./explanations/")
    parser.add_argument('--dataset_config_id', type=int, default=0)
    parser.add_argument('--dataset_index', type=int, default=0)
    parser.add_argument('--dataset_root', type=str, default="./synth_ds/")
    args = parser.parse_args()
    return args

if __name__=="__main__":
    args = parse_args()
    main(args)

# python explain.py  --experiment_number 26 --fold_id 0  --input_size 20 --dataset_config_id 0  --dataset_index 0

