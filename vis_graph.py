import networkx as nx
from networkx.drawing.nx_agraph import write_dot, graphviz_layout
import matplotlib.pyplot as plt
from pyvis.network import Network
import numpy as np
from tqdm import tqdm


def node_to_str(layer_id,node_idx):
    if layer_id==0:
        return "entrez ID\n#"+str(Mask1.index[node_idx])
    elif layer_id==1:
        return str(Mask2.index[node_idx])
    elif layer_id==2:
        return str(Mask3.index[node_idx])
    else:
        return str(node_idx)
    

def createGraph():
    a=[]
    l1=[]
    l1.append(("0","3",1))
    l1.append(("0","4",4))
    l1.append(("1","4",4))
    a.append(l1)
    l2=[]
    l2.append(("4","5",4))
    l2.append(("3","5",4))
    a.append(l2)
    return a
    

def create_color_seq(G):
    
    nodes = list(G.nodes)
    #print (nodes)
    seq  = ["black"]*len(nodes)
    for i in range(len(nodes)):
        node_info = G.nodes[nodes[i]]
        if "causal" in  node_info and node_info["causal"]==True:
            print ("changed to gree")
            seq[i] = "green"
        if "confounded" in  node_info and node_info["confounded"]==True:
            seq[i] = "blue"            
            print ("changed to blue")
    return seq

def drawGraph(G, title,  output_file="graph.png"):
    fig= plt.figure(figsize=(20,10))
    plt.title(title)
    pos =graphviz_layout(G, prog="dot",args="-Grankdir=LR")

    edges = G.edges()
    scale_constant=1
    weights = [ abs(G[u][v]['weight'])*scale_constant+0.1 for u,v in edges]

    color_seq = create_color_seq(G)
    nx.draw(G, pos, width=weights, arrows=True, node_color = color_seq)
    
    keys = pos.keys()
    for key in keys:
        x, y = pos[key]
        if x >= 100:
            plt.text(x+10,y+10,s=key)
        else:
            plt.text(x-30,y,s=key)
    print ("Graph saved at: {}".format(output_file))

    plt.savefig(output_file)
    return 

def get_active_nodes_layer_L(masks, node_labels, layer_index):
    if layer_index==3:
        return node_labels[3]

    mask = masks[layer_index]
    labels = node_labels[layer_index]
    indices = np.where(mask.sum(axis=1)>0)
    #print (indices[0])
    active_nodes = [labels[i] for i in indices[0]]
    return active_nodes



def createSampleMask():
    mask1  = np.random.randint(1,2,(20,4))
    mask2  = np.random.randint(1,2,(4,2))
    mask3  = np.random.randint(1,2,(2,1))
    weights1 = np.random.rand(20,4)
    weights2 = np.random.rand(4,2)
    weights3 = np.random.rand(2,1)
    masks= [mask1, mask2, mask3]
    weights = [weights1, weights2, weights3]
    labels1 = ["0-"+str(i) for i in range(20)]
    labels2 = ["1-"+str(i) for i in range(4)]
    labels3 = ["2-"+str(i) for i in range(2)]
    labels4= ["3-0"]
    labels = [labels1, labels2, labels3, labels4]
    causal= [3]
    confounded = [4]
    return [masks, labels, weights, causal, confounded]

def mask_to_graph(masks, node_labels, weights_layers, causal, confounded, output_file=""):
    G = nx.DiGraph()
    active_nodes_l1 = get_active_nodes_layer_L(masks, node_labels, 0)

    #print (active_nodes_l1)
    #print (node_labels)
    for node in active_nodes_l1:
        if node_labels[0].index(node) in causal:
            G.add_node(node, causal=True )
            #print ("{} added causal".format(node))
        elif node_labels[0].index(node) in confounded:
            G.add_node(node, confounded=True )
            #print ("{} added confounded".format(node))
        else:
            G.add_node(node)
    print ("#masks = {}".format(len(masks)))
    for i in range(len(masks)):
        print ("adding edges for mask {}".format(i))
        mask =  masks[i]
        labels_l = node_labels[i]
        labels_r = node_labels[i+1]
        weights= weights_layers[i]
        lhs_size, rhs_size = mask.shape
        for l in tqdm(range(lhs_size)):
            for r in range(rhs_size):
                if mask[l,r]==1:
                    G.add_edge(labels_l[l], labels_r[r], weight=weights[l,r])
    G=remove_unreachable_nodes(G, active_nodes_l1)
    return G


def remove_unreachable_nodes(G, input_nodes=None):
    reachable_nodes = set()
    for node in input_nodes:
        reachable_paths= nx.single_source_shortest_path(G, node)
        reachable_nodes=reachable_nodes.union(set(reachable_paths.keys()))
    all_nodes = set(G.nodes())
    to_be_removed = all_nodes - reachable_nodes
    for node in to_be_removed:
        G.remove_node(node)
    return G

def draw_from_mask(masks, node_labels, weights_layers, causal=[], confounded=[], output_file = "graph.png"):
    G = mask_to_graph(masks, node_labels, weights_layers, causal, confounded, output_file)
    #print (G.nodes)
    drawGraph(G, "", output_file = output_file) 
    return G

def draw_from_graph(G, output_file = "graph.png"):
    drawGraph(G, "", output_file = output_file)
    return G

