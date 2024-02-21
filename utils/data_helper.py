###############################################################################
#
# Some code is adapted from https://github.com/JiaxuanYou/graph-generation
#
###############################################################################
import os
import torch
import pickle
import numpy as np
from scipy import sparse as sp
import networkx as nx
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from HDMapGen.nuplanprocess import get_data_nuplan

__all__ = [
    'save_graph_list', 'load_graph_list', 'graph_load_batch',
    'preprocess_graph_list', 'create_graphs'
]


# save a list of graphs
def save_graph_list(G_list, fname):
  with open(fname, "wb") as f:
    pickle.dump(G_list, f)


def pick_connected_component_new(G):
  # import pdb; pdb.set_trace()

  # adj_list = G.adjacency_list()
  # for id,adj in enumerate(adj_list):
  #     id_min = min(adj)
  #     if id<id_min and id>=1:
  #     # if id<id_min and id>=4:
  #         break
  # node_list = list(range(id)) # only include node prior than node "id"

  adj_dict = nx.to_dict_of_lists(G)
  for node_id in sorted(adj_dict.keys()):
    id_min = min(adj_dict[node_id])
    if node_id < id_min and node_id >= 1:
      # if node_id<id_min and node_id>=4:
      break
  node_list = list(
      range(node_id))  # only include node prior than node "node_id"

  G = G.subgraph(node_list)
  G = max(nx.connected_component_subgraphs(G), key=len)
  return G


def load_graph_list(fname, is_real=True):
  with open(fname, "rb") as f:
    graph_list = pickle.load(f)

  # import pdb; pdb.set_trace()
  for i in range(len(graph_list)):
    edges_with_selfloops = list(graph_list[i].selfloop_edges())
    if len(edges_with_selfloops) > 0:
      graph_list[i].remove_edges_from(edges_with_selfloops)
    if is_real:
      graph_list[i] = max(
          nx.connected_component_subgraphs(graph_list[i]), key=len)
      graph_list[i] = nx.convert_node_labels_to_integers(graph_list[i])
    else:
      graph_list[i] = pick_connected_component_new(graph_list[i])
  return graph_list


def preprocess_graph_list(graph_list):
  for i in range(len(graph_list)):
    edges_with_selfloops = list(graph_list[i].selfloop_edges())
    if len(edges_with_selfloops) > 0:
      graph_list[i].remove_edges_from(edges_with_selfloops)
    if is_real:
      graph_list[i] = max(
          nx.connected_component_subgraphs(graph_list[i]), key=len)
      graph_list[i] = nx.convert_node_labels_to_integers(graph_list[i])
    else:
      graph_list[i] = pick_connected_component_new(graph_list[i])
  return graph_list


def graph_load_batch(data_dir,
                     min_num_nodes=20,
                     max_num_nodes=1000,
                     name='ENZYMES',
                     node_attributes=True,
                     graph_labels=True):
  '''
    load many graphs, e.g. enzymes
    :return: a list of graphs
    '''
  print('Loading graph dataset: ' + str(name))
  G = nx.Graph()
  # load data
  path = os.path.join(data_dir, name)
  data_adj = np.loadtxt(
      os.path.join(path, '{}_A.txt'.format(name)), delimiter=',').astype(int)
  if node_attributes:
    data_node_att = np.loadtxt(
        os.path.join(path, '{}_node_attributes.txt'.format(name)),
        delimiter=',')
  data_node_label = np.loadtxt(
      os.path.join(path, '{}_node_labels.txt'.format(name)),
      delimiter=',').astype(int)
  data_graph_indicator = np.loadtxt(
      os.path.join(path, '{}_graph_indicator.txt'.format(name)),
      delimiter=',').astype(int)
  if graph_labels:
    data_graph_labels = np.loadtxt(
        os.path.join(path, '{}_graph_labels.txt'.format(name)),
        delimiter=',').astype(int)

  data_tuple = list(map(tuple, data_adj))
  # print(len(data_tuple))
  # print(data_tuple[0])

  # add edges
  G.add_edges_from(data_tuple)
  # add node attributes
  for i in range(data_node_label.shape[0]):
    if node_attributes:
      G.add_node(i + 1, feature=data_node_att[i])
    G.add_node(i + 1, label=data_node_label[i])
  G.remove_nodes_from(list(nx.isolates(G)))

  # remove self-loop
  G.remove_edges_from(nx.selfloop_edges(G))

  # print(G.number_of_nodes())
  # print(G.number_of_edges())

  # split into graphs
  graph_num = data_graph_indicator.max()
  node_list = np.arange(data_graph_indicator.shape[0]) + 1
  graphs = []
  max_nodes = 0
  for i in range(graph_num):
    # find the nodes for each graph
    nodes = node_list[data_graph_indicator == i + 1]
    G_sub = G.subgraph(nodes)
    if graph_labels:
      G_sub.graph['label'] = data_graph_labels[i]
    # print('nodes', G_sub.number_of_nodes())
    # print('edges', G_sub.number_of_edges())
    # print('label', G_sub.graph)
    if G_sub.number_of_nodes() >= min_num_nodes and G_sub.number_of_nodes(
    ) <= max_num_nodes:
      graphs.append(G_sub)
      if G_sub.number_of_nodes() > max_nodes:
        max_nodes = G_sub.number_of_nodes()
      # print(G_sub.number_of_nodes(), 'i', i)
      # print('Graph dataset name: {}, total graph num: {}'.format(name, len(graphs)))
      # logging.warning('Graphs loaded, total num: {}'.format(len(graphs)))
  print('Loaded')
  return graphs

def graph_load_from_torchfile(data_dir, file_name, name='argoverse', node_attributes = True, graph_labels=True):
  path = os.path.join(data_dir, name, file_name)
  print("Loading graph from file: ", path)
  dataset  = np.load(path, allow_pickle=True)
  graphs = []

  for torch_graph in dataset:
      G = nx.Graph()
      num_nodes = torch_graph.num_nodes
      edge_index = torch_graph.edge_index.t().tolist()  # Convert edge_index to list of tuples
      for i in range(num_nodes):
          # node_features = {f'feature_{j}': dataset.x[i][j].item() for j in range(dataset.num_node_features)}
          if node_attributes:
            G.add_node(i, features = torch_graph.x[i])
          else:
            G.add_node(i)

      # Add edges
      for edge in edge_index:
        G.add_edge(edge[0], edge[1])
      if graph_labels:
        G.graph['label'] = torch_graph.y

      G.remove_nodes_from(list(nx.isolates(G)))
      G.remove_edges_from(nx.selfloop_edges(G))
      graphs.append(G)

  return graphs

def create_grid_with_embed(x_nodes, y_nodes, side_x = None, side_y = None, normalized = True,
                           noisy = False, noisy_std = 0.5, has_start_node = False):
    '''
    This function creates a grid with x_nodes in x-direction and y_nodes
    in y-direction. Each node has a 2D coordinate associated with it.
    The side-x and side-y specify side length in x and direction respectively.

    x_nodes: Number of nodes in x-direction
    y_nodes: Number of nodes in y-direction
    side_x: Sidelength in x-direction
    side_y: Sidelength in y-direction
    normalized: are nodes normalized or not
    noisy: Add a gaussian noise to each node position
    noisy_std: Standard deviation of noise, used only when noisy=True
    has_start_node: We add a start node which is connected to all nodes, this is our <START> token
    '''

    if side_x is None and side_y is not None:
        side_x = side_y
    elif side_x is not None and side_y is None:
        side_y = side_x
    elif side_x is None and side_y is None:
        side_x = side_y = 1
    
    mean_tensor = torch.tensor([0,0])
    std_tensor = torch.tensor([1,1])

    if normalized:
      coord_list = []
      for i in range(x_nodes):
        for j in range(y_nodes):
            coords = torch.tensor([float(i*side_x), float(j*side_y)])
            # if noisy:
            #   coords += torch.randn(coords.size()) * noisy_std
            coord_list.append(coords)
      coord_list = torch.stack(coord_list)
      mean_tensor = torch.mean(coord_list, dim=0)
      std_tensor = torch.var(coord_list, dim = 0)
    
    G = nx.Graph()
    num_node = 0
    for i in range(x_nodes):
        for j in range(y_nodes):
            coords = torch.tensor([float(i*side_x), float(j*side_y)]) - mean_tensor
            if noisy:
              coords += torch.randn(coords.size()) * noisy_std
            coords = coords / std_tensor
            G.add_node(num_node, features = coords)

            if j > 0:
                G.add_edge(num_node, num_node-1)
                G.add_edge(num_node -1, num_node)
            if i > 0:
                G.add_edge(num_node, num_node - y_nodes)
                G.add_edge(num_node - y_nodes, num_node)
            num_node += 1

    ## The node is connected to all nodes, in DFS since we select node by degree, this should
    ## by default be the start node then!!
    if has_start_node:
       G.add_node(num_node, features = torch.tensor([-1.0,-1.0], dtype=torch.float32))
       prev_node = num_node - 1
       while prev_node >= 0:
          G.add_edge(num_node, prev_node)
          G.add_edge(prev_node, num_node)
          prev_node -= 1
    return G

def create_subnode_with_embed(x_nodes, y_nodes, side_x = None, side_y = None, subdivisions=10, noisy = False, noisy_std = 0.1):
  '''
    This function creates a grid with 'x_nodes' in x-direction and 'y_nodes'
    in y-direction. Each node has a 2D coordinate associated with it.
    The side-x and side-y specify side length in x and direction respectively.
    Each edge is further subdivided into 'subdivisions' amount of sub-nodes

    x_nodes: Number of nodes in x-direction
    y_nodes: Number of nodes in y-direction
    side_x: Sidelength in x-direction
    side_y: Sidelength in y-direction
    subdivisions: Number of subdivisions of each edge 
  '''

  if side_x is None and side_y is not None:
        side_x = side_y
  elif side_x is not None and side_y is None:
      side_y = side_x
  elif side_x is None and side_y is None:
      side_x = side_y = 1
  
  G = nx.Graph()
  num_node = 0
  node_dict = {} # Hold all node coodinates
  for i in range(x_nodes):
      for j in range(y_nodes):
          coords = torch.tensor([float(i*side_x), float(j*side_y)])
          coords_noisy = coords.detach().clone()
          if noisy:
              coords_noisy += torch.randn(coords.size()) * noisy_std

          G.add_node(num_node, features = coords_noisy)
          node_dict[num_node] = coords
          t = torch.linspace(0, 1, subdivisions).reshape(-1, 1) #This will help us sample subnodes
          if j > 0:
              node_start = coords
              node_end = node_dict[num_node-1]
              subnodes = node_start + t*(node_end - node_start)
              G.add_edge(num_node, num_node-1)
              G.add_edge(num_node -1, num_node, subnodes = subnodes.flip(dims=(0,)))

          if i > 0:
              node_start = coords
              node_end = node_dict[num_node - y_nodes]
              subnodes = node_start + t*(node_end - node_start)

              G.add_edge(num_node, num_node - y_nodes)
              G.add_edge(num_node - y_nodes, num_node, subnodes = subnodes.flip(dims=(0,)))
          num_node += 1
  
  return G
    
def convert_nuplan_to_networkx(normalized = True):
  '''
  This function reads nuplan in the format that works for the model.
  normalized: Bool, we set this option to normalize the nodes, by default it will be true
  '''
  graphs = []
  maps = get_data_nuplan()

  for map in maps:
    G = nx.Graph()

    if normalized:
      mean_node = np.mean(map.nodes, axis=0)
      std_dev = np.std(map.nodes, axis=0)

    num_node = 0
    for node in map.nodes:
      if normalized:
         node = (node - mean_node) / std_dev
      G.add_node(num_node, features = torch.from_numpy(node).to(torch.float32))
      num_node += 1
    
    for edge_num in range(map.connections.shape[0]):
       start_node, end_node = map.connections[edge_num]
       subnodes = torch.from_numpy(map.subnodes[edge_num])
       if normalized:
          subnodes = (subnodes - torch.from_numpy(mean_node)) / torch.from_numpy(std_dev)

       G.add_edge(start_node, end_node, subnodes = subnodes)
       G.add_edge(end_node, start_node)
    
    graphs.append(G)
  
  return graphs
  

def create_graphs(graph_type, data_dir='data',is_noisy = False ,noise_std=1.0,has_start_node = False,seed=1234):
  npr = np.random.RandomState(seed)
  ### load datasets
  graphs = []
  # synthetic graphs
  if graph_type == 'grid':
    graphs = []
    for i in range(10, 20):
      for j in range(10, 20):
        graphs.append(nx.grid_2d_graph(i, j))    
  elif graph_type == 'grid_small':
    graphs = []
    for _ in range(50):
      graphs.append(nx.grid_2d_graph(10, 10))
  elif graph_type == 'lobster':
    graphs = []
    p1 = 0.7
    p2 = 0.7
    count = 0
    min_node = 10
    max_node = 100
    max_edge = 0
    mean_node = 80
    num_graphs = 100

    seed_tmp = seed
    while count < num_graphs:
      G = nx.random_lobster(mean_node, p1, p2, seed=seed_tmp)
      if len(G.nodes()) >= min_node and len(G.nodes()) <= max_node:
        graphs.append(G)
        if G.number_of_edges() > max_edge:
          max_edge = G.number_of_edges()
        
        count += 1

      seed_tmp += 1
  elif graph_type == 'DD':
    graphs = graph_load_batch(
        data_dir,
        min_num_nodes=100,
        max_num_nodes=500,
        name='DD',
        node_attributes=False,
        graph_labels=True)
    # args.max_prev_node = 230
  elif graph_type == 'FIRSTMM_DB':
    graphs = graph_load_batch(
        data_dir,
        min_num_nodes=0,
        max_num_nodes=10000,
        name='FIRSTMM_DB',
        node_attributes=False,
        graph_labels=True)
  elif graph_type == 'argoverse':
    graphs = graph_load_from_torchfile(data_dir,
                                       file_name='square_graphs_50x50.pkl',
                                       name='argoverse',
                                       node_attributes = True,
                                       graph_labels=True)
  elif graph_type == 'grid_embed':
    ## Try to add noise setting through config, currently hard coded
    graphs = []
    for _ in range(35):
        graphs.append(create_grid_with_embed(4,4, has_start_node=has_start_node, normalized=True))
        # graphs.append(create_grid_with_embed(3,3, has_start_node=has_start_node, normalized=True))
    # for _ in range(15):
    #     graphs.append(create_grid_with_embed(10,10, noisy=is_noisy, noisy_std=noise_std))
  
  elif graph_type == 'subnode':
    ## Try to add noise setting through config, currently hard coded
    graphs = []
    for _ in range(35):
        graphs.append(create_subnode_with_embed(3,3, subdivisions=20))
        graphs.append(create_subnode_with_embed(2,2, subdivisions=20))
    # for _ in range(15):
    #     graphs.append(create_subnode_with_embed(10,10,noisy=is_noisy, noisy_std=noise_std))
  
  elif graph_type == 'nuplan':
    output = convert_nuplan_to_networkx()
    # graphs = output
    graphs = []
    for i in range(35):
       graphs.append(output[0])
       graphs.append(output[1])



  num_nodes = [gg.number_of_nodes() for gg in graphs]
  num_edges = [gg.number_of_edges() for gg in graphs]
  print('max # nodes = {} || mean # nodes = {}'.format(max(num_nodes), np.mean(num_nodes)))
  print('max # edges = {} || mean # edges = {}'.format(max(num_edges), np.mean(num_edges)))
   
  return graphs

