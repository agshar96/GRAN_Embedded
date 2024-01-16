import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.collections import LineCollection


def draw_graph_list(G_list,
                    row,
                    col,
                    fname='exp/gen_graph.png',
                    layout='spring',
                    is_single=False,
                    k=1,
                    node_size=55,
                    alpha=1,
                    width=1.3):
  plt.switch_backend('agg')
  for i, G in enumerate(G_list):
    plt.subplot(row, col, i + 1)
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)
    # plt.axis("off")

    # turn off axis label
    plt.xticks([])
    plt.yticks([])

    if layout == 'spring':
      pos = nx.spring_layout(
          G, k=k / np.sqrt(G.number_of_nodes()), iterations=100)
    elif layout == 'spectral':
      pos = nx.spectral_layout(G)

    if is_single:
      # node_size default 60, edge_width default 1.5
      nx.draw_networkx_nodes(
          G,
          pos,
          node_size=node_size,
          node_color='#336699',
          alpha=1,
          linewidths=0,
          #font_size=0
          )
      nx.draw_networkx_edges(G, pos, alpha=alpha, width=width)
    else:
      nx.draw_networkx_nodes(
          G,
          pos,
          node_size=1.5,
          node_color='#336699',
          alpha=1,
          linewidths=0.2,
          #font_size=1.5
          )
      nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.2)

  plt.tight_layout()
  plt.savefig(fname, dpi=300)
  plt.close()


def draw_graph_list_separate(G_list,
                    fname='exp/gen_graph',
                    layout='spring',
                    is_single=False,
                    k=1,
                    node_size=55,
                    alpha=1,
                    width=1.3):
  
  for i, G in enumerate(G_list):
    plt.switch_backend('agg')
    
    plt.axis("off")

    # turn off axis label
    # plt.xticks([])
    # plt.yticks([])

    if layout == 'spring':
      pos = nx.spring_layout(
          G, k=k / np.sqrt(G.number_of_nodes()), iterations=100)
    elif layout == 'spectral':
      pos = nx.spectral_layout(G)

    if is_single:
      # node_size default 60, edge_width default 1.5
      nx.draw_networkx_nodes(
          G,
          pos,
          node_size=node_size,
          node_color='#336699',
          alpha=1,
          linewidths=0,
          #font_size=0
          )
      nx.draw_networkx_edges(G, pos, alpha=alpha, width=width)
    else:
      nx.draw_networkx_nodes(
          G,
          pos,
          node_size=1.5,
          node_color='#336699',
          alpha=1,
          linewidths=0.2,
          #font_size=1.5
          )
      nx.draw_networkx_edges(G, pos, alpha=0.3, width=0.2)

    plt.draw()
    plt.tight_layout()
    plt.savefig(fname+'_{:03d}.png'.format(i), dpi=300)
    plt.close()

## Function to draw graphs that use the 2D embeddings as x,y coordinate
def draw_graph_list_embed(G_list, row, col, fname= 'test_graph_plots'):

    for i,G in enumerate(G_list):
        plt.subplot(row,col,i+1)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1,
                        wspace=0, hspace=0)
        
        node_embed_output = list(G.nodes(data=True))
        node_embed_output = np.array([item[1]['features'] for item in node_embed_output])
        adj = np.asarray(nx.to_numpy_array(G))

        #Plot edges
        x =[]
        y =[]
        for i in range(node_embed_output.shape[0]):
            coord1 = node_embed_output[i]
            neighbors = np.where(np.isclose(adj[i], 1))[0]
            for node in neighbors:
                coord2 = node_embed_output[node]
                x.extend([coord1[0], coord2[0], None])
                y.extend([coord1[1], coord2[1], None])
        plt.scatter(x,y, s=3, color='r')
        plt.plot(x,y, '-k')
    
    plt.savefig(fname+'.png', dpi=600)
    plt.close()

## This function only draws the node location without any edges
def draw_graph_nodes_list(G_list, row, col, fname= 'test_node_plots'):

    for i,G in enumerate(G_list):
        plt.subplot(row,col,i+1)
        plt.subplots_adjust(left=0, bottom=0, right=1, top=1,
                        wspace=0, hspace=0)
        
        node_embed_output = list(G.nodes(data=True))
        node_embed_output = np.array([item[1]['features'] for item in node_embed_output])
        adj = np.asarray(nx.to_numpy_array(G))

        # #Plot edges
        # x =[]
        # y =[]
        # for i in range(node_embed_output.shape[0]):
        #     coord1 = node_embed_output[i]
        #     neighbors = np.where(np.isclose(adj[i], 1))[0]
        #     for node in neighbors:
        #         coord2 = node_embed_output[node]
        #         x.extend([coord1[0], coord2[0], None])
        #         y.extend([coord1[1], coord2[1], None])
        # plt.scatter(x,y, s=3, color='r')
        # plt.plot(x,y, '-k')

        x_coordinates = node_embed_output[:, 0]
        y_coordinates = node_embed_output[:, 1]
        plt.scatter(x_coordinates, y_coordinates, s=5, color='r')
    
    plt.savefig(fname+'.png', dpi=600)
    plt.close()

def draw_animated_plot(graph_list, theta_list, num_graphs, x_lim=(-1,4), y_lim = (-1,4)):

   for i in range(num_graphs):
      
      graph_to_plot = graph_list[i]
      theta_cur = theta_list[i]

      node_embed_output = list(graph_to_plot.nodes(data=True))
      node_embed_output = np.array([item[1]['features'] for item in node_embed_output])
      adj = np.asarray(nx.to_numpy_array(graph_to_plot))
      lower_adj = np.tril(adj)

      fig, ax = plt.subplots()
      sc = ax.scatter([], [])
      lines = LineCollection([])
      ax.add_collection(lines)
      ax.set_xlim(x_lim)
      ax.set_ylim(y_lim)

      prev_alpha = None
      def update(frame):

          nonlocal prev_alpha
          if frame % 2 == 0:
              cur_frame = int(frame/2)
              sc.set_offsets(np.column_stack((node_embed_output[:cur_frame+1,0], 
                                              node_embed_output[:cur_frame+1,1])))
              if cur_frame != 0:
                  segments = []
                  i=0
                  while i < cur_frame:
                      cur_line = np.vstack((node_embed_output[i],node_embed_output[cur_frame]))
                      segments.append(cur_line)
                      i+=1
                  prev_segments = lines.get_segments()
                  segments = np.stack(segments)
                  theta = theta_cur[cur_frame,:cur_frame]
                  if prev_segments:
                      prev_segments = np.stack(prev_segments)
                      segments = np.vstack((prev_segments, segments))
                      theta = np.concatenate((prev_alpha, theta))
                  
                  lines.set_segments(segments)
                  lines.set_alpha(theta)

          else:
              # pass
              if frame != 1:
                  cur_frame = int((frame - 1)/2)
                  alpha = lower_adj[cur_frame,:cur_frame]
                  if prev_alpha is not None:
                      alpha = np.concatenate((prev_alpha, alpha))
                  # if frame == 2*node_embed_output.shape[0] -1 :
                  lines.set_alpha(alpha)
                  prev_alpha = alpha

          return sc,
          
      animation = FuncAnimation(fig, update, frames=2*node_embed_output.shape[0], interval=500, blit=True)
      animation_fname = 'graph_embed_animation_' + str(i) + '.mp4'
      animation.save(animation_fname, fps=1)