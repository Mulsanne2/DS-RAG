import matplotlib.pyplot as plt
import networkx as nx

def visualize_graph(G):
  labels = {node: f"{node}" for node, data in G.nodes(data=True)}
  plt.figure(figsize=(12, 12))
  pos = nx.spring_layout(G, k=1.5)

  node_colors = [data.get('color', '#00BFFF') for node, data in G.nodes(data=True)]
  nx.draw(G, pos, labels=labels, node_color=node_colors, edge_color='gray', node_size=5000, font_size=10, arrows=True)
  edge_labels = nx.get_edge_attributes(G, 'relation')
  nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
  plt.show()