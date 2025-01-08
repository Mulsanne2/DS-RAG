import torch
import os
from openai import OpenAI
from collections import Counter
import re
import pandas as pd
from torch_geometric.data.data import Data
import numpy as np

os.environ["OPENAI_API_KEY"] = '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'
client = OpenAI()

def get_embedding(text, model="text-embedding-3-large"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

def get_destinations(source_number, data):
    """
    Given a source number, return all associated destination numbers.
    """
    # Find indices where the source matches the source_number
    indices = np.where(data[0] == source_number)
    # Return the corresponding destinations
    return data[1][indices]

def GenGraph(ques_graph, core, doc1, doc2):

    def textualize_graph(ques_rel):
        nodes_ques = {}
        edges_ques = []
        node_counter = Counter()
        for src, dst in ques_rel:
            src = src.lower().strip()
            dst = dst.lower().strip()
            # Increment node occurrences
            node_counter[src] += 1
            node_counter[dst] += 1
            if src not in nodes_ques:
                nodes_ques[src] = len(nodes_ques)
            if dst not in nodes_ques:
                nodes_ques[dst] = len(nodes_ques)
            edges_ques.append({'src': nodes_ques[src], 'dst': nodes_ques[dst], })

        nodes_ques = pd.DataFrame(nodes_ques.items(), columns=['node_attr', 'node_id'])
        nodes_ques['node_weight'] = nodes_ques['node_attr'].map(node_counter)
        edges_ques = pd.DataFrame(edges_ques)
        return nodes_ques, edges_ques
    
        
    def _encode_graph(nodes, edges, core):

        #Generate Document Graph Embedding
        x = []
        core_node_idx = -1
        for idx, a in enumerate(nodes.node_attr.tolist()):
            a = str(a)
            x.append(get_embedding(a))
            if core is not None and core.lower() == a.lower():
                core_node_idx = idx

        x = torch.tensor(x)
        edge_index = torch.LongTensor([edges.src, edges.dst])
        node_weight = torch.from_numpy(nodes.node_weight.values)

        core_onehop = get_destinations(core_node_idx, edge_index)

        data = Data(x=x,
                    edge_index=edge_index, 
                    node_weight=node_weight,
                    core_node = core_node_idx,
                    core_onehop = core_onehop,
                    node_num = len(x)
                    )
        
        return data

    nodes_q, edges_q = textualize_graph(ques_graph)
    ques_graph = _encode_graph(nodes_q, edges_q, core)

    doc_emb1 = []
    for doc in doc1:
        doc_emb1.append(get_embedding(doc))
    
    doc_emb2 = []
    for doc in doc2:
        doc_emb2.append(get_embedding(doc))

    return ques_graph, doc_emb1, doc_emb2