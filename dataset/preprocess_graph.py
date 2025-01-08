import os
import torch
import pandas as pd
from openai import OpenAI
from tqdm import tqdm
from torch_geometric.data.data import Data
from generate_split import generate_split
import json
from collections import Counter
import numpy as np


os.environ['OPENAI_API_KEY'] = ""
client = OpenAI()

path = 'graph/compasrion/train'
test_data_path = 'comparison/comparison_train_graph.json'

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

def textualize_graph(ques_graph):
    nodes_ques = {}
    edges_ques = []
    node_counter = Counter()
    for src,dst in ques_graph:
        src = src.lower().strip()
        dst = dst.lower().strip()
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


def step_one(relations,documents):
    # generate textual graphs
    def replace_string_in_csv(path):
        df = pd.read_csv(path)
        df.to_csv(path, index=False)

    os.makedirs(f'{path}/nodes_q', exist_ok=True)
    os.makedirs(f'{path}/relations_q', exist_ok=True)
    os.makedirs(f'{path}/document_emb', exist_ok=True)
            
    print("question embedding start")
    for i,ques in enumerate(tqdm(relations)):
        nodes_q, edges_q = textualize_graph(ques)
        nodes_q.to_csv(f'{path}/nodes_q/{i}.csv', index=False, columns=['node_id', 'node_attr', 'node_weight'])
        edges_q.to_csv(f'{path}/relations_q/{i}.csv', index=False, columns=['src', 'dst'])

        replace_string_in_csv(f'{path}/nodes_q/{i}.csv')

    print("question embedding finish")
    
    print("document embedding start")
    for i,a in enumerate(tqdm(documents)):
        a = str(a)
        emb = get_embedding(a)
        torch.save(emb, f'{path}/document_emb/{i}.pt')

    print("document embedding finish")
    
def step_two(relations,documents,cores, cores_onehop):

    def _encode_graph():
        print('Encoding graphs...')
        os.makedirs(f'{path}/graphs_q', exist_ok=True)
        for i in tqdm(range(len(relations))):
            #Generate Document Graph Embedding
            nodes = pd.read_csv(f'{path}/nodes_q/{i}.csv')
            edges = pd.read_csv(f'{path}/relations_q/{i}.csv')
            x=[]
            core_node_idx= -1
            core_onehop_idx=-1
            for idx, a in enumerate(nodes.node_attr.tolist()):
                a = str(a)
                x.append(get_embedding(a))
                if cores[i] is not None and cores[i].lower() == a.lower():
                    core_node_idx = idx
                if cores_onehop[i] is not None and cores_onehop[i].lower() == a.lower():
                    core_onehop_idx = idx

            x = torch.tensor(x)
            edge_index = torch.LongTensor([edges.src, edges.dst])
            node_weight = nodes.node_weight.tolist()

            # core_onehop = get_destinations(core_node_idx, edge_index)

            data = Data(x=x,
                        edge_index=edge_index, 
                        node_weight = node_weight,
                        core_node = core_node_idx,
                        core_onehop = core_onehop_idx,
                        node_num = len(x)
                        )
            torch.save(data, f'{path}/graphs_q/{i}.pt')

    _encode_graph()


if __name__ == '__main__':

    relations = []
    documents = []
    cores = []
    cores_onehop = []
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    for entry in test_data:
        relation_temp = []

        document = entry.get("gt_doc")
        rel_temp = entry.get("rener")
        core = entry.get("core_node")
        qdpair = entry.get("dqd_pair")
        for temp in rel_temp:
            src = temp['subject']
            dst = temp['object']
            relation_temp.append([src,dst])
        for qd in qdpair:
            documents.append(qd['gt_doc'])
            relations.append(relation_temp)
            cores.append(core)
            cores_onehop.append(qd['one_hop_node'])

    step_one(relations,documents)
    step_two(relations,documents,cores,cores_onehop)
    generate_split(len(relations), f'{path}')
