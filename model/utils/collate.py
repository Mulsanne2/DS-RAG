from torch_geometric.data import Batch
import torch


def collate_fn(original_batch):
    batch = {}
    original_batch = original_batch[0]
    for k in original_batch[0].keys():
        batch[k] = [d[k] for d in original_batch]
    if 'ques_graph' in batch:
        batch['ques_graph'] = Batch.from_data_list(batch['ques_graph'])
    
    batch['doc_emb'] = torch.tensor(batch['doc_emb']) #Chang List -> Tensor
    return batch
