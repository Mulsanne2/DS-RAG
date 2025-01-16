import torch
from torch.utils.data import Dataset

PATH = '/dataset/rankingqa/train'

class RetriveTrainGraphsDataset(Dataset):
    def __init__(self):
        super().__init__()

        self.graph = None

    def __getitem__(self, index):
        graph_ques = torch.load(f'{PATH}/graphs_q/{index}.pt',weights_only=False)
        doc_emb = torch.load(f'{PATH}/document_emb/{index}.pt',weights_only=False)
        
        return {
            'id': index,
            'ques_graph': graph_ques,
            'doc_emb' : doc_emb
        }

    def get_idx(self):
        # Load the saved indices
        with open(f'{PATH}/indices.txt', 'r') as file:
            indices = [int(line.strip()) for line in file]

        return {'idx': indices}

if __name__ == '__main__':
    dataset = RetriveTrainGraphsDataset()

    data = dataset[1]
    for k, v in data.items():
        print(f'{k}: {v}')

    split_ids = dataset.get_idx_split()
    for k, v in split_ids.items():
        print(f'# {k}: {len(v)}')
