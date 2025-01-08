""" QGS RAG Graph Attention Network using Constrastive Learning """
import torch
import random
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv, GATConv
from torch.nn import Linear, Dropout
from torch.utils.data import DataLoader, IterableDataset
from utils.train_retriever_graphs import RetriveTrainGraphsDataset
from utils.test_retriever_graphs import RetriveTestGraphsDataset
from utils.collate import collate_fn
from utils.config import parse_args_GR
from torch_scatter import scatter
from tqdm import tqdm
import torch.nn as nn
import wandb

wandb.init(project="QG_Graph_comparison")
args = parse_args_GR()
hidden_dim = args.hidden_dim
out_dim = args.out_dim
wandb.run.name = args.expt_name
device = torch.device(f'{args.gpu_device}' if torch.cuda.is_available() else 'cpu')
print(f"device using : {device}")
print(args)
cfg = {
    "learning_rate" : args.lr,
    "epochs" : args.num_epochs,
    "batch_size" : args.batch_size,
    "dropout" : args.dropout,
    "hidden_dim" : args.hidden_dim,
    "out_dim" : args.out_dim
}
wandb.config.update(cfg)

class DynamicBatchDataset(IterableDataset):
    def __init__(self, data, batch_sizes):
        self.data = data
        self.batch_sizes = batch_sizes

    def __iter__(self):
        data_iter = iter(self.data)
        for batch_size in self.batch_sizes:
            batch = []
            for _ in range(batch_size):
                try:
                    batch.append(next(data_iter))
                except StopIteration:
                    break
            if batch:
                yield batch

train = RetriveTrainGraphsDataset()
test = RetriveTestGraphsDataset()

with open('dataset/graph/compasrion/train', "r") as file:
    train_batch_list = [int(line.strip()) for line in file if line.strip()]

with open('dataset/graph/compasrion/test', "r") as file:
    test_batch_list = [int(line.strip()) for line in file if line.strip()]

train_dataset = DynamicBatchDataset(train, train_batch_list)
test_dataset = DynamicBatchDataset(test, test_batch_list)

def accuracy(ques, doc, batch_size):
    """Calculate accuracy."""

    acc = 0
    for i in range(batch_size):
        acc += F.cosine_similarity(ques[i].unsqueeze(0), doc[i].unsqueeze(0))

    accuracy = acc.item() / batch_size
    return accuracy

def GetGraphRepresentation(ques_data, ques_node, batch_size, option_no):
    ques_graph = []
    ques_node_split = []
    for i in range(batch_size):
        startidx = ques_data.ptr[i]
        endidx = ques_data.ptr[i+1]
        ques_node_split.append(ques_node[startidx:endidx])
    
    for i in range(batch_size):
        one_hop_idx = ques_data[i].core_onehop[0].item() + ques_data.ptr[i].item()
        ques_graph.append(ques_node[one_hop_idx])

    ques_graph = torch.stack(ques_graph)
    return ques_graph

class GATModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads=4):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=8, concat=False))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, hidden_channels, heads=num_heads, concat=False))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GATConv(hidden_channels, out_channels, heads=num_heads, concat=False))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index=edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index=edge_index)
        return x

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads=4):
        super().__init__()
        self.gat_ques = GATModel(in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads)
        self.doc_projection = nn.Linear(in_channels, out_channels)

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, ques_data, doc_data):
        #graph attention network only using node feature and edge index features
        ques_node_embed = self.gat_ques(ques_data.x, ques_data.edge_index.long())

        doc_emb = self.doc_projection(doc_data)

        return ques_node_embed, doc_emb
        
    def info_nce_loss(self, ques, doc, batch_size = 4 ,temperature=0.07):
        similarity_matrix = torch.zeros(batch_size, batch_size).to(device)

        for i in range(batch_size):
            for j in range(batch_size):
                similarity_matrix[i, j] = F.cosine_similarity(ques[i].unsqueeze(0), doc[j].unsqueeze(0))

        similarity_matrix = similarity_matrix / temperature

        labels = torch.arange(batch_size).to(device)
        loss_1 = F.cross_entropy(similarity_matrix, labels)
        loss_2 = F.cross_entropy(similarity_matrix.T, labels)

        loss = (loss_1 + loss_2) / 2
        return loss
    

    def fit(self, train_data, test_data, epochs, lr=1e-4):

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=0.01)

        LOSS_MIN = 99999.0
        self.train()
        for epoch in tqdm(range(epochs+1)):
            self.train()

            train_loss = 0.0
            train_acc = 0.0
            train_samples = 0
            for _, batch in enumerate(train_data):
                batch_size=len(batch['id'])
                if(batch_size<=1):
                    continue
                ques_data = batch['ques_graph'].to(device)
                doc_data = batch['doc_emb'].to(device)

                # Go to Graph Attention Network
                ques_node, doc_emb = self(ques_data, doc_data)
                ###### 32x1536 Todo.Changing method algoritm ######
                ques_graph = GetGraphRepresentation(ques_data,ques_node,batch_size,args.weight_opt)

                optimizer.zero_grad()
                loss = self.info_nce_loss(ques_graph, doc_emb, batch_size)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                acc = accuracy(ques_graph, doc_emb, batch_size)
                train_acc += acc
                train_samples += 1
 
            if(epoch % 5 == 0):
                self.eval()  #change mode to eval
                test_samples = 0
                test_loss=0
                test_acc = 0
                temp_id = 246

                with torch.no_grad():  
                    for _, batch in enumerate(test_data):
                        if _ == temp_id:
                            hi=1
                        batch_size=len(batch['id'])
                        if(batch_size<=1):
                            continue
                        ques_data = batch['ques_graph'].to(device)
                        doc_data = batch['doc_emb'].to(device)

                        # Go to Graph Attention Network
                        ques_node, doc_emb = self(ques_data, doc_data)

                        ques_graph = GetGraphRepresentation(ques_data,ques_node,batch_size,args.weight_opt)
                        
                        loss = self.info_nce_loss(ques_graph, doc_emb, batch_size=batch_size)
                        test_loss += loss.item()
                        acc = accuracy(ques_graph, doc_emb, batch_size=batch_size)
                        test_acc += acc
                        test_samples += 1
                test_loss /= test_samples
                test_acc /= test_samples 
                train_acc /= train_samples
                train_loss /= train_samples
            
                print(f'\nEpoch {epoch:>3} | Train Loss: {train_loss:.3f} | Train Acc: {train_acc*100:5.2f}% | Test Loss: {test_loss:.2f} | Test Acc: {test_acc*100:5.2f}%')
                wandb.log({"train_loss": train_loss, "train_acc":train_acc*100, "test_loss":test_loss, "test_acc":test_acc*100}, step=epoch)

                print(f'Minimum Loss : {LOSS_MIN:0.3f}, New Loss : {test_loss:0.3f}')
                if test_loss < LOSS_MIN:
                    print("***Update Model***")
                    LOSS_MIN = test_loss
                    torch.save(self.state_dict(), f'model/model_weight/{args.expt_name}.pth')
 
    @torch.no_grad()
    def test(self, data):
        self.eval()  
        total_samples = 0
        test_loss=0
        test_acc = 0

        with torch.no_grad():  
            for _, batch in enumerate(data):
                batch_size = len(batch['id'])
                if(batch_size<=1):
                    continue
                ques_data = batch['ques_graph'].to(device)
                doc_data = batch['doc_emb'].to(device)

                # Go to Graph Attention Network
                ques_node, doc_emb = self(ques_data, doc_data)

                ques_graph = GetGraphRepresentation(ques_data,ques_node,batch_size,args.weight_opt)
                
                loss = self.info_nce_loss(ques_graph, doc_emb, batch_size)
                test_loss += loss.item()
                acc = accuracy(ques_graph, doc_emb, batch_size)
                test_acc += acc
                total_samples += 1

        loss = test_loss/total_samples
        acc = test_acc/total_samples

        print(f'Graph Similarity test | loss : {loss:0.3f} | accuracy : {acc*100:>5.2f}%')
        wandb.log({"final_test_loss":loss, "final_test_acc":acc})

        return 
 
def main(args):
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    random.seed(args.seed)
    # Create the Vanilla GNN model
    gat = GAT(in_channels=3072, hidden_channels=hidden_dim, out_channels=out_dim, num_layers=4, dropout=args.dropout, num_heads=4).to(device)
    print(gat)

    # Get test Dataset
    train_loader = DataLoader(train_dataset, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, collate_fn=collate_fn)

    # Train
    gat.fit(train_loader, test_loader, epochs=args.num_epochs, lr=args.lr)

    # Test
    gat_test = GAT(in_channels=3072, hidden_channels=hidden_dim, out_channels=out_dim, num_layers=4, dropout=0, num_heads=4).to(device)
    gat_test.load_state_dict(torch.load(f'model/model_weight/{args.expt_name}.pth', weights_only=True,map_location=device))
    gat_test.test(test_loader)

if __name__ == "__main__":
    main(args)