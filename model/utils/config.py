import argparse


def parse_args_GR():
    parser = argparse.ArgumentParser(description="G-Reranker")

    parser.add_argument("--model_name", type=str, default='Graph-Reranker')
    parser.add_argument("--gpu_device", type=str, required=True)
    parser.add_argument("--expt_name", type=str, required=True)

    # Model Training
    parser.add_argument("--hidden_dim", type=int, default=3072, required=True)
    parser.add_argument("--out_dim", type=int, default=1536, required=True)
    parser.add_argument("--weight_opt", type=int, default=1, required=True)
    parser.add_argument("--num_epochs", type=int, default=100, required=True)
    parser.add_argument("--batch_size", type=int, default=4, required=True)
    parser.add_argument("--lr", type=float, default=1e-5, required=True)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)

    args = parser.parse_args()
    return args
