import argparse


def load_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--cuda', type=bool, default=True)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--results', type=bool, default=True)
    parser.add_argument('--print_intervals', type=int, default=100)
    parser.add_argument('--training', type=bool, default=False)
    parser.add_argument('--evaluation', type=bool, default=False)

    return parser.parse_args()
