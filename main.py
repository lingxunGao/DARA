"""DARA offline training and online adaptation entry point."""
import argparse
from pathlib import Path
import math

import torch
from disaster_response.training import train, online


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError('must be positive')
    return number


def nonnegative_float(value):
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError('must be finite and nonnegative')
    return number


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    offline = commands.add_parser('train', help='Jointly train the model on historical observations')
    streaming = commands.add_parser('online', help='Replay later observations and adapt the shared readout')
    for command in (offline, streaming):
        command.add_argument('--data', type=Path, required=True, help='Hourly NPZ observations; see README')
        command.add_argument('--output', type=Path, required=True)
        command.add_argument('--device', default='cpu')
        command.add_argument('--threads', type=positive_int, default=1)
        command.add_argument('--lr', type=nonnegative_float, default=1e-3)
    offline.add_argument('--epochs', type=positive_int, default=100)
    offline.add_argument('--batch-size', type=positive_int, default=32)
    offline.add_argument('--patience', type=positive_int, default=10)
    offline.add_argument('--train-fraction', type=float, default=.8)
    offline.add_argument('--weight-decay', type=nonnegative_float, default=1e-4)
    offline.add_argument('--balanced-weight', type=nonnegative_float, default=.05)
    offline.add_argument('--regularization', type=nonnegative_float, default=1e-4)
    offline.add_argument('--seed', type=int, default=0)
    streaming.add_argument('--checkpoint', type=Path, required=True)
    streaming.add_argument('--window', type=positive_int, default=12)
    streaming.add_argument('--anchor', type=nonnegative_float, default=.1)
    args = parser.parse_args()
    if args.lr <= 0:
        parser.error('--lr must be positive')
    if args.command == 'train':
        if not 0 < args.train_fraction < 1:
            parser.error('--train-fraction must be between 0 and 1')
        if args.balanced_weight > 1:
            parser.error('--balanced-weight must be between 0 and 1')
        if not 0 <= args.seed < 2**32:
            parser.error('--seed must be between 0 and 2**32 - 1')
    torch.set_num_threads(args.threads)
    (train if args.command == 'train' else online)(args)


if __name__ == '__main__':
    main()
