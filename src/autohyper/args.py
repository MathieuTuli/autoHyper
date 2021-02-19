from argparse import _SubParsersAction

import sys

mod_name = vars(sys.modules[__name__])['__package__']


def train_args(sub_parser: _SubParsersAction):
    # print("\n---------------------------------")
    # print("autoHyper Train Args")
    # print("---------------------------------\n")
    sub_parser.add_argument(
        '-vv', '--very-verbose', action='store_true',
        dest='very_verbose',
        help="Set flask debug mode")
    sub_parser.add_argument(
        '-v', '--verbose', action='store_true',
        dest='verbose',
        help="Set flask debug mode")
    sub_parser.set_defaults(verbose=False)
    sub_parser.set_defaults(very_verbose=False)
    sub_parser.add_argument(
        '--config', dest='config',
        default='config.yaml', type=str,
        help="Set configuration file path: Default = 'config.yaml'")
    sub_parser.add_argument(
        '--data', dest='data',
        default='data', type=str,
        help="Set data directory path: Default = 'data'")
    sub_parser.add_argument(
        '--output', dest='output',
        default='output', type=str,
        help="Set output directory path: Default = 'output'")
    sub_parser.add_argument(
        '--checkpoint', dest='checkpoint',
        default='checkpoint', type=str,
        help="Set checkpoint directory path: Default = 'checkpoint'")
    sub_parser.add_argument(
        '--resume', dest='resume',
        default=None, type=str,
        help="Set checkpoint resume path: Default = None")
    # sub_parser.add_argument(
    #     '-r', '--resume', action='store_true',
    #     dest='resume',
    #     help="Flag: resume training from checkpoint")
    # sub_parser.set_defaults(resume=False)
    sub_parser.add_argument(
        '--root', dest='root',
        default='.', type=str,
        help="Set root path of project that parents all others: Default = '.'")
    sub_parser.add_argument(
        '--save-freq', default=25, type=int,
        help='Checkpoint epoch save frequency: Default = 25')
    sub_parser.add_argument(
        '--cpu', action='store_true',
        dest='cpu',
        help="Flag: CPU bound training: Default = False")
    sub_parser.set_defaults(cpu=False)
    sub_parser.add_argument(
        '--gpu', default=0, type=int,
        help='GPU id to use: Default = 0')
    sub_parser.add_argument(
        '--multiprocessing-distributed', action='store_true',
        dest='mpd',
        help='Use multi-processing distributed training to launch '
        'N processes per node, which has N GPUs. This is the '
        'fastest way to use PyTorch for either single node or '
        'multi node data parallel training: Default = False')
    sub_parser.set_defaults(mpd=False)
    sub_parser.add_argument(
        '--dist-url', default='tcp://127.0.0.1:23456', type=str,
        help="url used to set up distributed training:" +
             "Default = 'tcp://127.0.0.1:23456'")
    sub_parser.add_argument(
        '--dist-backend', default='nccl', type=str,
        help="distributed backend: Default = 'nccl'")
    sub_parser.add_argument(
        '--world-size', default=-1, type=int,
        help='Number of nodes for distributed training: Default = -1')
    sub_parser.add_argument(
        '--rank', default=-1, type=int,
        help='Node rank for distributed training: Default = -1')
