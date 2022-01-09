"""
Script for evaluating the metrics of the results.
"""
from argparse import ArgumentParser
from evaluate_functions import evaluate_results

"""
Arguments loading
"""
parser = ArgumentParser()
parser.add_argument('--real_path', default='../DAVIS/2017/trainval')
parser.add_argument('--pred_path', default='../results')
parser.add_argument('--output_path', default='../metric_results')
# parser.add_argument('--imset')
args = parser.parse_args()

real_path = args.real_path
pred_path = args.pred_path
out_path = args.output_path
# imset = args.imset

evaluate_results(pred_path, real_path, out_path)
