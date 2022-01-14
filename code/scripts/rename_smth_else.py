"""
Script to rename files in something-else folder.
"""

import os
from argparse import ArgumentParser

parser = ArgumentParser()

parser.add_argument('--smth_else_path', default='../videos')

args = parser.parse_args()

if os.path.exists(args.smth_else_path):
    os.rename(args.smth_else_path, os.path.join(os.path.dirname(args.smth_else_path), 
            'something-something'))

new_smth_else_path = os.path.join(os.path.dirname(args.smth_else_path), 'something-something')

if not os.path.exists(os.path.join(new_smth_else_path, 'videos')):
    os.makedirs(os.path.join(new_smth_else_path, 'videos'))

filenames = [f for f in os.listdir(new_smth_else_path) if not f == 'tracking_annotations' and
        not f == 'videos' and not f.startswith('.')]

for filename in filenames:
    os.rename(os.path.join(new_smth_else_path, filename), 
              os.path.join(new_smth_else_path, 'videos', filename))
