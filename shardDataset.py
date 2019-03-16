#!/usr/bin/python

import argparse
import os


def main(args):
    # First pass counts number of sentences in the dataset
    sentences = 0
    fname = ''
    with open(args.dataset_file, 'r') as f:
        fname = os.path.basename(f.name)
        for sentence in f:
            sentences += 1
    
    sentences_per_shard = sentences / args.shards
    
    # # Delete the shard files if they exist
    # for shard_id in range(args.shards):
    #     f = open(args.save_dir + '/%s.shard%d' % (fname, shard_id), 'w')
    #     f.close()

    shard_id = 0
    saved = 0
    fw = open(args.save_dir + '/%s.shard%d' % (fname, shard_id), 'w')
    with open(args.dataset_file, 'r') as f:
        for sentence in f:
            fw.write(sentence)
            saved += 1
            if saved % sentences_per_shard == 0:
                fw.close()
                shard_id += 1
                fw = open(args.save_dir + '/' + 'fname.shard%d' % shard_id, 'w')
    fw.close()   
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_file', help='Location of the dataset file')
    parser.add_argument('--save_dir', help='Location of the folder to store the dataset')
    parser.add_argument('--shards', help='Number of shards to split the dataset into', type=int, default=100)
    
    args = parser.parse_args()
    main(args)
