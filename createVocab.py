import argparse
from collections import defaultdict
from operator import itemgetter

parser = argparse.ArgumentParser(description='Process command line arguments.')
parser.add_argument("-r", "--read", dest="filename", help="path to the file from which to extract the vocab", metavar="TR_FILE")
parser.add_argument("-v", "--vocab", dest="vocabulary", help="path to the file in which to save the vocab", metavar="VOC_FILE")
args = parser.parse_args()

with open(args.filename, 'r') as f:
	counts = defaultdict(int)
	for line in f:
		tokens = line.split()
		for token in tokens:
			counts[token] += 1

if '<s>' in counts: del counts['<s>']
if '</s>' in counts: del counts['</s>']
sorted_counts = sorted(counts.items(), key=itemgetter(1), reverse=True)
with open(args.vocabulary, 'w') as f:
	f.write( '<S>' + '\n' )
	f.write( '</S>' + '\n' )
	f.write( '<UNK>' + '\n' )
	for k, v in sorted_counts:
		if v >= 5: f.write( k + '\n')
		
#print(counts)
