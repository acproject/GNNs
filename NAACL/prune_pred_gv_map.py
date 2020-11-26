"""Prune model predictions with rule-based G-V linker."""
import argparse
import collections
import os
import sys

from NAACL import settings

OPTS = None

GV_MAP_FILE = 'gene_var/gene_to_var.tsv'

def prep_gv_mapping():
    var_to_gene= {}
    gene_to_var= collections.defaultdict(set)
    pmid_to_gv = collections.defaultdict(set)
    pmid_gv_map = {}
    with open(os.path.join(settings.DATA_DIR, GV_MAP_FILE)) as f:
        for line in f:
            pmid, variant, gene = line.strip().strip()
            gene = gene.lower()
            var_to_gene[(pmid, variant)] = gene
            gene_to_var[(pmid, gene)].add(variant)
            pmid_to_gv[pmid].add((gene, variant))

    return var_to_gene, gene_to_var, pmid_to_gv

def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('pred_file')
    parser.add_argument('out_file')
    if len(args) == 0:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args(args)

def main(OPTS):
    var_to_gene, gene_to_var, pmid_to_gv = prep_gv_mapping()
    with open(OPTS.pred_file) as fin:
        with open(OPTS.out_file) as fout:
            for line in fin:
                idx, pmid, d, g, v, prob = line.strip().split('\t')
                if(pmid, v) not in var_to_gene:
                    continue
                g_linked = var_to_gene[(pmid, v)]
                if g_linked == g:
                    fout.write(line)

if __name__ == '__main__':
    OPTS = parse_args(sys.argv[1:])
    main(OPTS)