"""Repeatable full-length corpus sample timing; no relevance labels are consulted."""
import argparse
import gc
import json
import time
from pathlib import Path
import numpy as np
from convrag.data import corpus,COLLECTIONS
from convrag.neural import Encoder
from convrag.io import write_json

def main():
    p=argparse.ArgumentParser();p.add_argument('--backend',choices=['dense','splade'],required=True)
    p.add_argument('--batch-size',type=int,default=8);args=p.parse_args()
    root=Path.cwd();encoder=Encoder(root,args.backend)
    sample=[]
    for d in COLLECTIONS:
        # Deterministic distributed samples from the corpus, rather than short first passages only.
        sample.extend([r['text'] for i,r in enumerate(corpus(root/'data/upstream',d)) if i%5000==0][:12])
    durations=[];count=0
    for i in range(0,len(sample),args.batch_size):
        batch=sample[i:i+args.batch_size];t=time.perf_counter();encoder.encode(batch)
        elapsed=time.perf_counter()-t;durations.append(elapsed);count+=len(batch)
    rate=count/sum(durations)
    result={'backend':args.backend,'model':encoder.model_id,'revision':encoder.revision,
            'device':encoder.device,'sample_passages':count,'max_length':encoder.max_length,
            'batch_size':args.batch_size,'passages_per_second':rate,
            'estimated_full_index_hours':366479/rate/3600,
            'estimate_limits':'Small sample, cold-start and concurrent work; not a completed indexing measurement',
            'decision':'defer full corpus indexing in this CPU-only session; keep implementation available'}
    write_json(root/'analysis'/('feasibility_'+args.backend+'.json'),result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
