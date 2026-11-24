from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
import bm25s
import Stemmer
from .io import write_json
from .data import corpus

def rrf(rankings,weights=None,k=60):
    weights=weights or [1.0]*len(rankings)
    if len(weights)!=len(rankings): raise ValueError("Weight/list mismatch")
    scores={}
    for ranking,w in zip(rankings,weights):
        seen=set()
        for rank,(pid,_) in enumerate(ranking,1):
            if pid in seen: continue
            seen.add(pid); scores[pid]=scores.get(pid,0)+w/(k+rank)
    return sorted(scores.items(),key=lambda x:(-x[1],x[0]))

class BM25Index:
    def __init__(self,path):
        self.path=Path(path)
        self.engine=bm25s.BM25.load(str(path),mmap=True,load_corpus=False,verbosity=0)
        self.ids=json.loads((self.path/"ids.json").read_text())
        self.stemmer=Stemmer.Stemmer("english")
    @classmethod
    def build(cls,path,docs):
        path=Path(path)
        docs=list(docs); start=time.perf_counter()
        stemmer=Stemmer.Stemmer("english")
        tokens=bm25s.tokenize([d["text"] for d in docs],stopwords="english",stemmer=stemmer,show_progress=False)
        engine=bm25s.BM25(method="lucene",k1=1.2,b=.75)
        engine.index(tokens,show_progress=False)
        engine.save(str(path),show_progress=False)
        write_json(path/"ids.json",[d["id"] for d in docs])
        write_json(path/"metadata.json",{"backend":"bm25","documents":len(docs),"seconds":time.perf_counter()-start,
                   "tokenizer":"bm25s default, English stopwords, English Snowball stemmer",
                   "k1":1.2,"b":.75,"method":"lucene"})
    def search(self,query,k=100):
        tokens=bm25s.tokenize(query,stopwords="english",stemmer=self.stemmer,show_progress=False)
        ids,scores=self.engine.retrieve(tokens,k=min(k,len(self.ids)),show_progress=False)
        ranking=[(self.ids[int(i)],float(s)) for i,s in zip(ids[0],scores[0])]
        return sorted(ranking,key=lambda x:(-x[1],x[0]))

def build_bm25(root,upstream,domains):
    for d in domains:
        p=Path(root)/"indexes/bm25"/d
        if (p/"metadata.json").exists(): continue
        print("Indexing BM25",d,flush=True)
        BM25Index.build(p,corpus(upstream,d))

