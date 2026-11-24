"""Local neural inference; no hosted API keys or paid services required."""
from __future__ import annotations
import gc
import json
import os
import time
from pathlib import Path
import numpy as np
from scipy import sparse
from .data import corpus, select_tasks
from .io import read_jsonl, write_json
from .queries import latest, rewrite_messages, cache_key

MODELS = {"splade":"naver/splade-cocondenser-ensembledistil", "dense":"BAAI/bge-small-en-v1.5"}
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

def environment(root):
    os.environ.setdefault("HF_HOME",str(Path(root).resolve()/"cache/huggingface"))
    os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
    os.environ.setdefault("HF_HUB_DISABLE_XET","1")

def device():
    import torch
    torch.set_num_threads(4)
    return "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

def revision(root, model):
    from huggingface_hub import HfApi
    path=Path(root)/"cache/model_revisions.json"
    versions=json.loads(path.read_text()) if path.exists() else {}
    if model not in versions:
        versions[model]=HfApi().model_info(model).sha
        write_json(path,versions)
    return versions[model]

class Encoder:
    def __init__(self,root,backend,rev=None,max_length=256):
        environment(root)
        import torch
        from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
        self.backend=backend;self.model_id=MODELS[backend]
        self.revision=rev or revision(root,self.model_id)
        self.device=device();self.max_length=max_length
        self.tokenizer=AutoTokenizer.from_pretrained(self.model_id,revision=self.revision)
        cls=AutoModelForMaskedLM if backend=="splade" else AutoModel
        self.model=cls.from_pretrained(self.model_id,revision=self.revision).to(self.device).eval()

    def encode(self,texts,query=False):
        import torch
        if query and self.backend=="dense": texts=[QUERY_PREFIX+t for t in texts]
        tokens=self.tokenizer(texts,padding=True,truncation=True,max_length=self.max_length,return_tensors="pt").to(self.device)
        with torch.inference_mode():
            outputs=self.model(**tokens)
            if self.backend=="splade":
                values=torch.log1p(torch.relu(outputs.logits))*tokens["attention_mask"].unsqueeze(-1)
                vectors=values.max(dim=1).values
                # Explicit document pruning for the 16 GB local resource budget.
                if not query:
                    val,idx=torch.topk(vectors,k=128,dim=1)
                    rows=np.repeat(np.arange(len(texts)),128)
                    return sparse.csr_matrix((val.cpu().numpy().ravel(),(rows,idx.cpu().numpy().ravel())),shape=(len(texts),vectors.shape[1]))
                return sparse.csr_matrix(vectors.cpu().numpy())
            vectors=torch.nn.functional.normalize(outputs.last_hidden_state[:,0],p=2,dim=1)
            return vectors.cpu().numpy()

def build_neural(root,upstream,args):
    root=Path(root); encoder=Encoder(root,args.backend,max_length=args.max_length)
    for domain in args.domains:
        path=root/"indexes"/args.backend/domain
        if (path/"metadata.json").exists(): continue
        path.mkdir(parents=True,exist_ok=True)
        docs=list(corpus(upstream,domain))
        if args.limit: docs=docs[:args.limit]
        parts=[]; start=time.perf_counter()
        # Resumable shards avoid losing a long indexing job on interruption.
        for i in range(0,len(docs),args.batch_size):
            shard=path/"shards"/(f"{i:08d}.npz" if args.backend=="splade" else f"{i:08d}.npy")
            if not shard.exists():
                shard.parent.mkdir(exist_ok=True)
                value=encoder.encode([d["text"] for d in docs[i:i+args.batch_size]])
                if args.backend=="splade": sparse.save_npz(shard,value)
                else: np.save(shard,value)
            parts.append(shard)
            if i%(args.batch_size*100)==0:
                print("Index",args.backend,domain,i,"/",len(docs),"seconds",round(time.perf_counter()-start),flush=True)
        if args.backend=="splade":
            matrix=sparse.vstack([sparse.load_npz(s) for s in parts],format="csr")
            sparse.save_npz(path/"vectors.npz",matrix)
        else:
            matrix=np.concatenate([np.load(s) for s in parts]);np.save(path/"vectors.npy",matrix)
        write_json(path/"ids.json",[d["id"] for d in docs])
        write_json(path/"metadata.json",{"backend":args.backend,"model":encoder.model_id,"revision":encoder.revision,
                   "documents":len(docs),"max_length":args.max_length,"batch_size":args.batch_size,
                   "document_top_terms":128 if args.backend=="splade" else None,
                   "seconds":time.perf_counter()-start,"limited_corpus":bool(args.limit),"device":encoder.device})
        for s in parts: s.unlink()
        del matrix;gc.collect()

class NeuralIndex:
    def __init__(self,path):
        self.path=Path(path);meta=json.loads((self.path/"metadata.json").read_text())
        if meta["limited_corpus"]: raise ValueError("Pilot index cannot be used for benchmark evaluation")
        self.backend=meta["backend"];self.ids=json.loads((self.path/"ids.json").read_text())
        self.encoder=Encoder(self.path.parents[2],self.backend,meta["revision"],meta["max_length"])
        self.matrix=sparse.load_npz(self.path/"vectors.npz").tocsr() if self.backend=="splade" else np.load(self.path/"vectors.npy",mmap_mode="r")
    def search(self,query,k=100):
        vector=self.encoder.encode([query],query=True)
        scores=(self.matrix@vector.T).toarray().ravel() if self.backend=="splade" else (self.matrix@vector[0])
        k=min(k,len(self.ids));indices=np.argpartition(-scores,k-1)[:k]
        return sorted([(self.ids[int(i)],float(scores[i])) for i in indices],key=lambda x:(-x[1],x[0]))

class Reranker:
    def __init__(self,root,model):
        environment(root)
        from sentence_transformers import CrossEncoder
        self.model_id=model;self.revision=revision(root,model)
        self.model=CrossEncoder(model,revision=self.revision,device=device(),max_length=384)
    def rank(self,query,ranking,texts):
        scores=self.model.predict([(query,texts[p]) for p,_ in ranking],batch_size=32,show_progress_bar=False)
        return sorted([(p,float(s)) for (p,_),s in zip(ranking,scores)],key=lambda x:(-x[1],x[0]))

def generate_rewrites(root,args):
    environment(root)
    modelrev=revision(root,args.model)
    if args.model.startswith("mlx-community/"):
        from mlx_lm import load, generate
        from mlx_lm.sample_utils import make_sampler
        from huggingface_hub import snapshot_download
        path=snapshot_download(args.model,revision=modelrev,allow_patterns=["*.json","*.safetensors","*.model","*.txt","*.jinja"])
        model,tokenizer=load(path)
        def infer(messages):
            prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            return generate(model,tokenizer,prompt=prompt,max_tokens=128,sampler=make_sampler(temp=0),verbose=False)
    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        target=device()
        tokenizer=AutoTokenizer.from_pretrained(args.model,revision=modelrev)
        model=AutoModelForCausalLM.from_pretrained(args.model,revision=modelrev).to(target).eval()
        def infer(messages):
            prompt=tokenizer.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
            tokens=tokenizer(prompt,return_tensors="pt").to(target)
            with torch.inference_mode():
                result=model.generate(**tokens,max_new_tokens=128,do_sample=False,pad_token_id=tokenizer.eos_token_id)
            return tokenizer.decode(result[0,tokens['input_ids'].shape[1]:],skip_special_tokens=True)
    output=Path(root)/args.output;output.parent.mkdir(parents=True,exist_ok=True)
    existing={r["cache_key"] for r in read_jsonl(output)} if output.exists() else set()
    tasks=select_tasks(root,args.dataset,args.split,args.limit)
    with output.open("a") as f:
        for i,t in enumerate(tasks):
            key=cache_key(t,args.model,modelrev,args.history_mode)
            if key in existing: continue
            start=time.perf_counter()
            answer=infer(rewrite_messages(t,args.history_mode)).strip()
            answer=answer.removeprefix("Standalone question:").strip().strip('"')
            if not answer: answer=latest(t)
            row={"task_id":t["task_id"],"cache_key":key,"rewrite":answer,"model":args.model,"revision":modelrev,
                 "history_mode":args.history_mode,"seconds":time.perf_counter()-start}
            f.write(json.dumps(row,ensure_ascii=False)+"\n");f.flush()
            if i%10==0: print("Rewrite",i,"/",len(tasks),round(row["seconds"],2),flush=True)
