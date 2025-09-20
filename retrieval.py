import argparse
from sentence_transformers import SentenceTransformer
from sentence_transformers import util
import csv
import sys
import time
import torch
from tqdm import tqdm
import json

csv.field_size_limit(sys.maxsize)

# By default using "codesage/codesage-base-v2", /home/avisingh/retreival
# Command: retrieval.py --input_query_csv instruction.csv --input_corpus_csv retreival_docs_wo_solutions.csv

parser = argparse.ArgumentParser()
parser.add_argument('--model', default="codesage/codesage-base-v2", help="sentence-transformer model for the retriever LM.")
parser.add_argument('--input_corpus_csv', default="./corpus.csv", help="Input CSV file to use as corpus for retriever.")
parser.add_argument('--input_query_json', default="./Python_Queries/OriginalPythonQueries.json", help="Input CSV file to use as queries for retriever.")
parser.add_argument('--top_k', default=3, help="Number of documents to retrieve.")
parser.add_argument('--output_csv', default="./output.csv", help="CSV file to save the output in.")
parser = parser.parse_args()

start_time = time.perf_counter()

embedder = SentenceTransformer(parser.model, trust_remote_code=True, device='cuda', model_kwargs={"torch_dtype": torch.float16})

print("Loaded embedding model...")

corpus = []
with open(parser.input_corpus_csv, "r", newline='') as f:
    reader = csv.DictReader(f)
    for row in tqdm(iterable=reader,total=477591):
        corpus.append(row["Document"])

# 10k from the  50k from instruction.csv

corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True, show_progress_bar=True,batch_size=512)

print("Embedded corpus...")

queries = []
results = []
classes = []
with open(parser.input_query_json, "r", newline='') as f:
    json_reader = json.load(f)
    for i in range(1,4310):
        if "APIs" not in json_reader[str(i)]:
            continue
        queries.append(json_reader[str(i)]["OriginalQuery"])
        results.append("; ".join(json_reader[str(i)]["APIs"]))
        classes.append("; ".join(json_reader[str(i)]["APIClasses"]))

all_res = []
query_embeddings = embedder.encode(queries, convert_to_tensor=True, show_progress_bar=True,batch_size=512)
print("Embedded queries...")

print("Performing semantic search...")
for i, (query,query_embedding) in enumerate(zip(queries, query_embeddings)):
    if (i%1000==0):
        print("Finished Query: " + str(i))
    hits = util.semantic_search(query_embedding, corpus_embeddings, top_k=int(parser.top_k))
    hits = hits[0]      
    
    res = {"id":i, "Instruction":query, "Results":results[i], "APIClasses":classes[i]}
    
    for j, hit in enumerate(hits):
        col_name = "rank_" + str(j)
        res[col_name] = corpus[hit['corpus_id']]
    all_res.append(res)

print("Performed semantic search...")

fieldname = ["id","Instruction","Results","APIClasses"]
for i, hit in enumerate(hits):
    col_name = "rank_" + str(i)
    fieldname.append(col_name)

with open(parser.output_csv,"w") as file:
    writer = csv.DictWriter(file,fieldnames=fieldname)
    writer.writeheader()
    writer.writerows(all_res)

print(f"CSV File with Retrievals written to {parser.output_csv}")

end_time = time.perf_counter()

# 4. Calculate the elapsed time
duration = end_time - start_time

print(f"The code took {duration:.4f} seconds to run.")