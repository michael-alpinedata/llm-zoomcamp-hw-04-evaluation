import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Homework - module 4 - Evaluation
    """)
    return


@app.cell
def _():
    # marimo uses pure python so no direct shell (=/= jupyter notebook)
    import subprocess

    def sh(cmd: str):
        subprocess.run(cmd, shell=True, check=True)

    return (sh,)


@app.cell
def _(sh):
    # from hw 2:
    sh("uv add gitsource")

    # for hw 4:
    sh("uv add google-genai pydantic python-dotenv pandas")
    return


@app.cell
def _():
    # Configure Client
    from dotenv import load_dotenv
    from google.genai import Client
    import os

    load_dotenv()

    google_client = Client(api_key=os.environ['GEMINI_API_KEY'])
    MODEL_NAME = os.environ['MODEL_NAME']
    return (google_client,)


@app.cell
def _():
    # Load documents from source (list of pages (1 page = 1 json object))
    from gitsource import GithubRepositoryDataReader

    reader = GithubRepositoryDataReader(
        repo_owner="DataTalksClub",
        repo_name="llm-zoomcamp",
        commit_id="8c1834d",
        allowed_extensions={"md"},
        filename_filter=lambda path: "/lessons/" in path,
    )
    documents = [file.parse() for file in reader.read()]
    return (documents,)


@app.cell
def _(documents):
    # 1 page (json) = 1 content, 1 filename 
    documents[0]
    return


@app.cell
def _(documents):
    len(documents)
    return


@app.cell
def _():
    # # load helpers for RAG and evaluation
    # sh("""
    # PREFIX=https://raw.githubusercontent.com/DataTalksClub/llm-zoomcamp/main
    # wget ${PREFIX}/01-agentic-rag/code/rag_helper.py
    # wget ${PREFIX}/04-evaluation/code/evaluation_utils.py
    # """)
    return


@app.cell
def _():
    # Prepare specific user prompt to generate Ground Truth answers (LLM-generated)
    data_gen_instructions = """
    You emulate a student who is taking our LLM course.
    You are given one lesson page from the course.
    Formulate 5 questions this student might ask that are answered by this page.

    Rules:
    - The page should contain the answer to each question.
    - Make the questions complete and not too short.
    - Use as few words as possible from the page; don't copy its phrasing.
    - The questions shoulder resemble how people actually ask things online:
      not too formal, not too short, not too long.
    - Ask about the content of the lesson, not about its formatting or filename.
    """.strip()
    return (data_gen_instructions,)


@app.cell
def _(documents):
    # stringify the list of json documents (list of pages, each page is a json object)
    import json

    user_prompts = []
    for doc in documents[0:3]: 
      user_prompts.append(json.dumps(doc))
    return (user_prompts,)


@app.cell
def _(user_prompts):
    user_prompts
    return


@app.cell
def _():
    # prepare the structured output formatter
    from pydantic import BaseModel

    class Questions(BaseModel):
        questions: list[str]

    return (Questions,)


@app.cell
def _():
    # import the llm_structured utils
    from evaluation_utils import llm_structured

    return (llm_structured,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Question 1 - Generating questions
    """)
    return


@app.cell
def _(
    Questions,
    data_gen_instructions,
    google_client,
    llm_structured,
    user_prompts,
):

    input_tokens = []

    for user_prompt in user_prompts:
        result, usage = llm_structured(
            google_client,
            data_gen_instructions,
            user_prompt,
            Questions
        )

        input_tokens.append(usage.input_tokens)
    return (input_tokens,)


@app.cell
def _(input_tokens):
    print(input_tokens)
    return


@app.cell
def _(input_tokens):
    print(sum(input_tokens)/3.)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Question 2 - First result with text search
    """)
    return


@app.cell
def _():
    # load already prepared ground_truth questions for the 72 pages in a df
    import pandas as pd

    ground_truth= pd.read_csv('ground-truth.csv').to_dict(orient='records')
    return (ground_truth,)


@app.cell
def _():
    # ground_truth
    return


@app.cell
def _(documents):
    # Chunk the documents as in module 2 - Vector search
    from gitsource import chunk_documents

    chunks = chunk_documents(documents, size=2000, step=1000)
    return (chunks,)


@app.cell
def _(chunks):
    len(chunks)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Build search tools (from hw 2)
    """)
    return


@app.cell
def _(chunks):
    # text_search:

    from minsearch import Index

    text_index = Index(text_fields=["content"],
                   keyword_fields=["filename"])
    text_index.fit(chunks)
    return (text_index,)


@app.cell
def _(text_index):
    def text_search(query, num_results=5):
      return text_index.search(query,num_results=num_results)


    return (text_search,)


@app.cell
def _(ground_truth):
    # start question 2

    q = ground_truth[0]["question"]
    return (q,)


@app.cell
def _(q, text_search):
    # run text_search on it
    text_search(q)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Question 3 - First result with vector search
    """)
    return


@app.cell
def _():
    # For vector search, first download ONNX version of Xenova/all-MiniLM-L6-v2
    #  for the embedder
    # sh("uv run python embed/download.py")
    return


@app.cell
def _(chunks):
    # embed chunks for vector search
    import numpy as np
    from embed.embedder import Embedder

    embed = Embedder()

    X = []

    texts = [chunk['content'] for chunk in chunks]

    X = embed.encode_batch(texts)

    # convert to numpy array
    X = np.array(X)
    return X, embed


@app.cell
def _(X, chunks):
    # build vector_search
    from minsearch import VectorSearch

    vindex = VectorSearch(keyword_fields=["filename"])
    vindex.fit(X, chunks)
    return (vindex,)


@app.cell
def _(embed, vindex):
    def vector_search(query, num_results=5):
      query_vector = embed.encode(query)
      return vindex.search(query_vector, num_results=num_results)

    return (vector_search,)


@app.cell
def _(q, vector_search):
    vector_search(q)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Question 4 - Evaluating text search
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Compute the hit rate the relevance function then
    """)
    return


@app.function
# for one question of the ground_truth
def compute_relevance(q, search_function):
    doc_id = q["filename"]
    results = search_function(query=q["question"])

    relevance = []
    for d in results:
        # print(f"d = {d}")
        relevance.append(int(d["filename"] == doc_id))

    return relevance


@app.cell
def _(ground_truth, text_search):
    compute_relevance(ground_truth[0],text_search)
    return


@app.cell
def _():
    # then for all questions of the ground_truth
    from tqdm.auto import tqdm

    def compute_relevance_total(ground_truth, search_function):
        relevance_total = []

        for q in tqdm(ground_truth):
            relevance = compute_relevance(q, search_function)
            relevance_total.append(relevance)

        return relevance_total

    return (compute_relevance_total,)


@app.function
# define then hit_rate as the fraction of questions have (at least(chunks)) one
#  good result in their top 5 result
# ie: the document on which the ground truth question has been built
# is found in the top 5 results

def hit_rate(relevance):
    cnt = 0

    for line in relevance:
        if 1 in line:
            cnt = cnt + 1

    return cnt / len(relevance)


@app.cell
def _(compute_relevance_total, ground_truth, text_search):
    text_relevance_total = compute_relevance_total(ground_truth, text_search)
    return (text_relevance_total,)


@app.cell
def _(text_relevance_total):
    text_hit_rate=hit_rate(text_relevance_total)
    text_hit_rate
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Question 5 - Evaluating vector search
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Compute the Mean Reciprocal Rank (MRR)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Hit Rate tells us if we found the right document, but not where it was.

    MRR also considers the position.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    MRR calculation method:

    ```python
    total_score = 0.0

    for line in example:
        for rank in range(len(line)):
            if line[rank] == 1:
                total_score = total_score + 1 / (rank + 1)
                break

    total_score
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The total score is `12.333333333333334`. We use `rank + 1` because
    Python counts positions from zero. The first position should score
    `1/1`, and without the `+ 1` we'd divide by zero.

    Divide it by the number of queries:

    ```python
    total_score / len(example)
    # 5 0.822
    ```
    """)
    return


@app.function
# MRR def
def mrr(relevance):
    total_score = 0.0

    for line in relevance:
        for rank in range(len(line)):
            if line[rank] == 1:
                total_score = total_score + 1 / (rank + 1)
                break

    return total_score / len(relevance)


@app.cell
def _(compute_relevance_total, ground_truth, vector_search):
    vector_relevance_total = compute_relevance_total(ground_truth, vector_search)
    return (vector_relevance_total,)


@app.cell
def _():
    # vector_relevance_total
    return


@app.cell
def _(vector_relevance_total):

    mrr_vector_search=mrr(vector_relevance_total)
    mrr_vector_search
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Question 6 - Tuning hybrid search
    """)
    return


@app.function
# build hybrid search
def rrf(result_lists, k=60, num_results=5):
    scores = {}
    docs = {}

    for results in result_lists:
        for rank, doc in enumerate(results):
            key = (doc["filename"], doc["start"])
            scores[key] = scores.get(key, 0) + 1 / (k + rank)
            docs[key] = doc

    ranked = sorted(scores, key=scores.get, reverse=True)
    return [docs[key] for key in ranked[:num_results]]


@app.cell
def _(text_search, vector_search):
    # Then define `hybrid_search` on top of it:
    def hybrid_search(query, k=200):
        text_results = text_search(query, num_results=10)
        vector_results = vector_search(query, num_results=10)
        return rrf([text_results, vector_results], k=k)


    return (hybrid_search,)


@app.cell
def _(compute_relevance_total, ground_truth, hybrid_search):
    hybrid_relevance_total = compute_relevance_total(
        ground_truth, 
        hybrid_search)
    return (hybrid_relevance_total,)


@app.cell
def _(hybrid_relevance_total):

    mrr_hybrid_search=mrr(hybrid_relevance_total)
    mrr_hybrid_search
    return


if __name__ == "__main__":
    app.run()
