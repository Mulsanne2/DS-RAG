import os
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

os.environ['OPENAI_API_KEY'] = ""
answer_llm_gpt = ChatOpenAI(model_name="gpt-4o", temperature=0)
ANSWER_MODEL = answer_llm_gpt
SELECT_NUM = 5

num_queries = [0 for i in range(5)] #check number of each sub-queries in experiments

#< 1.  set directory >#
test_data_path = 'dataset/comparison/comparison_test_graph.json'
vectorspace_path = 'dataset/vectorstore/multisourceqa'
outdir = 'result_log/comparison'
output_file_path = f"{outdir}/test"
os.makedirs(f'{outdir}', exist_ok=True)
total_file = open(f"{output_file_path}_total.txt", 'w')

#< 2. Set LLM and Prompt >#
"""
Prompt for common dataset
"""
# template = """You are an assistant for question-answering tasks.
# Use the following pieces of retrieved context to answer the question.
# Answer using only the provided context. Do not use any background knowledge at all.
# Answer with only “yes” or “no” without adding a comma or period at the end.
# If you don’t know the answer, say “I don’t know.”
# Question: {question}
# Context: {context}
# """

"""
Prompt for comparison dataset
"""
template = """You are an assistant for question-answering tasks.
Use the following pieces of retrieved context to answer the question.
Answer using only the provided context. Do not use any background knowledge at all.
Provide the most accurate answer possible and respond using the full name of the subject mentioned in the question.
Provide only the full name, not a sentence.
If you don’t know the answer, say “I don’t know.”
Question: {question}
Context: {context}
"""

prompt = ChatPromptTemplate.from_template(template)

#< 3. Set Vector Store >#
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
loaded_vectorstore = FAISS.load_local(
    folder_path=vectorspace_path,
    embeddings=embeddings,
    allow_dangerous_deserialization=True
)

#< 4. Set GAT Model >#
import torch
import json
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, GCNConv, GATConv
from tqdm import tqdm
import torch.nn as nn
from model.utils.generate import GenGraph
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"device using : {device}")

def accuracy(ques, doc):
    if ques.dim() == 1:
        ques = ques.unsqueeze(0)
    if doc.dim() == 1:
        doc = doc.unsqueeze(0)
    similarity = F.cosine_similarity(ques, doc)
    return similarity

def GetGraphRepresentation(ques_data, ques_node, option_no, one_hops):
    
    ques_graph = ques_node[one_hops[0]]

    return ques_graph
        


class GATModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads=4):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=8, concat=False))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, hidden_channels, heads=num_heads, concat=False))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GATConv(hidden_channels, out_channels, heads=num_heads, concat=False))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index=edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index=edge_index)
        return x

class GAT(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads=8,):
        super().__init__()
        self.gat_ques = GATModel(in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads)
        self.doc_projection = nn.Linear(3072, out_channels)

    def forward(self, ques_data, doc_data):
        ques_node_embed = self.gat_ques(ques_data.x, ques_data.edge_index.long())

        doc_emb = self.doc_projection(doc_data)

        return ques_node_embed, doc_emb

 
    @torch.no_grad()
    def test(self, ques, doc, GS_opt, one_hops):
        self.eval()  

        with torch.no_grad():  
            ques_data = ques.to(device)
            doc_data = torch.tensor(doc).to(device)
            ques_node, doc_emb = self(ques_data, doc_data)

            #Get Graph Representation
            ques_graph = GetGraphRepresentation(ques_data, ques_node, GS_opt, one_hops)
            acc = accuracy(ques_graph, doc_emb)
            acc = acc[0]

        return acc

# Load Model Parameter

gat1 = GAT(in_channels=3072, hidden_channels=3072, out_channels=1536, num_layers=4, dropout=0, num_heads=4).to(device)
gat1.load_state_dict(torch.load('model/model_weight/comparison_GAT.pth', weights_only=True, map_location='cuda:0'))

#< 4. Set RAG >#
# Set RAG Sytem
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser

retriever_1 = loaded_vectorstore.as_retriever(search_kwargs={"k": 1})
retriever_3 = loaded_vectorstore.as_retriever(search_kwargs={"k": 3})
retriever_4 = loaded_vectorstore.as_retriever(search_kwargs={"k": 4})
retriever_5 = loaded_vectorstore.as_retriever(search_kwargs={"k": 5})
retriever_6 = loaded_vectorstore.as_retriever(search_kwargs={"k": 6})
retriever_7 = loaded_vectorstore.as_retriever(search_kwargs={"k": 7})
retriever_8 = loaded_vectorstore.as_retriever(search_kwargs={"k": 8})
retriever_10 = loaded_vectorstore.as_retriever(search_kwargs={"k": 10})
retriever_20 = loaded_vectorstore.as_retriever(search_kwargs={"k": 20})
rerank_model = HuggingFaceCrossEncoder(model_name = "BAAI/bge-reranker-v2-m3")
compressor_b_1 = CrossEncoderReranker(model=rerank_model, top_n=1)
reranker_b_1 = ContextualCompressionRetriever(
    base_compressor=compressor_b_1, base_retriever=retriever_3
)
compressor_b_3 = CrossEncoderReranker(model=rerank_model, top_n=3)
reranker_b_3 = ContextualCompressionRetriever(
    base_compressor=compressor_b_3, base_retriever=retriever_5
)
compressor_f_3 = FlashrankRerank(top_n=3)
reranker_f_3 = ContextualCompressionRetriever(
    base_compressor=compressor_f_3, base_retriever=retriever_5
)
compressor_b_4 = CrossEncoderReranker(model=rerank_model, top_n=4)
reranker_b_4 = ContextualCompressionRetriever(
    base_compressor=compressor_b_4, base_retriever=retriever_6
)
compressor_b_5 = CrossEncoderReranker(model=rerank_model, top_n=5)
reranker_b_5 = ContextualCompressionRetriever(
    base_compressor=compressor_b_5, base_retriever=retriever_8
)
compressor_b_6 = CrossEncoderReranker(model=rerank_model, top_n=6)
reranker_b_6 = ContextualCompressionRetriever(
    base_compressor=compressor_b_6, base_retriever=retriever_10
)
compressor_b_8 = CrossEncoderReranker(model=rerank_model, top_n=8)
reranker_b_8 = ContextualCompressionRetriever(
    base_compressor=compressor_b_8, base_retriever=retriever_10
)
compressor_b_10 = CrossEncoderReranker(model=rerank_model, top_n=10)
reranker_b_10 = ContextualCompressionRetriever(
    base_compressor=compressor_b_10, base_retriever=retriever_20
)
rag_chain = (
    {"context": RunnablePassthrough(),  "question": RunnablePassthrough()}
    | prompt
    | ANSWER_MODEL
    | StrOutputParser()
)

#< 5. Test Dataset Preprocessing >#
questions = []
questions_d = []
gt_ans_list = []
gt_doc_list = []
relations = []
cores = []
with open(test_data_path, 'r', encoding='utf-8') as f:
    test_data = json.load(f)
    
for entry in test_data:
    relation_temp = []
    question = entry.get("question")
    questions.append(question)
    answer_temp = entry.get("gt_ans")
    gt_ans_list.append(answer_temp)
    supporting_facts_temp = entry.get("gt_doc")
    supporting_facts_temp = [evidence.strip() for evidence in supporting_facts_temp]
    gt_doc_list.append(supporting_facts_temp)
    questions_d_temp = entry.get("question_d")
    questions_d.append(questions_d_temp)
    core_temp = entry.get("core_node")
    cores.append(core_temp)
    rel_temp = entry.get("rener")
    for temp in rel_temp:
        src = temp['subject']
        dst = temp['object']
        relation_temp.append([src,dst])
    relations.append(relation_temp)

#< 8. Evaluation Metrics >#
from anls import anls_score

def list_to_multiline_string(input_list):
    return "\n".join(input_list)

def preprocess_gold_item(gold_item):
    percentage = 1.0
    length = len(gold_item)

    # 앞뒤 15%를 제거한 버전
    start_percentage = 1-percentage
    start_half = start_percentage/2
    end_half = 1-start_half
    
    start_middle = int(length * start_half)
    end_middle = int(length * end_half)
    trimmed_mid = gold_item[start_middle:end_middle]

    
    start_front = int(length * start_percentage)
    trimmed_front = gold_item[start_front:]

    end_back = int(length * percentage)
    trimmed_back = gold_item[:end_back]

    return [trimmed_mid, trimmed_front, trimmed_back]


def has_intersection(a, b):
    a_words = set(a.lower().split())  
    b_words = set(b.lower().split())  
    return len(a_words.intersection(b_words)) > 0

def hit_at_k(retrieved_list, gold_list, k):
    return int(any(
        any(
            preprocessed in retrieved_item
            for retrieved_item in retrieved_list[:k]
            for preprocessed in preprocess_gold_item(gold_item)
        )
        for gold_item in gold_list
    ))

def map_at_k(retrieved_list, gold_list, k):
    relevant_items = 0
    score = 0.0
    for i, retrieved_item in enumerate(retrieved_list[:k], start=1):
        if any(
            preprocessed in retrieved_item
            for gold_item in gold_list
            for preprocessed in preprocess_gold_item(gold_item)
        ):
            relevant_items += 1
            score += relevant_items / i
    return score / min(len(gold_list), k) if gold_list else 0.0

def mrr_at_k(retrieved_list, gold_list, k):
    for i, retrieved_item in enumerate(retrieved_list[:k], start=1):
        if any(
            preprocessed in retrieved_item
            for gold_item in gold_list
            for preprocessed in preprocess_gold_item(gold_item)
        ):
            return 1.0 / i
    return 0.0

def Precision(retrieved_list, gold_list):
    matched = sum(
        any(
            preprocessed in retrieved_item
            for retrieved_item in retrieved_list
            for preprocessed in preprocess_gold_item(gold_item)
        )
        for gold_item in gold_list
    )
    if len(retrieved_list) == 0: 
        return 0
    return matched / len(retrieved_list)

def Coverage(retrieved_list, gold_list):
    matched = sum(
        any(
            preprocessed in retrieved_item
            for retrieved_item in retrieved_list
            for preprocessed in preprocess_gold_item(gold_item)
        )
        for gold_item in gold_list
    )
    if len(gold_list) == 0:
        return 0
    return matched / len(gold_list)

def ExactMatch(string1, string2):
    if string1 is not None and string2 is not None:
        return 1 if string1 == string2 else 0
    return 0

def retrieve_evaluation_metrics(retrieved_list, gold_list):
    hit_3 = hit_at_k(retrieved_list, gold_list, 3)
    hit_5 = hit_at_k(retrieved_list, gold_list, 5)
    map_5 = map_at_k(retrieved_list, gold_list, 5)
    mrr_5 = mrr_at_k(retrieved_list, gold_list, 5)
    precision = Precision(retrieved_list, gold_list)
    coverage = Coverage(retrieved_list, gold_list)

    return {
        "Hit@3": hit_3,
        "Hit@5": hit_5,
        "MAP@5": map_5,
        "MRR@5": mrr_5,
        "PREC" : precision,
        "COV" : coverage,
    }

def accumulate_result(RESULT_SCORE, expt, idx):
    EXPT_HIT3, EXPT_HIT5, EXPT_MAP5, EXPT_MRR5, EXPT_PREC, EXPT_COV, EXPT_ANLS, EXPT_INTER, EXPT_EM = 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0

    for i in range(idx+1):
        EXPT_HIT3+=RESULT_SCORE[i][expt][0]['Hit@3']
        EXPT_HIT5+=RESULT_SCORE[i][expt][0]['Hit@5']
        EXPT_MAP5+=RESULT_SCORE[i][expt][0]['MAP@5']
        EXPT_MRR5+=RESULT_SCORE[i][expt][0]['MRR@5']
        EXPT_PREC+=RESULT_SCORE[i][expt][0]['PREC']
        EXPT_COV+=RESULT_SCORE[i][expt][0]['COV']
        EXPT_ANLS+=RESULT_SCORE[i][expt][1]
        EXPT_INTER+=RESULT_SCORE[i][expt][2]
        EXPT_EM+=RESULT_SCORE[i][expt][3]

    return EXPT_HIT3/(idx+1), EXPT_HIT5/(idx+1), EXPT_MAP5/(idx+1), EXPT_MRR5/(idx+1), EXPT_PREC/(idx+1), EXPT_COV/(idx+1), EXPT_ANLS/(idx+1), EXPT_INTER/(idx+1), EXPT_EM/(idx+1)

def show_result(RESULT_SCORE, expt, idx):
    EXPT_HIT3, EXPT_HIT5, EXPT_MAP5, EXPT_MRR5, EXPT_PREC, EXPT_COV, EXPT_ANLS, EXPT_INTER, EXPT_EM = 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0

    EXPT_HIT3+=RESULT_SCORE[idx][expt][0]['Hit@3']
    EXPT_HIT5+=RESULT_SCORE[idx][expt][0]['Hit@5']
    EXPT_MAP5+=RESULT_SCORE[idx][expt][0]['MAP@5']
    EXPT_MRR5+=RESULT_SCORE[idx][expt][0]['MRR@5']
    EXPT_PREC+=RESULT_SCORE[idx][expt][0]['PREC']
    EXPT_COV+=RESULT_SCORE[idx][expt][0]['COV']
    EXPT_ANLS+=RESULT_SCORE[idx][expt][1]
    EXPT_INTER+=RESULT_SCORE[idx][expt][2]
    EXPT_EM+=RESULT_SCORE[idx][expt][3]

    return EXPT_HIT3, EXPT_HIT5, EXPT_MAP5, EXPT_MRR5, EXPT_PREC, EXPT_COV, EXPT_ANLS, EXPT_INTER, EXPT_EM

def get_descendants(edge_index, start_node):
    # 시작 노드를 탐색하기 위한 큐와 방문 여부를 저장
    visited = set()
    queue = [start_node]
    descendants = []

    # 그래프 탐색
    while queue:
        current = queue.pop(0)  # 큐에서 노드 하나 꺼냄
        if current not in visited:
            visited.add(current)
            descendants.append(current)

            # edge_index에서 current를 source로 가지는 모든 destination 노드를 찾음
            mask = edge_index[0] == current  # source가 current인 부분만 필터링
            children = edge_index[1][mask]

            # 큐에 자식 노드 추가
            queue.extend(children.tolist())

    return descendants

#< 8. RAG Architecture >#
def rag_architecture(i, question, sub_question, retriever, select_num):
    rag_result = []
    all_context = []

    # retrieve all the document from decompostioned question
    for query in sub_question:
        for docs in retriever.invoke(query):
            if docs.page_content not in all_context:
                all_context.append(docs.page_content)

    answer = rag_chain.invoke({"context": all_context, "question": question})

    rag_result.append(answer)
    rag_result.append(all_context)
    return rag_result

def rag_architecture_Rerank(i, question, documents, select_num):
    rag_result = []
    sorted_documents= []

    LEN = len(documents)

    faiss_temp = FAISS.from_texts(documents,embedding=embeddings)
    RET = faiss_temp.as_retriever(search_kwargs={"k":LEN})
    COMP = CrossEncoderReranker(model=rerank_model, top_n=select_num)
    RERANK = ContextualCompressionRetriever(
    base_compressor=COMP, base_retriever=RET
    )
    for docs in RERANK.invoke(question):
        sorted_documents.append(docs.page_content)
    
    answer = rag_chain.invoke({"context": sorted_documents, "question": question})
    rag_result.append(answer)
    rag_result.append(sorted_documents)
    return rag_result

def rag_architecture_GS(i, question, ques_graph, documents, Doc_emb, gat, GS_opt, select_num):
    rag_result = []

    Scores = []
    for d in Doc_emb:
        Scores.append(gat.test(ques_graph, d, GS_opt, []))

    sorted_documents = [d for Score, d in sorted(zip(Scores,documents), reverse=True)]
    if len(sorted_documents)>select_num:
        sorted_documents = sorted_documents[:select_num]

    answer = rag_chain.invoke({"context": sorted_documents, "question": question})
    rag_result.append(answer)
    rag_result.append(sorted_documents)
    return rag_result

#Select only subgraph or sub node
def rag_architecture_GS_S(i, question, ques_graph, documents, Doc_emb, gat, GS_opt, select_num):
    rag_result = []
    sorted_documents = []

    core_node = ques_graph.core_node
    #만약 core node가 없는 경우
    if core_node == -1:
        return rag_architecture_GS(i, question, ques_graph, documents, Doc_emb, gat, GS_opt, select_num)

    one_hops = ques_graph.core_onehop

    if i == 1:
        num_queries[3] += len(one_hops)
    elif i == 2:
        num_queries[4] += len(one_hops)

    #모든 onehop-core에 대해 Node에 대해 가장 잘 맞는 한가지를 찾는다.
    if GS_opt==5:
        #모든 노드에 대해서 반복
        for one_hop in one_hops:
            one_hop = [one_hop.item()]

            Scores = []
            for d in Doc_emb:
                Scores.append(gat.test(ques_graph, d, GS_opt,one_hop))

            sorted_documents_temp = [d for Score, d in sorted(zip(Scores,documents), reverse=True)]
            sorted_documents.append(sorted_documents_temp[0])
    
    elif GS_opt==6:
        #모든 노드에 대해서 반복
        for one_hop in one_hops:
            one_hop = get_descendants(ques_graph.edge_index, one_hop.item())

            Scores = []
            for d in Doc_emb:
                Scores.append(gat.test(ques_graph, d, GS_opt,one_hop))

            sorted_documents_temp = [d for Score, d in sorted(zip(Scores,documents), reverse=True)]
            sorted_documents.append(sorted_documents_temp[0])

    answer = rag_chain.invoke({"context": sorted_documents, "question": question})
    rag_result.append(answer)
    rag_result.append(sorted_documents)
    return rag_result

print("Start RAG")

out_files = [open(f"{output_file_path}_{i+1}.txt", 'w') for i in range(5)]
RESULT_SCORE =[]
for idx, QUESTION in enumerate(tqdm(questions)):
    RESULT = []

    RESULT_SCORE.append([])
    ANSWER = gt_ans_list[idx]
    ANS_DOCUMENT = gt_doc_list[idx]
    DECOMP_QUERIES = questions_d[idx]
    RELATION = relations[idx]
    CORE = cores[idx]

    #1. VS(10) - Reranking(3,4,5)
    compressor_n = CrossEncoderReranker(model=rerank_model, top_n=SELECT_NUM)
    reranker_n = ContextualCompressionRetriever(
        base_compressor=compressor_n, base_retriever=retriever_10
    )
    RESULT.append(rag_architecture(idx, QUESTION, [QUESTION], reranker_n, SELECT_NUM))
    num_queries[0] += SELECT_NUM

    #2. QD - VS(3) -Reranking(1)
    compressor_1 = CrossEncoderReranker(model=rerank_model, top_n=1)
    reranker_1 = ContextualCompressionRetriever(
        base_compressor=compressor_1, base_retriever=retriever_3
    )
    RESULT.append(rag_architecture(idx, QUESTION, DECOMP_QUERIES, reranker_1, SELECT_NUM))
    num_queries[1] += len(DECOMP_QUERIES)
    
    
    compressor_3 = CrossEncoderReranker(model=rerank_model, top_n=3)
    reranker_3 = ContextualCompressionRetriever(
        base_compressor=compressor_3, base_retriever=retriever_7
    )
    compressor_4 = CrossEncoderReranker(model=rerank_model, top_n=4)
    reranker_4 = ContextualCompressionRetriever(
        base_compressor=compressor_4, base_retriever=retriever_7
    )
    compressor_5 = CrossEncoderReranker(model=rerank_model, top_n=5)
    reranker_5 = ContextualCompressionRetriever(
        base_compressor=compressor_5, base_retriever=retriever_7
    )

    retrieve_context = []
    rerank_context = []
    #retrieve 3 on each sub-queries
    for query in DECOMP_QUERIES:
        for docs in retriever_3.invoke(query):
            if docs.page_content not in retrieve_context:
                retrieve_context.append(docs.page_content)
    #rerank 3 on each sub-queries
    for query in DECOMP_QUERIES:
        for docs in reranker_3.invoke(query):
            if docs.page_content not in rerank_context:
                rerank_context.append(docs.page_content)

    documents = retrieve_context
    reranking7_document= []
    LEN = len(documents)
    faiss_temp = FAISS.from_texts(documents,embedding=embeddings)
    RET = faiss_temp.as_retriever(search_kwargs={"k":LEN})
    COMP = CrossEncoderReranker(model=rerank_model, top_n=7)
    RERANK = ContextualCompressionRetriever(
    base_compressor=COMP, base_retriever=RET
    )
    for docs in RERANK.invoke(QUESTION):
        reranking7_document.append(docs.page_content)

    Ques_graph, Doc_retrieve_emb, Reranking7_emb= GenGraph(RELATION, CORE, retrieve_context, reranking7_document)

    #3. QD-VS(3)-Reranking7)-Reranking(3,4,5)
    RESULT.append(rag_architecture_Rerank(idx, QUESTION, retrieve_context, SELECT_NUM))
    num_queries[2] += SELECT_NUM

    #4. QD-VS(3)-Reranking(7)-GS
    RESULT.append(rag_architecture_GS_S(1, QUESTION, Ques_graph, reranking7_document, Reranking7_emb, gat1, 5, SELECT_NUM))

    #5. QD-VS(3)-GS
    RESULT.append(rag_architecture_GS_S(2, QUESTION, Ques_graph, retrieve_context, Doc_retrieve_emb, gat1, 5, SELECT_NUM))

    QUESTIONS = []
    ANSWERS = []
    CONTEXTS = []
    GROUNDTRUTHS = []
    for i in range(5):
        QUESTIONS.append(QUESTION)
        ANSWERS.append(RESULT[i][0])
        CONTEXTS.append(RESULT[i][1])
        GROUNDTRUTHS.append(gt_ans_list[idx])

    data = {
        "question": QUESTIONS,
        "answer": ANSWERS,
        "contexts": CONTEXTS,
        "ground_truth": GROUNDTRUTHS
    }
    
    for i in range(5):
        Result = retrieve_evaluation_metrics(RESULT[i][1], gt_doc_list[idx])
        Anls = anls_score(prediction=RESULT[i][0], gold_labels=[gt_ans_list[idx]])
        intersection = int(has_intersection(RESULT[i][0], gt_ans_list[idx]))
        EM = ExactMatch(RESULT[i][0],gt_ans_list[idx])
        RESULT_SCORE[idx].append([Result,Anls,intersection,EM])

    for i in range(5):
        out_files[i].write(f"Question ID : {idx}\n")
        hit3, hit5, map5, mrr5, prec, cov, anls, intersec, em = show_result(RESULT_SCORE,i, idx)
        out_files[i].write(f"PRECISION: {prec:.4f}\t")
        out_files[i].write(f"COVERAGE: {cov:.4f}\n")
        out_files[i].write(f"ANLS Score: {anls:.4f}\t")
        out_files[i].write(f"EM: {em:.4f}\n")

        out_files[i].write(f"Question: {QUESTION}\n")
        out_files[i].write(f"Ground Truth: [{gt_ans_list[idx]}]\n")
        out_files[i].write(f"Answer: {RESULT[i][0]}\n")
        out_files[i].write(f"GT Context:\n{list_to_multiline_string(gt_doc_list[idx])}\n")
        out_files[i].write(f"Context:\n{list_to_multiline_string(RESULT[i][1])}\n")
        out_files[i].write("------------------------------------------------------------\n\n")

    #Show Accumulate Score
    for i in range(5):
        print(f'Expt Number : {i+1}  \tQuestion ID: {idx}')
        hit3, hit5, map5, mrr5, prec, cov, anls, intersec, em = show_result(RESULT_SCORE,i, idx)
        print(f'NOW: Hit@3: {hit3:.4f}\tHit@5: {hit5:.4f}\tMAP@5: {map5:.4f}\tMRR@5: {mrr5:.4f}\tPRECISION: {prec:.4f}\tCOVERAGE: {cov:.4f}\tANLS: {anls:.4f}\tIntersec: {intersec:.4f}\tExactMatch: {em:.4f}')
        hit3, hit5, map5, mrr5, prec, cov, anls, intersec, em = accumulate_result(RESULT_SCORE,i, idx)
        print(f'TOTAL: Hit@3: {hit3:.4f}\tHit@5: {hit5:.4f}\tMAP@5: {map5:.4f}\tMRR@5: {mrr5:.4f}\tPRECISION: {prec:.4f}\tCOVERAGE: {cov:.4f}\tANLS: {anls:.4f}\tIntersec: {intersec:.4f}\tExactMatch: {em:.4f}')

for i in range(5):
    out_files[i].write("*******************************************************************\n")
    out_files[i].write(f"Experiment Number : {i+1}\n")
    out_files[i].write(f"Mean Number of Query : {num_queries[i]/len(questions)}\n")
    hit3, hit5, map5, mrr5, prec, cov, anls, intersec, em = accumulate_result(RESULT_SCORE,i, len(questions)-1)
    out_files[i].write(f"PRECISION: {prec:.4f}\n")
    out_files[i].write(f"COVERAGE: {cov:.4f}\n")
    out_files[i].write(f"ANLS Score: {anls:.4f}\n")
    out_files[i].write(f"Exact Match: {em:.4f}\n")
    out_files[i].write("*******************************************************************\n")
    total_file.write(f"expt{i+1} : PRECISION: {prec:.4f}\tCOVERAGE: {cov:.4f}\tANLS: {anls:.4f}\tIntersec: {intersec:.4f}\tExactMatch: {em:.4f}\tDocument Num: {num_queries[i]/len(questions)}\n")

    print("*******************************************************************")
    print(f"Experiment Number : {i+1}")
    print(f"Mean Number of Query : {num_queries[i]/len(questions)}")
    hit3, hit5, map5, mrr5, prec, cov, anls, intersec, em = accumulate_result(RESULT_SCORE,i, len(questions)-1)
    print(f"Hit@3: {hit3:.4f}")
    print(f"Hit@5: {hit5:.4f}")
    print(f"MAP@5: {map5:.4f}")
    print(f"MRR@5: {mrr5:.4f}")
    print(f"PRECISION: {prec:.4f}")
    print(f"COVERAGE: {cov:.4f}")
    print(f"ANLS Score: {anls:.4f}")
    print(f"Exact match : {em:.4f}")
    print("*******************************************************************")

for f in out_files:
    f.close()