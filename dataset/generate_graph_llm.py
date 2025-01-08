import json
import networkx as nx
import re
from tqdm import tqdm
from collections import deque

from utils.test_utils import visualize_graph
from utils.node_utils import get_entities_load, get_entities_llm
from utils.utils import compose_entity_dict, convert_rener, find_best_match, link_apart_node, merge_quoted_entities, compose_entity_dict_v2, compose_graph, is_interrogative_word
from utils.edge_utils import relation_extract_between_entities
from utils.triple_store import TripleStore
from collections import defaultdict


# question, question_d, gt_doc, gt_ans가 존재하는 decomposed query의 json 파일
decomposed_query_file_path = 'comparison/comparison_test_qd.json'

# 그래프 및 core_node, 그 외 정보들을 저장하는 json 파일 경로
save_path = "comparison/comparison_test_graph.json"



with open(decomposed_query_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

questions = []
ground_truths = []
evidence_list = []
decomposed_queries = []
for entry in data:
    question = entry.get("question", "")
    supporting_facts = entry.get("gt_doc", [])
    answer = entry.get("gt_ans", "")
    
    decom_query = entry.get("question_d", [])
    decomposed_queries.append(decom_query)
    questions.append(question)
    evidence_list.append(supporting_facts)
    ground_truths.append(answer)


loaded_facts_list = evidence_list

data = []
for idx_o, q in enumerate(tqdm(questions)):
    entities_set = set()   
    tripleset_stores = []
    each_entity_dict_list = []

    for idx, decom_q in enumerate(decomposed_queries[idx_o]):

        result = get_entities_llm(decom_q)  # Assume this returns a \t-separated string
        entities_set.update(result.split("\t"))  # Split and add to list

    entities = '\t'.join(entities_set)
    for idx, decom_q in enumerate(decomposed_queries[idx_o]):
        
        # 1단계 node 생성
        # entities = get_entities_llm(decom_q) #\t로 연결된 list 형태
        entity_dict = compose_entity_dict_v2(entities, decom_q) 
        each_entity_dict_list.append(entity_dict)
        #eN: 문장 내에 존재하는 entity을 이용해서 각 sub query에 해당하는 dict 형태로 변경
        # print(each_entity_dict_list)

        # 2단계 edge 생성 및 3단계 sub-graph 생성을 위한 tripleset 생성 (e1 || r1 || e3 형식)
        relations_dict, relations = relation_extract_between_entities(decom_q, entity_dict) #entity가 아닌 사이의 relation을 추출, 
        # print(relations_dict)

        # 각 subquery마다 triple set store 생성하고 저장하기
        store = TripleStore()

        for line in relations.split("\n"):
            try:
                e1, r, e2 = line.split(" || ")
                e1 = e1.strip()
                r = r.strip()
                e2 = e2.strip()

                e1_num = int(re.search(r'\d+', e1).group())  # e1에서 숫자 추출
                r_num = int(re.search(r'\d+', r).group())    # r에서 숫자 추출
                e2_num = int(re.search(r'\d+', e2).group())  # e2에서 숫자 추출

                # 여기서 만약에 숫자 순서가 다르면 맞춰주기 (node 순서 - 방향성 정렬)
                if e2_num < e1_num:
                    e1_num, e2_num = e2_num, e1_num

                store.add_triple(idx, entity_dict[e1_num], e1_num, entity_dict[e2_num], e2_num, relations_dict[r_num], r_num)

            except Exception as e:
                # 잘못된 형식이거나 매핑되지 않는 값이 있는 경우 건너뜀
                continue
        

        # triple set이 생성되지 않은 노드 있으면 연결하기 위한 과정 (3-2, 3-3)
        # entity_dict에 존재하는 value 중, store의 subject, object 어디에도 포함되지 않는다면 앞(또는 뒤) entity와 연결되는 tripleset 추가
        entity_ids = set()
        entity_ids.update(store.get_subject_ids())
        entity_ids.update(store.get_object_ids())

        # question_words = {'what', 'who', 'when', 'where', 'why', 'how', 'which'}
        # # Check for keys in entity_dict not present in subject_ids or object_ids
        # missing_ids = [key for key in entity_dict.keys() if key not in entity_ids and 
        #        entity_dict[key].lower() not in question_words]

        missing_ids = [key for key in entity_dict.keys() if key not in entity_ids]
        
        for mid in missing_ids:
            
            if mid == 0: #앞에 entity가 존재하지 않는 경우, 뒤와 연결
                if 1 in entity_dict:
                    store.add_triple(idx, entity_dict[0], 0, entity_dict[1], 1, "", 0)
            else:
                store.add_triple(idx, entity_dict[mid-1], mid-1, entity_dict[mid], mid, "", 0)
            
        
        # print문, 출력되지 않음
        def print_all_stores(stores):
            for i, store in enumerate(stores):
                print(f"Store {i + 1}:")
                triples = store.get_triples()
                if not triples:
                    print("  No data in this store.")
                else:
                    for triple in triples:
                        print(f"  {triple}")

        # print_all_stores([store])
        # 앞 또는 뒤와 연결하되, 둘 다 없으면 연결하지 않는다.
        
        tripleset_stores.append(store)


    graph_tripleset_store = TripleStore() # 최종 그래프의 tripleset을 저장할 공간
    node_attributes = defaultdict(set) # 그래프의 각 node가 각 subquery의 어디에 존재하는지에 대한 정보
    # (ex) subquery 1번에 0번째와 2번의 0번째에 해당 entity가 존재한다 -> (0, 1), (0, 2)
    # 강제 연결을 바로 앞의 entity에 할당해주기 위함

    for tripleset_store in tripleset_stores:
        for triple in tripleset_store.get_triples():
            subject = triple["subject"]
            obj = triple["object"]

            node_attributes[subject].add((triple["subject_id"], triple["sub_query_number"]))
            node_attributes[obj].add((triple["object_id"], triple["sub_query_number"]))
            # node attribute 저장

            # Check if the (subject, object) combination already exists
            # 이미 같은 조합이 존재하면 추가하지 않음 (양방향성 등 제거)
            exists = any(
                (existing_triple["subject"] == subject and existing_triple["object"] == obj) or (existing_triple["subject"] == obj and existing_triple["object"] == subject)
                for existing_triple in graph_tripleset_store.get_triples()
            )

            # Add if it doesn't exist
            if not exists:
                graph_tripleset_store.add_triple(
                    triple["sub_query_number"], 
                    subject, 
                    triple["subject_id"], 
                    obj, 
                    triple["object_id"], 
                    triple["relation"], 
                    triple["relation_id"]
                )

    # 중간 저장을 위한 코드
    # import json
    # with open("C:/Users/bjimi/my_files/tripleset.json", 'w') as file:
    #     json.dump(graph_tripleset_store.to_dict(), file, indent=4)

    # with open("C:/Users/bjimi/my_files/node_attributes.json", 'w') as file:
    #     json.dump({key: list(value) for key, value in node_attributes.items()}, file, indent=4)

    # 그래프 형태로 그린 뒤, 따로 떨어진 sub graph 병합 및 순환 구조 제거하는 코드
    graph = None
    graph = compose_graph(graph_tripleset_store, node_attributes)
    extracted_triplesets = []

    pre_core_node = None
    core_node = None

    if graph is not None:


        # 그래프가 존재하는 경우, 일단 연결되어 있지 않은 그래프 분리하고, 더 많은 노드를 가지는 그래프 하나 선택
        components = list(nx.weakly_connected_components(graph))
        largest_component_nodes = max(components, key=len)
        largest_component = graph.subgraph(largest_component_nodes).copy()

        # root node들 뽑기
        root_nodes = [node for node, degree in largest_component.in_degree() if degree == 0]

        # if len(root_nodes) == 0:
            # 순환 그래프 등, root node가 존재하지 않는 경우 - 순환 그래프 없으니까 고려하지 않음
            # node_attribute에 존재하는 노드들 중, 가장 작은 sub_query_number를 가지고 있는 node들 중에
            # 가장 작은 sub_id를 가진 node를 core_node로
            # candidate_nodes = [
            #     (node, 
            #     min(node_attributes[node], key=lambda x: x[1])[1], 
            #     min(node_attributes[node], key=lambda x: x[0])[0])
            #     for node in largest_component.nodes
            # ]
            # core_node = candidate_nodes[0][0] if candidate_nodes else None


        # 만약 root node가 하나라면, 나가는 방향으로 쭉 따라가면서 처음으로 나가는 방향이 두 개 이상인 노드를 core node로 택
        if len(root_nodes) == 1:
            # Root 노드가 하나인 경우만 탐색
            current_node = root_nodes[0]
            while True:
                out_edges = list(largest_component.out_edges(current_node))
                if len(out_edges) > 1:
                    # 처음으로 나가는 방향이 두 개 이상인 노드
                    if is_interrogative_word(current_node):
                        pre_core_node = current_node
                        root_nodes = [target for _, target in out_edges]
                        break
                    else:
                        core_node = current_node
                        break
                elif len(out_edges) == 1:
                    current_node = out_edges[0][1]  # 다음 노드로 이동
                else:
                    break  # 더 이상 나갈 간선이 없으면 종료
                    
        # 만약 root node가 두 개 이상이라면, 
        # root node들의 sub query number를 비교하고, 같다면 node number를 비교해서 더 작은 것이 우선 queue에 들어가도록 한다.
        # queue에서 먼저 들어간 node를 꺼내서 나가는 방향으로 따라가서 queue에 넣고, 다시 다음거 빼고, 또 따라가서 담고 하면서 검사하는데 처음으로 나가는 방향이 두 개 이상인 노드가 발견되었다면 그 노드를 core node로 만든다
        
        if core_node is None:
            if len(root_nodes) > 1:
                # Root node들을 sub query number 기준으로 정렬
                sorted_root_nodes = sorted(
                    root_nodes,
                    key=lambda node: (min(node_attributes[node], key=lambda x: x[1])[1],
                                    min(node_attributes[node], key=lambda x: x[0])[0])
                )


                # 정렬된 root nodes를 큐에 삽입
                queue = deque(sorted_root_nodes)
                visited = set()

                while queue:
                    current_node = queue.popleft()

                    if current_node in visited:
                        continue
                    visited.add(current_node)
                    out_edges = list(largest_component.out_edges(current_node))
                    if len(out_edges) > 1:
                        core_node = current_node  # 처음으로 나가는 방향이 두 개 이상인 노드
                        if is_interrogative_word(core_node):
                            pre_core_node = core_node
                            core_node = None
                        else:
                            break
                    
                    for _, next_node in out_edges:
                        if next_node not in visited:
                            queue.append(next_node)
    
        
        # tripleset의 형태로 추출
        def extract_triplesets(graph):
            triplesets = []
            for u, v, data in graph.edges(data=True):
                subject = u
                obj = v
                predicate = data.get('relation', 'relation')
                triplesets.append((subject, predicate, obj))
            return triplesets

        triplesets = extract_triplesets(graph)

        for trp in triplesets:
            subject = trp[0]
            relation = trp[1]
            obj = trp[2]
            extracted_triplesets.append({'subject': subject, 'relation': relation, 'object': obj})
    
    # else: # 그래프는 없는데 entity는 있는 경우 - 그래프 없는데 core가 왜..필요하지
    #     if len(each_entity_dict_list) != 0:
    #         candidate_nodes = [
    #             (node, node_attributes[node][1], node_attributes[node][0])
    #             for node in largest_component.nodes
    #         ]
    #         candidate_nodes.sort(key=lambda x: (x[1], x[2]))
    #         candidate_nodes 돌면서

    #         core_node = candidate_nodes[0][0] if candidate_nodes else None

    if core_node is None and pre_core_node is not None:
        core_node = pre_core_node
        
    unique_question_entities = list(set(value for d in each_entity_dict_list for value in d.values()))
    entry = {
        "question": q,
        "question_d": decomposed_queries[idx_o],
        "gt_ans": ground_truths[idx_o],
        "gt_doc": evidence_list[idx_o],
        "entities": unique_question_entities,
        "rener": extracted_triplesets,
        "core_node": core_node
    }
    data.append(entry)

    # 출력 테스트 - 그래프 그림
    # import matplotlib.pyplot as plt
    # G = graph

    # labels = {node: f"{node}" for node, data in G.nodes(data=True)}
    # plt.figure(figsize=(12, 12))
    # pos = nx.spring_layout(G, k=1.5)

    # node_colors = [data.get('color', '#00BFFF') for node, data in G.nodes(data=True)]
    # nx.draw(G, pos, labels=labels, node_color=node_colors, edge_color='gray', node_size=5000, font_size=10, arrows=True)
    # edge_labels = nx.get_edge_attributes(G, 'relation')
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    # plt.show()


with open(save_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
