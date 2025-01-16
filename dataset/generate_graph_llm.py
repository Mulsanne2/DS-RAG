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


decomposed_query_file_path = 'rankingqa/rankingqa_test_qd.json'

save_path = "rankingqa/rankingqa_test_graph.json"



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
        
        # Step 1: Node creation
        # entities = get_entities_llm(decom_q) #\t로 연결된 list 형태
        entity_dict = compose_entity_dict_v2(entities, decom_q) 
        each_entity_dict_list.append(entity_dict)
        # eN: Convert entities present in the sentence into a dict format corresponding to each sub-query
        # print(each_entity_dict_list)

        # Step 2: Edge creation and Step 3: Generate tripleset for sub-graph creation (in the format e1 || r1 || e3)
        relations_dict, relations = relation_extract_between_entities(decom_q, entity_dict) # Extract relations between entities, not the entities themselves

        # Create and save a triple set store for each sub-query
        store = TripleStore()

        for line in relations.split("\n"):
            try:
                e1, r, e2 = line.split(" || ")
                e1 = e1.strip()
                r = r.strip()
                e2 = e2.strip()

                e1_num = int(re.search(r'\d+', e1).group())  
                r_num = int(re.search(r'\d+', r).group())    
                e2_num = int(re.search(r'\d+', e2).group())  

                # If the order of numbers is incorrect, adjust it (node order - directional alignment)
                if e2_num < e1_num:
                    e1_num, e2_num = e2_num, e1_num

                store.add_triple(idx, entity_dict[e1_num], e1_num, entity_dict[e2_num], e2_num, relations_dict[r_num], r_num)

            except Exception as e:
                # Skip if the format is incorrect or contains unmapped values
                continue
        

        # For nodes that do not have a triple set, connect them (Steps 3-2, 3-3)
        # If a value in entity_dict is not included in the subject or object of the store, add a tripleset connecting it to the previous (or next) entity
        entity_ids = set()
        entity_ids.update(store.get_subject_ids())
        entity_ids.update(store.get_object_ids())

        # question_words = {'what', 'who', 'when', 'where', 'why', 'how', 'which'}
        # # Check for keys in entity_dict not present in subject_ids or object_ids
        # missing_ids = [key for key in entity_dict.keys() if key not in entity_ids and 
        #        entity_dict[key].lower() not in question_words]

        missing_ids = [key for key in entity_dict.keys() if key not in entity_ids]
        
        for mid in missing_ids:
            
            if mid == 0: 
                if 1 in entity_dict:
                    store.add_triple(idx, entity_dict[0], 0, entity_dict[1], 1, "", 0)
            else:
                store.add_triple(idx, entity_dict[mid-1], mid-1, entity_dict[mid], mid, "", 0)
        
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
        
        tripleset_stores.append(store)


    graph_tripleset_store = TripleStore() # Space to store the tripleset of the final graph
    node_attributes = defaultdict(set) # Information about where each node exists in each subquery within the graph
    # (e.g.) If an entity exists at the 0th and 2nd positions of subquery 1 -> (0, 1), (0, 2)
    # This is to assign forced connections to the immediately preceding entity

    for tripleset_store in tripleset_stores:
        for triple in tripleset_store.get_triples():
            subject = triple["subject"]
            obj = triple["object"]

            node_attributes[subject].add((triple["subject_id"], triple["sub_query_number"]))
            node_attributes[obj].add((triple["object_id"], triple["sub_query_number"]))
            # store node attribute

            # Check if the (subject, object) combination already exists
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

    # Code to draw in graph form, merge disconnected sub-graphs, and remove cyclic structures
    graph = None
    graph = compose_graph(graph_tripleset_store, node_attributes)
    extracted_triplesets = []

    pre_core_node = None
    core_node = None

    if graph is not None:


        # If a graph exists, first separate disconnected sub-graphs and select the one with the most nodes
        components = list(nx.weakly_connected_components(graph))
        largest_component_nodes = max(components, key=len)
        largest_component = graph.subgraph(largest_component_nodes).copy()

        root_nodes = [node for node, degree in largest_component.in_degree() if degree == 0]

        # If there is only one root node, follow the outgoing edges and select the first node with more than one outgoing edge as the core node
        if len(root_nodes) == 1:
            # Root 노드가 하나인 경우만 탐색
            current_node = root_nodes[0]
            while True:
                out_edges = list(largest_component.out_edges(current_node))
                if len(out_edges) > 1:
                    # Explore only when there is one root node
                    if is_interrogative_word(current_node):
                        pre_core_node = current_node
                        root_nodes = [target for _, target in out_edges]
                        break
                    else:
                        core_node = current_node
                        break
                elif len(out_edges) == 1:
                    current_node = out_edges[0][1]  # Move to the next node
                else:
                    break  # Exit if there are no more outgoing edges
                    
        # If there are multiple root nodes,
        # Compare the sub-query numbers of the root nodes. If they are the same, compare the node numbers and prioritize the smaller one to enter the queue first.
        # Take the first node from the queue, follow the outgoing edges, add nodes to the queue, and repeat. While doing so, if a node with more than one outgoing edge is found, designate it as the core node.
        
        if core_node is None:
            if len(root_nodes) > 1:
                # Sort root nodes based on sub-query numbers
                sorted_root_nodes = sorted(
                    root_nodes,
                    key=lambda node: (min(node_attributes[node], key=lambda x: x[1])[1],
                                    min(node_attributes[node], key=lambda x: x[0])[0])
                )


                # Insert sorted root nodes into the queue
                queue = deque(sorted_root_nodes)
                visited = set()

                while queue:
                    current_node = queue.popleft()

                    if current_node in visited:
                        continue
                    visited.add(current_node)
                    out_edges = list(largest_component.out_edges(current_node))
                    if len(out_edges) > 1:
                        core_node = current_node  # The first node with more than one outgoing edge
                        if is_interrogative_word(core_node):
                            pre_core_node = core_node
                            core_node = None
                        else:
                            break
                    
                    for _, next_node in out_edges:
                        if next_node not in visited:
                            queue.append(next_node)
    
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

with open(save_path, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)
