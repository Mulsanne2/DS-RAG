from fuzzywuzzy import fuzz
import re
from utils.edge_utils import get_new_relation
import networkx as nx
from utils.triple_store import TripleStore

def parse_relations(relations_string):
    relations = []
    for line in relations_string.strip().split("\n"):
        match = re.match(r"([^|]+)\|([^|]+)\|([^|]+)", line.strip())
        if match:
            relations.append({
                "subject": match.group(1).strip(),
                "relation": match.group(2).strip(),
                "object": match.group(3).strip()
            })
    return relations

def link_apart_node(question, entities, relations):
  additional_relations = []
  used_entities = set()
  for relation in relations:
    start_entity = relation["subject"].strip()
    end_entity = relation["object"].strip()
    used_entities.add(start_entity)
    used_entities.add(end_entity)

  unused_entities = [entity for entity in entities if entity not in used_entities]
  question_words = {"where", "what", "why", "how", "who", "which"}
  filtered_entities = [entity for entity in unused_entities if entity.lower() not in question_words]
  for entity in filtered_entities:
    new_relations_string = get_new_relation(question, list(used_entities), entity)
    new_relations = parse_relations(new_relations_string)
    additional_relations.extend(new_relations)

  return additional_relations

def merge_quoted_entities(decom_q, entities):
    import re

    # Use regex to find quoted sections in the original sentence
    # quoted_matches = re.findall(r'"(.*?)"', decom_q)
    quoted_matches = re.findall(r'\\"(.*?)\\"', decom_q)

    # Split entities by tabs
    entity_list = entities.split('\t')

    # Merge entities that match quoted text
    for quoted in quoted_matches:
        # Remove tabs within the quoted text to create a single entity
        merged_quoted = quoted.replace('\t', '')
        # Find the parts of the quoted text in the entity list
        parts = quoted.split('\t')
        
        # Replace the split parts in the entity list with the merged quoted text
        for part in parts:
            if part in entity_list:
                entity_list[entity_list.index(part)] = merged_quoted
                # Remove duplicates if they exist
                entity_list = [e for i, e in enumerate(entity_list) if e != merged_quoted or i == entity_list.index(e)]

    # Rejoin entities with tabs
    return '\t'.join(entity_list)


def compose_entity_dict(entities, question):
    entity_dict = {}
    splitted_entities = entities.split("\t")
    splitted_entities = [entity.strip() for entity in entities.split("\t") if len(entity.strip()) > 0]
    splitted_entities = sorted(splitted_entities, key=len, reverse=True)

    entity_num = 1
    i = 0

    while i < len(question):
        found = False
        for entity in splitted_entities:
            if len(entity) == 0:
                continue
            if question[i:i+len(entity)].lower() == entity.lower() and len(entity) > 0:
                # If a matching entity is found, assign a number and add it to the dict
                entity_dict[f"e{entity_num}"] = entity
                entity_num += 1
                i += len(entity)  # Skip the index by the length of the matched entity
                found = True
                break
        # If no match is found, move to the next character
        if not found:
            i += 1

    return entity_dict

def compose_entity_dict_v2(entities, question):
    """
    Split the sentence into tokens
    Split entities into tokens as well
    Find entities that start with the first token of the sentence
    Check if the found entities match the subsequent tokens of the sentence
    Store the entity with the most matches first
    Next, find entities that start with the second token of the sentence
    Repeat this process
    """
    entity_dict = {}
    question_tokens = question.split()
    splitted_entities = [entity.strip() for entity in entities.split("\t") if len(entity.strip()) > 0]
    entities_tokens = [entity.split() for entity in splitted_entities]

    def clean_token(token):
        return re.sub(r"[^\w\s]", "", token).strip().lower()
    
    matched_entities = []
    
    for i, token in enumerate(question_tokens):
        for entity, entity_tok in zip(splitted_entities, entities_tokens):
            # Check if the current token matches the first token of the entity

            if clean_token(entity_tok[0]) == clean_token(token):
                match_length = 1  # At least one token matches
                # Check subsequent tokens for a longer match
                while (match_length < len(entity_tok) and 
                       (i + match_length) < len(question_tokens) and
                       clean_token(entity_tok[match_length]) == clean_token(question_tokens[i + match_length])):
                    match_length += 1
                # Append the match details
                if match_length == len(entity_tok):  # Full match
                    def first_last_clean(token):
                        return re.sub(r"^[^\w]+|[^\w]+$", "", token).strip()
                    matched_entities.append((first_last_clean(entity), i, match_length))
    
    matched_entities.sort(key=lambda x: (x[1], -x[2]))

    entity_dict = {}
    unique_matched = {}

    for entity, index, length in matched_entities:
        if index in unique_matched:
            if length > unique_matched[index][1]:  
                unique_matched[index] = (entity, length)
        else:
            unique_matched[index] = (entity, length)

    filtered_data = {}
    for index, (entity, length) in sorted(unique_matched.items()):
        current_range = range(index, index + length)
        # check if contains element
        if any(
            range(existing_index, existing_index + existing_length).start <= current_range.start and
            range(existing_index, existing_index + existing_length).stop >= current_range.stop
            for existing_index, (_, existing_length) in filtered_data.items()
        ):
            continue  
        filtered_data[index] = (entity, length)

    for idx, (index, (entity, _)) in enumerate(sorted(filtered_data.items())):
        entity_dict[idx] = entity

    return entity_dict


def convert_rener(entities, relations):
    # extract entity
    entities = [item.strip() for item in entities.split("\t")]

    # extract relation
    extracted_relations = []
    lines = relations.splitlines()
    for line in lines:
        parts = line.split("||")
        if len(parts) == 3:
            subject = parts[0].strip()
            relation = parts[1].strip()
            obj = parts[2].strip()
            extracted_relations.append({'subject': subject, 'relation': relation, 'object': obj})

    result = {
        'entities': entities,
        'relations': extracted_relations
    }

    return result


def is_interrogative_word(word):
    interrogative_words = {"what", "when", "which", "who", "whom", "whose", "why", "where", "how"}
    return word.lower() in interrogative_words



def find_best_match(new_entity, unique_entities, similarity_threshold=80):
    included_entities = []  

    for existing_entity in unique_entities:
        if new_entity in existing_entity or existing_entity in new_entity:
            included_entities.append(existing_entity)

    # 1. If there is an identical entity, return it
    if new_entity in unique_entities:
        return new_entity

    # 2. If it is included in multiple entities, treat it independently (return None)
    if len(included_entities) > 1:
        return None

    # 3. If it is included in only one entity, return the included entity
    if len(included_entities) == 1:
        return included_entities[0]
    
    for existing_entity in unique_entities:
      similarity = fuzz.ratio(new_entity, existing_entity)
      if similarity >= similarity_threshold:
          return existing_entity  # Return a similar entity
      
    return None  # No match found

def find_best_match_with_original(new_entity, unique_entities, similarity_threshold=80):
    included_entities = []  # List to store included entities

    for existing_entity in unique_entities:
        # If new_entity is included in existing_entity or vice versa, add to the list
        if new_entity in existing_entity or existing_entity in new_entity:
            included_entities.append(existing_entity)

    # 1. If there is an identical entity, return it
    if new_entity in unique_entities:
        return new_entity

    # 2. If it is included in multiple entities, treat it independently (return None)
    if len(included_entities) > 1:
        return None

    # 3. If it is included in only one entity, return the included entity
    if len(included_entities) == 1:
        return included_entities[0]
    
    for existing_entity in unique_entities:
      similarity = fuzz.ratio(new_entity, existing_entity)
      if similarity >= similarity_threshold:
          return existing_entity  # Return a similar entity
      
    return None  # No match found

from collections import defaultdict


def compose_graph(tripleset_store, node_attributes):
    G = nx.DiGraph()

    valid_subject_nodes = set()
    valid_object_nodes = set()

    node_attributes = dict(
        sorted(
            node_attributes.items(),
            key=lambda item: min(x[1] for x in item[1])  # Sort based on the minimum sub_query_number of each group
        )
    )

    for triple in tripleset_store.get_triples():
        
        subj = triple["subject"]
        obj = triple["object"]
        G.add_node(subj, id=triple["subject_id"], sub_query_number=triple["sub_query_number"])
        G.add_node(obj, id=triple["object_id"], sub_query_number=triple["sub_query_number"])


        # Add edge with attributes
        G.add_edge(subj, obj, relation=triple["relation"], relation_id=triple["relation_id"], sub_query_number=triple["sub_query_number"])


        valid_subject_nodes.add(triple["subject"])
        valid_object_nodes.add(triple["object"])


    weakly_connected = list(nx.weakly_connected_components(G))
    if not weakly_connected: 
        return None

    subgraph_node_counts = {i: len(sg) for i, sg in enumerate(weakly_connected)}
    largest_subgraph_idx = max(subgraph_node_counts, key=subgraph_node_counts.get)
    largest_subgraph_nodes = weakly_connected[largest_subgraph_idx]

    remaining_subgraphs = [sg for i, sg in enumerate(weakly_connected) if i != largest_subgraph_idx]


    def find_closest_node(largest_nodes, subgraph_nodes):
        min_diff = float('inf')
        closest_pair = None
        cur_id = float('inf')

        for ln in largest_nodes:
            if ln in valid_object_nodes:
                ln_sub_query_info = {sq: nid for nid, sq in node_attributes[ln]}

                for sn in subgraph_nodes:
                    sn_sub_query_info = {sq: nid for nid, sq in node_attributes[sn]}

                    # Check if the sub_query_numbers are the same
                    common_sub_query_numbers = ln_sub_query_info.keys() & sn_sub_query_info.keys()
                    
                    if common_sub_query_numbers:
                        # Process from the smallest sub_query_number
                        for sq in sorted(common_sub_query_numbers):
                            ln_id = ln_sub_query_info[sq]
                            sn_id = sn_sub_query_info[sq]

                            diff = abs(ln_id - sn_id)

                            if diff < min_diff or (diff == min_diff and ln_id < cur_id) :
                                min_diff = diff
                                cur_id = ln_id
                                closest_pair = (ln, sn)

                        if closest_pair is not None:
                            return closest_pair
                
        return closest_pair
    
    for subgraph_nodes in remaining_subgraphs:

        filtered_nodes = [
            n for n in subgraph_nodes
            if n in valid_subject_nodes
        ]

        # Find the node in largest_subgraph_nodes to connect
        closest_pair = find_closest_node(largest_subgraph_nodes, filtered_nodes)

        if closest_pair:
            ln, sn = closest_pair
            G.add_edge(ln, sn)
            largest_subgraph_nodes.update(subgraph_nodes)  # Merge nodes into largest subgraph

    G = find_and_remove_cycles_with_nx(G)

    return G


from collections import defaultdict, deque


def find_and_remove_cycles_with_nx(graph):
    G = nx.DiGraph(graph)

    i = 0

    while not nx.is_directed_acyclic_graph(G):
        if (i > 10):
            break

        i += 1
        try:
            # Step 1: Find a cycle
            cycle = nx.find_cycle(G, orientation="original")
            cycle_nodes = {v for u, v, _ in cycle}  # Nodes in the cycle

            # Step 2: Calculate degrees of cycle nodes
            node_degrees = {node: (G.in_degree(node), G.degree(node)) for node in cycle_nodes}

            # Step 3: Find the node with the highest total degree
            target_node = max(node_degrees, key=lambda x: node_degrees[x][1])

            # Step 4: Remove one incoming edge to the target node
            for u, v in G.in_edges(target_node):
                if u in cycle_nodes:
                    G.remove_edge(u, v)
                    break  # Remove only one edge
        except nx.NetworkXNoCycle:
            break

    return G