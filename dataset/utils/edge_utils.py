import spacy
import os
import re
import openai
from utils.triple_store import TripleStore

openai_model = os.getenv("openai_model")

relations_extract_prompt = """
relationships should be selected as an entity number corresponding to the target and subject.
following guidlines:
1. Only the numbers is used at the tripeset.
2. All entered relationship numbers must exist at least once.
Please print it in the form of entity number(eN) || relationship number(rN) || entity number(eN)\nentity number(eN) || relationship number(rN) || entity number(eN) ...
"""

relations_extract_prompt_v2 = """
relationships should be selected as an entity number corresponding to the target and subject.
following guidlines:
1. Only the numbers is used at the tripeset.
2. All entered entities numbers must use at least once.
3. Extract implicit or omitted relationships to indicate the meaning of all the sentences. 
Please print it in the form of entity number(eN) | relationship number(rN) | entity number(eN)\nentity number(eN) | relationship number(rN) | entity number(eN) ...
"""

add_additional_relation_prompt = """
Your task is to identify relationships between the specified CORE and other entities from the list when given a question and a list of entities, and CORE.
**Only extract relationships that explicitly include the CORE.**
output format is entity | relation | entity\nentity | relation | entity...

e.g.)
Question: The artist painting received praise from critics
Entities: painting\t praise\t critics\t
CORE: artist

->
painting | of | artist
"""

def relations_extract(sentence, numbered_entities, numbered_relations):
  
  all = "Sentense: " + sentence + "\nNumbered Entities: " + numbered_entities + "\nNumbered relations: " + numbered_relations
  messages = [{
      "role": "system",
      "content": relations_extract_prompt
  }, {
      "role": "user",
      "content": all
  }]
  response = openai.chat.completions.create(
    model=openai_model,
    messages=messages,
    temperature=0
  )

  return response.choices[0].message.content

def relations_extract_v2(sentence, numbered_entities, numbered_relations):
  
  all = "Sentense: " + sentence + "\nNumbered Entities: " + numbered_entities + "\nNumbered relations: " + numbered_relations
  messages = [{
      "role": "system",
      "content": relations_extract_prompt_v2
  }, {
      "role": "user",
      "content": all
  }]
  response = openai.chat.completions.create(
    model=openai_model,
    messages=messages,
    temperature=0
  )

  return response.choices[0].message.content

spacy_model = os.getenv("spacy_model")
nlp = spacy.load(spacy_model)

def extract_text_between_entities(sentence, entities):
    split_pos=["DET", "CCONJ", "INTJ", "PRON", "PROPN", "PUNCT", "SYM", "X", "NOUN", "ADJ"]
    removed_pos =["DET", "CCONJ", "INTJ", "PUNCT", "X"]
    doc = nlp(sentence)
    tokens_with_pos = [(token.text, token.pos_, token.idx) for token in doc]

    lower_sentence = sentence.lower()

    # 결과를 저장할 리스트
    extracted_texts = []

    # 문장의 시작과 첫번째 엔티티
    if entities:
        first_entity = entities[0].lower()
        start_pattern = r"^(.*?)" + re.escape(first_entity)
        start_match = re.search(start_pattern, lower_sentence)
        if start_match:
            start_text = start_match.group(1)
            tokens_in_start = [
                (token, pos, idx) for token, pos, idx in tokens_with_pos
                if 0 <= idx < start_match.end(1)
            ]
            filtered_start = [
                token.strip() for token, pos, idx in tokens_in_start if pos not in removed_pos
            ]
            if filtered_start:
                extracted_texts.append(" ".join(filtered_start))

    start_idx = 0  # 매칭을 시작할 시작 인덱스
    
    for i in range(len(entities) - 1):
        # 엔터티 간 텍스트를 추출하는 정규표현식
        pattern = re.escape(entities[i].lower()) + r"(.*?)" + re.escape(entities[i + 1].lower())

        # 맨 앞, 맨 뒤도 추가

        
        # 시작 인덱스를 기준으로 패턴 매칭
        match = re.search(pattern, lower_sentence[start_idx:])
        
        if match:
            match_text = match.group(1)
            start_idx_in_match = match.start(1)
            end_idx_in_match = match.end(1)
            
            # 매칭된 텍스트 앞 공백 처리
            leading_spaces = len(match_text) - len(match_text.lstrip())
            adjusted_start_idx = start_idx + start_idx_in_match + leading_spaces
            adjusted_end_idx = start_idx + end_idx_in_match

            tokens_in_match = [
                (token, pos, idx) for token, pos, idx in tokens_with_pos
                if adjusted_start_idx <= idx < adjusted_end_idx
            ]

            # split_text = []
            # current_text = []
            # for token, pos, idx in tokens_in_match:
            #     if pos in split_pos:
            #         # split_pos 토큰을 기준으로 분할, 기준 토큰은 제외
            #         joined_text = " ".join(current_text).strip()
            #         if joined_text:
            #             extracted_texts.append(joined_text)
            #         current_text = []
            #     else:
            #         if token.strip():
            #             current_text.append(token.strip())

            # # 마지막 분할된 텍스트가 남아있다면 추가
            # joined_text = " ".join(current_text).strip()
            # if joined_text:
            #     extracted_texts.append(joined_text)
                
            # # start_idx를 갱신하여 다음 매칭부터 이어서 검사
            # start_idx = adjusted_end_idx
            filtered_tokens = []
            for token, pos, idx in tokens_in_match:
                if pos not in removed_pos:  # 제거할 POS가 아닌 경우
                    filtered_tokens.append(token.strip())  # 공백 제거 후 추가
            
            # 결과를 텍스트로 합쳐서 저장
            if filtered_tokens:
                filtered_text = " ".join(filtered_tokens)  # 토큰들을 연결
                extracted_texts.append(filtered_text)

            # start_idx를 갱신하여 다음 매칭부터 이어서 검사
            start_idx = adjusted_end_idx

    # 마지막 엔티티와 문장 끝
    if entities:
        last_entity = entities[-1].lower()
        end_pattern = re.escape(last_entity) + r"(.*?)$"
        end_match = re.search(end_pattern, lower_sentence)
        if end_match:
            end_text = end_match.group(1)
            tokens_in_end = [
                (token, pos, idx) for token, pos, idx in tokens_with_pos
                if end_match.start(1) <= idx < len(sentence)
            ]
            filtered_end = [
                token.strip() for token, pos, idx in tokens_in_end if pos not in removed_pos
            ]
            if filtered_end:
                extracted_texts.append(" ".join(filtered_end))
                  

    # 탭 구분자를 넣고 문자열로 반환
    return "\t".join(extracted_texts)

def transform_data(data, entity_dict, relations_dict):
    result = []
    for line in data.split("\n"):
      try:
        e1, r, e2 = line.split(" || ")
        e1 = e1.strip()
        r = r.strip()
        e2 = e2.strip()

        e1_num = int(re.search(r'\d+', e1).group())  # e1에서 숫자 추출
        r_num = int(re.search(r'\d+', r).group())    # r에서 숫자 추출
        e2_num = int(re.search(r'\d+', e2).group())  # e2에서 숫자 추출

        # 여기서 만약에 숫자 순서가 다르면 맞춰주기
        if e2_num < e1_num:
            e1_num, e2_num = e2_num, e1_num


        transformed = f"{entity_dict[e1_num]} || {relations_dict[r_num]} || {entity_dict[e2_num]}"

        result.append(transformed)

      except (ValueError, KeyError):
          # 잘못된 형식이거나 매핑되지 않는 값이 있는 경우 건너뜀
          continue
    return "\n".join(result)

def relation_extract_between_entities(question, entity_dict):
    entity_list = list(entity_dict.values())
    extracted_relations = extract_text_between_entities(question, entity_list)
    splitted_relations = extracted_relations.split("\t")

    relations_dict = {i: relation for i, relation in enumerate(splitted_relations)}
    formatted_relations = "\t".join([f"{v}: r{k}" for k, v in relations_dict.items()])
    formatted_entities = "\t".join([f"{v}: e{k}" for k, v in entity_dict.items()])

    relations = relations_extract(question, formatted_entities, formatted_relations)

    # real_relations = transform_data(relations, entity_dict, relations_dict)

    return relations_dict, relations


def relation_extract_between_entities_v2(question, entity_dict):
    entity_list = list(entity_dict.values())
    extracted_relations = extract_text_between_entities(question, entity_list)
    splitted_relations = extracted_relations.split("\t")
    splitted_relations.extend(["that"]) #v2
    print(splitted_relations)
    print()
    
    relations_dict = {f"r{i+1}": relation for i, relation in enumerate(splitted_relations)}
    formatted_relations = "\t".join([f"{v}: {k}" for k, v in relations_dict.items()])
    formatted_entities = "\t".join([f"{v}: {k}" for k, v in entity_dict.items()])

    relations = relations_extract_v2(question, formatted_entities, formatted_relations)
    real_relations = transform_data(relations, entity_dict, relations_dict)

    print(real_relations) #test
    print()
    
    return real_relations

def get_new_relation(query, entities, entity):
  entities = "\t".join(entities)
  all = "Question: " + query + "\nEntities: " + entities + "\nCORE: " + entity
  messages = [{
      "role": "system",
      "content": add_additional_relation_prompt
  }, {
      "role": "user",
      "content": all
  }]

  response = openai.chat.completions.create(
    model=openai_model,
    messages=messages,
    temperature=0
  )

  return response.choices[0].message.content