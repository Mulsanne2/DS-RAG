import openai
import os
from dotenv import load_dotenv
import spacy
import json
from groq import Groq


load_dotenv()

openai_model = os.getenv("openai_model")
openai.api_key = os.getenv("OPENAI_API_KEY")
gclient = Groq(
    api_key=os.getenv("GROQ_API_KEY")
)

NER_prompt = """
You are a capable entity extractor.
You need to extract all Entities from the given sentence.
When extract entity, follow these guidelines:
1. Entities in all noun forms must be extracted.
2. Extracts all entities with explicitly stated meanings in sentences. Extract entities as specifically as possible without duplicating.
3. All Entities should be individually meaningful, You shouldn't extract meaningless Entities such as Be verbs
4. if a relationship is not explicitly stated, connect and extract related entities. if there is no relationship between entities, list them separately.
   - Entities should be connected based on their semantic relationship or if they belong to the same category (e.g., nationality -> American).
   - Avoid connecting entities where the relationship is unclear or ambiguous.
5. interrogative word must should be treated as an Entity.
All Entities should be extracted in the form of Entities\tEntities\tEntities.
Over-extracting is better than missing out.
Don't print anything other than what you asked

e.g. )
Question: What measures might the international community take if X (formerly Twitter) fails to comply with the European Union's Code?

->
What\tmeasures\tinternational community\tX (formerly Twitter)\tEuropean Union's Code

e.g. )
Question: Who was the Super Bowl MVP in 1979 and 1980.

->
Who\tSuper Bowl MVP\t1979 and 1980

e.g. )
Question: Is Kelly coming to the party tonight?

->
Kelly\tparty\ttonight
"""

def get_entities_llm(query):
  messages = [{
      "role": "system",
      "content": NER_prompt
  }, {
      "role": "user",
      "content": query
  }]
  response = openai.chat.completions.create(
    model=openai_model,
    messages=messages,
    temperature=0
  )

  return response.choices[0].message.content


def get_entities_llama(query):
  messages = [
          {"role": "system", "content": NER_prompt},
          {"role": "user", "content": f"Question: {query}"}
      ]
  
  chat_completion = gclient.chat.completions.create(
      messages=messages, # input prompt to send to the model
      model="llama-3.2-3b-preview", # according to GroqCloud labeling
      temperature=0, # controls diversity
      max_tokens=50, # max number tokens to generate
      top_p=1, # proportion of likelihood weighted options to consider
      stop=None, # string that signals to stop generating
      stream=False, # if set partial messages are sent
  )
  return chat_completion.choices[0].message.content


def get_entities_load(file_path, query):
  with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

  for entry in data:
    question = entry.get("question", "")
    if question == query:  # query와 question 비교
        entities = entry.get("rener", {}).get("entities", [])
        return '\t'.join(entities)
    
  return ""

  
  





spacy_model = os.getenv("spacy_model")
nlp = spacy.load(spacy_model)

def get_entities_spacy(query):
  doc = nlp(query)
  tokens_with_pos = [(token.text, token.pos_, token.dep_, token.idx) for token in doc]
  for token, pos, dep, idx in tokens_with_pos:
    print(f"{token}: {pos} / {dep}")
  print("\n\n")

  valid_pos = {"ADJ", "DET", "NOUN", "NUM", "PRON", "PROPN", "SYM"}
  valid_dep = {
      "PREDET", "DET", "POSS", "POBJ", "NOUNMOD", "ACOMP",
      "AMOD", "APPOS", "ATTR", "COMPOUND", "DATIVE", "DOBJ", "NSUBJ",
      "NSUBJPASS", "NUMMOD", "PCOMP", "QUANTMOD"
  }
  remove_pos = {"CONJ", "CCONJ", "INTJ", "SCONJ"}
  remove_dep = {"PRECONJ", "MARK", "CC", "INTJ", "EXPL"}

  # middle_pos = {"AUX", "PART", "VERB"}

  entities = []  # 최종 엔티티 목록
  current_entity = []  # 현재 연결 중인 엔티티
  
  for token in doc:
    pos, dep = token.pos_, token.dep_
    pos = pos.upper()
    dep = dep.upper()

    # 제거 조건에 해당하는 토큰은 무시
    if pos in remove_pos or dep in remove_dep:
        if current_entity:
            entities.append(" ".join(current_entity))
            current_entity = []
        continue

    # 연결 조건에 해당하는 토큰을 현재 엔티티에 추가
    if (pos in valid_pos or dep in valid_dep):
        current_entity.append(token.text)
    else:
        # 연결 조건을 벗어나면 현재 엔티티를 저장하고 새로 시작
        if current_entity:
            entities.append(" ".join(current_entity))
            current_entity = []

  if current_entity:
      entities.append(" ".join(current_entity))

  return entities


# def get_entities_model(query):


from transformers import LukeTokenizer, LukeForEntitySpanClassification
import spacy
import torch

def get_entities_luke(query):

  # 모델 및 토크나이저 로드
  model_name = "studio-ousia/luke-large-finetuned-conll-2003"
  tokenizer = LukeTokenizer.from_pretrained(model_name)
  model = LukeForEntitySpanClassification.from_pretrained(model_name)

  # SpaCy 모델 로드 (언어 모델에 따라 다르게 설정 가능)
  nlp = spacy.load("en_core_web_trf")  # 영어 텍스트용 SpaCy 모델

  # SpaCy로 엔터티 스팬 추출
  doc = nlp(query)
  entity_spans = [(ent.start_char, ent.end_char) for ent in doc.ents]

  if not entity_spans:
      return []

  # 토크나이저로 입력 변환
  inputs = tokenizer(query, entity_spans=entity_spans, return_tensors="pt")

  # 모델 추론
  outputs = model(**inputs)
  logits = outputs.logits

  # 가장 높은 확률의 엔터티 라벨 예측
  predicted_class_indices = torch.argmax(logits, dim=-1)  # 각 엔터티에 대해 가장 높은 확률의 클래스 인덱스
  entity_labels = [model.config.id2label[idx.item()] for idx in predicted_class_indices.flatten()]  # 텐서를 평탄화 후 처리

  # 결과 출력
  # for span, label in zip(entity_spans, entity_labels):
  #     print(f"Entity: {query[span[0]:span[1]]}, Label: {label}")

  filtered_entities = [
    query[span[0]:span[1]]
    for span, label in zip(entity_spans, entity_labels)
    if label != "O"  # 'O'는 'Outside'로, 라벨이 없는 경우를 나타냄
  ]

  return filtered_entities


def get_entities_spacy_ner(query):
  nlp = spacy.load("en_core_web_trf")  # 영어 모델 사용

  # 텍스트 분석
  doc = nlp(query)

  # 엔터티 추출 (중복 제거)
  entities = list(set(ent.text for ent in doc.ents))

  return entities