import os
import re
import json
import openai
from tqdm import tqdm

#you have to insert your own openai key
os.environ['OPENAI_API_KEY'] = ""


before_qd_file = "rankingqa/rankingqa_test.json"
qd_output_path = "rankingqa/rankingqa_test_qd.json"  
model = 'gpt-4o'

def split_postprocess(split):
    filtered_array = [element for element in split if element]

    result_string = ', '.join(f"'{element}'" for element in filtered_array)

    return result_string

def reductant_postproces(strings):
    original_strings_with_index = list(enumerate(strings))
    
    # Convert to lowercase for processing
    lowercase_strings_with_index = [(i, s.lower()) for i, s in original_strings_with_index]
    
    # Sort by length (descending order)
    lowercase_strings_with_index.sort(key=lambda x: len(x[1]), reverse=True)
    
    # Remove duplicates and select the longest string
    filtered_indices = set()
    seen = set()
    for index, string in lowercase_strings_with_index:
        if not any(string in other for other in seen):
            filtered_indices.add(index)
        seen.add(string)
    
    # Restore the original order
    result = [strings[i] for i in range(len(strings)) if i in filtered_indices]
    return result

def split_sentence(text):
    # Exception patterns to handle specific conjunction cases
    exception_patterns = [
        r'\b(?:both|either|neither)\b.*?\b(?:and|or|nor)\b',
    ]

    # Temporarily replace commas within double quotes
    text = re.sub(r'(".*?")', lambda match: match.group(0).replace(",", "_COMMA_"), text)

    # Process exception patterns to avoid splitting on conjunctions within the patterns
    for pattern in exception_patterns:
        text = re.sub(pattern, lambda match: match.group(0).replace(" ", "_EXCEPTION_"), text)

    # Split based on conjunctions and commas
    conjunctions = r'\b(?:and|or|but|so|nor)\b'
    parts = re.split(fr'\s*(?:{conjunctions}|,)\s*', text)

    # Restore text for exception patterns
    parts = [part.replace("_COMMA_", ",").replace("_EXCEPTION_", " ") for part in parts]

    # Post-processing: Merge segments starting with a double quote if the previous segment ends with a comma
    merged_parts = []
    i = 0
    while i < len(parts):
        part = parts[i].strip()

        # If a segment starts with a double quote
        if part.startswith('"') and merged_parts and merged_parts[-1].endswith(","):
            # Merge with the previous segment
            merged_parts[-1] = merged_parts[-1].strip() + " " + part
        else:
            merged_parts.append(part)

        i += 1

    # 결과 반환
    merged_parts = [part for part in merged_parts if part]
    return merged_parts

Instruction_str = """
Given a query with multiple entities or phrases separated by conjunctions or commas, create each sub-sentence by keeping as much of the original phrasing as necessary to maintain the intent and meaning of the original query.

Instructions:
1. For each separated part, complete the meaning of the sentence using the previous part of the current section.
2. Preserve original terminology wherever possible, guaranteeing that each sub-sentence forms a grammatically complete sentence.
3. The number of sub-sentences created must match the number of separated parts.

Example:
Original Query: "Who's born date is the second earliest, among Neil Armstrong, who attended Purdue University on a Navy scholarship and earned a degree in aeronautical engineering, and later received a master's degree in aerospace engineering from the University of Southern California, and Taylor Swift, and Kylian Mbappe?"
Seperated Parts: ['Who's born date is the second earliest', 'among Neil Armstrong', 'who attended Purdue University on a Navy scholarship', 'earned a degree in aeronautical engineering', 'later received a master's degree in aerospace engineering from the University of Southern California', 'Taylor Swift', 'Kylian Mbappe?']
->
- subquery: Who's born date is the second earliest?
- subquery: What is the born date of Neil Armstrong?
- subquery: Did Neil Armstrong attend Purdue University on a Navy scholarship?
- subquery: Did Neil Armstrong earn a degree in aeronautical engineering?
- subqeury: Did Neil Armstrong receive a master's degree in aerospace engineering from the University of Southern California?
- subqeury: What is the born date of Taylor Swift?
- subqeury: What is the born date of Kylian Mbappe?

Example:
Original Query: "Is nationality of Neil Armstrong, who grew up in Wapakoneta, Ohio, and developed an early interest in flying, and attended Purdue University on a Navy scholarship before transferring to the University of Southern California, and Taylor Swift and Kylian Mbappe the same?"
Seperated Parts: ['Is nationality of Neil Armstrong', 'who grew up in Wapakoneta', 'Ohio', 'developed an early interest in flying', 'attended Purdue University on a Navy scholarship before transferring to the University of Southern California', 'Taylor Swift', 'Kylian Mbappe the same?']
->
- subquery: What is nationality of Neil Armstrong?
- subquery: Did Neil Armstrong grew up in Wapakoneta?
- subquery: Did Neil Armstrong grew up in Wapakoneta, Ohio?
- subqeury: Did Neil Armstrong developed an early interest in flying?
- subqeury: Did Neil Armstrong attended Purdue University on a Navy scholarship before transferring to the University of Southern California?
- subqeury: What is nationality of Taylor Swift?
- subqeury: What is nationality of Kylian Mbappe?
"""

def generate_subqueries2(original_query, split_results):
    prompt = f"""
Original Query: "{original_query}"
Separated Parts:
{split_results}


Now, based on the above examples, provide a list of subqueries for each part of the sentence while preserving the essential meaning.
Don't say anything other than the format that starts with this form (- subquery: ). And the decomposed query that is generated can't just generate the original question. Don't forget the purpose of dividing in the original question.
"""
    return prompt

def subquestion(original_query):
    split_temp = split_sentence(original_query)
    split_results = split_postprocess(split_temp)
    if len(split_temp) == 1:
        return [original_query]
    prompt = generate_subqueries2(original_query, split_results)

    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": Instruction_str},
            {"role": "user", "content": prompt}
            ],
        temperature=0
    )

    subqueries = [
        query.replace("- subquery: ", "").replace("?", "").strip()
        for query in response.choices[0].message.content.splitlines() if query.strip()
    ]
    subqueries = reductant_postproces(subqueries)
    return subqueries

def process_json_entry(data, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write('[') 
        first_entry = True

        for idx,entry in enumerate(tqdm(data, desc="Processing entries")):
            processed_entry = {
                "question": entry["question"],
                "question_d": subquestion(entry["question"]),
                "gt_ans": entry.get("gt_ans"),
                "gt_doc": entry.get("gt_doc", [])
            }

            if not first_entry:
                file.write(',\n')
            else:
                first_entry = False
            

            json.dump(processed_entry, file, ensure_ascii=False, indent=4)
        file.write(']')  

# Load and process JSON
with open(before_qd_file, 'r') as file:
    data = json.load(file)

process_json_entry(data, qd_output_path)
