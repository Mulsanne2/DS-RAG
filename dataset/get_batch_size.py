import json

with open('comparison/comparison_train_graph.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

TOTAL= 0
output_lines = []
for entry in data:
    count = len(entry.get("dqd_pair", []))
    if count > 0:
        output_lines.append(str(count))
        TOTAL+=count

# Save to txt file
output_path = "graph/compasrion/train/batch.txt"
with open(output_path, "w") as file:
    file.write("\n".join(output_lines))

print(TOTAL)
total_dqd_count = sum(len(entry.get("dqd_pair", [])) for entry in data)
print(total_dqd_count)