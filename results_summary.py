import json

json_file = "test_results.json"
output_file = "results_summary.json"

exclude_fields = {"i_frame_num", "p_frame_num"}

with open(json_file, "r", encoding="utf-8") as f:
    data = json.load(f)

result = {}

for dataset_name, sequences in data.items():
    field_sums = {}
    field_counts = {}

    for seq_name, metrics in sequences.items():
        for field, value in metrics.items():
            if field in exclude_fields:
                continue

            if isinstance(value, (int, float)):
                field_sums[field] = field_sums.get(field, 0.0) + value
                field_counts[field] = field_counts.get(field, 0) + 1

    result[dataset_name] = {
        field: field_sums[field] / field_counts[field]
        for field in field_sums
    }

# 打印
print(json.dumps(result, indent=2, ensure_ascii=False))

# 保存
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)