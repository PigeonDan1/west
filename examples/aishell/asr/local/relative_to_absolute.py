import json
import sys

AUDIO_BASE_DIR = "/F00120240032/audio_segments"

try:
    input_file = sys.argv[1]
    output_file = sys.argv[2]
except IndexError:
    print("Usage: python relative_to_absolute.py <input_jsonl> <output_jsonl>")
    sys.exit(1)

with open(input_file, 'r', encoding='utf-8') as infile, \
     open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        data = json.loads(line)
        relative_path = data['wav']
        absolute_path = f"{AUDIO_BASE_DIR}/{relative_path}"
        data['wav'] = absolute_path
        outfile.write(json.dumps(data, ensure_ascii=False) + '\n')