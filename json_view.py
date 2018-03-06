import json
import pprint

json_file = 'isolation-result-215674.json'


with open(json_file, 'r') as f:
    content = json.loads(f.read())

content = content['critiques']['680']['rubric_items']['5510']['observation']
#pprint.pprint(content)
print(content)

