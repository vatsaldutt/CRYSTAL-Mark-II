import json

with open("../finetune.json", 'r') as json_data:
    data = json.loads(json_data.read())

print(len(data))
 