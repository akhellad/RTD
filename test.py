import json

with open('labels.json') as f:
    labels = json.load(f)

for category in labels['categories']:
    print(category['id'])