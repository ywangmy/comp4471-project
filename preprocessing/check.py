import json


with open('folds.json') as of:
    json_data = json.load(of)

json_train = json_data['train']
json_val = json_data['val']
json_test = json_data['test']

print('sizes of train, val, test:', len(json_train), len(json_val), len(json_test))

frame_number = set([])

for key in json_train:
    frame_number.add(len(json_train[key]['frames']))

print('range of frame numbers', frame_number)
