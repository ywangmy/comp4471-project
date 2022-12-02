import json


with open('folds.json') as of:
    json_data = json.load(of)

json_train = json_data['train']
json_val = json_data['val']
json_test = json_data['test']

print('sizes of train, val, test:', len(json_train), len(json_val), len(json_test))

frame_number = [0] * 65

for v in json_train:
    frame_number[len(v[3])] += 1
for v in json_val:
    frame_number[len(v[3])] += 1
for v in json_test:
    frame_number[len(v[3])] += 1
print('range of frame numbers', frame_number[30])

print('train[:2]', json_train[:2])
print('val[:2]', json_val[:2])
print('test[:2]', json_test[:2])
