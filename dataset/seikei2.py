import os

for i in range(1, 10133+1):
    file_name = 'data/label/' + str(i) + '.txt'
    with open(file_name, 'r') as f:
        s = f.read()
    
    original_file_path = 'data/HR/' + s + '.png'
    new_file_path = 'data/HR/' + str(i) + '.png'
    os.rename(original_file_path, new_file_path)