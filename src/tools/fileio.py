import json


def read_txt_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return content


def write_json_file(file_path, data):
    with open(file_path, 'w') as file:
        json.dump(data, file)
