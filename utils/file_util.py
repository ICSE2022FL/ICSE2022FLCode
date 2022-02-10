import chardet


def process_coding(path):
    with open(path, 'rb') as fp:
        file_data = fp.read()
        coding_result = chardet.detect(file_data)
        file_content = file_data.decode(encoding=coding_result['encoding']).strip()
    return file_content


def choose_newlines(content):
    newlines_list = ["\r\n", "\n", "\r"]
    for token in newlines_list:
        if token in content:
            return token
    return "\r\n"
