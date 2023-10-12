import json
from typing import List


def prepare_data(filenames: List[str], datapath: str='datafiles/', test: bool=False) -> dict[str, List]:
    """
    Function that simply parses .json & .jsonl files, and appends
    dialogues (documents) and summaries in separate lists, to be stored
    in a dictionary.
    """

    data_dict = {'document': [], 'summary': []}

    # Extracting & reformatting  files
    for f_name in filenames:
        with open(datapath + f_name, mode='r', encoding='utf8') as json_file:
            # Parsing .json file
            if f_name.endswith('.json'):
                data = json.load(json_file)
            elif f_name.endswith('.jsonl'):
                data = [json.loads(sample) for sample in list(json_file)]

            for sample in data:
                data_dict['document'].append(sample['dialogue'])
                if not test:
                    data_dict['summary'].append(sample['summary'])
    
    return data_dict