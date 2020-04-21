import os
import re
from collections import defaultdict
path = os.getcwd()
new_path = os.path.join(path,'IR_FPS_TIME')
from common.settings import PROJECT_NAME

result = os.walk(new_path)

file_list = [i for i in result][0][2]
off = -500
file_dict = defaultdict(list)
# for i in file_list:
#     file_path = os.path.join(new_path,i)
#     with open(file_path,'rb') as f:
#         while True:
#             f.seek(off,2)
#             last_line = f.readlines()
#             # print(last_line)
#             if len(last_line) >= 1:
#                 top1 = re.search(r"\d+(\.\d{2}%)?", last_line[-4].decode('utf-8'))[0]
#                 top3 = re.search(r"\d+(\.\d{2}%)?", last_line[-3].decode('utf-8'))[0]
#                 top5 = re.search(r"\d+(\.\d{2}%)?", last_line[-2].decode('utf-8'))[0]
#                 spend_time = re.search(r"\d+(\.\d{2})?", last_line[-1].decode('utf-8'))[0]
#                 list1 = [top1, top3, top5, spend_time]
#                 print(f'{i},{top1},{top3},{top5},{spend_time}')
#                 file_dict.update({i:[top1, top3, top5, spend_time]})
#                 # last_line_read.append(last_line_read2)
#                 break
#             off *= 2

for i in file_list:
    project = re.search(r'[a-z]+.[a-z]+', i)[0]
    project_index = [i.split('/')[1] for i in PROJECT_NAME]
    realproject = PROJECT_NAME[project_index.index(project)]
    file_path = os.path.join(new_path,i)
    # project_name = re.search(r'[a-z]+',i)[0]
    # print(project_name)
    project_list = []

    with open(file_path,'rb') as f:
        while True:
            # f.seek(off,2)
            last_line = f.readlines()
            # print(last_line)
            if len(last_line) >= 1:
                # top1 = re.search(r"\d+(\.\d{2}%+)?", last_line[-4].decode('utf-8'))[0]
                # top3 = re.search(r"\d+(\.\d{2}%+)?", last_line[-3].decode('utf-8'))[0]
                # top5 = last_line[-2].decode('utf-8').strip().replace('""', ',').replace('"','')
                spend_time = re.search(r"\d+(\.\d{2})?", last_line[-1].decode('utf-8'))[0]
                # update_time = re.search(r"\d+(\.\d{2})?", last_line[-2].decode('utf-8'))[0]
                predict_time = re.search(r"\d+(\.\d{2})?", last_line[-2].decode('utf-8'))[0]
                train_time = re.search(r"\d+(\.\d{2})?", last_line[-3].decode('utf-8'))[0]
                # list1 = [top1, top3, top5, spend_time]
                print(f'{realproject},{train_time},{predict_time},{spend_time}')
                project_list = [train_time,predict_time,spend_time]
                file_dict[realproject].append(project_list)
                # last_line_read.append(last_line_read2)
                break
            off *= 2

import json
print(dict(file_dict))
with open('{}.json'.format(new_path),'w') as f:
    f.write(json.dumps(file_dict))
