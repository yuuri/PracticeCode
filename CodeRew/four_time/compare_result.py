import os
import json
from collections import Counter
now = os.getcwd()
ir_path = os.path.join(now,'IR')
fps_path = os.path.join(now,'FPS')
tie_path = os.path.join(now,'TIE')
from common.settings import PROJECT_NAME
from storage.utils import get_reviewer
from storage.utils import get_test_number
from itertools import product
import re
if not os.path.exists(tie_path):
    os.mkdir(tie_path)
ir_list= [i for i in os.walk(ir_path)][0][2]
# fps_file = [i for i in os.walk(fps_path)][0][2]
print(ir_list)
for i in ir_list:
    project = re.search(r'[a-z]+.[a-z]+', i)[0]
    project_index = [i.split('/')[1] for i in PROJECT_NAME]
    realproject = PROJECT_NAME[project_index.index(project)]
    test_number = get_test_number(realproject)
    print(test_number)
    all_reviewer = get_reviewer(realproject)
    print(realproject)
    ir_file = os.path.join(ir_path, i)
    fps_file = os.path.join(fps_path, i)
    tie_file = os.path.join(tie_path, i)
    with open(ir_file, 'r') as f:
        ir_content = f.readlines()[:-1]
    with open(fps_file, 'r') as f2:
        fps_content = f2.readlines()[:-2]
    with open(ir_file, 'r') as f3:
        ir_spend_time = f3.readlines()[-1]
        ir_spend_time = float(re.search(r"\d+(\.\d{2})?",ir_spend_time)[0])
    with open(fps_file, 'r') as f4:
        fps_spend_time = f4.readlines()[-1]
        fps_spend_time = float(re.search(r"\d+(\.\d{2})?",fps_spend_time)[0])

    print('spend_time',ir_spend_time,type(ir_spend_time))
    print('spend_time',fps_spend_time,type(fps_spend_time))
    # 每一个文件
    print(ir_content)
    judge_dict1 = {'right': 0,'wrong': 0,'all': 0}
    judge_dict3 = {'right': 0,'wrong': 0,'all': 0}
    judge_dict5 = {'right': 0,'wrong': 0,'all': 0}
    # for ir,fps in product(ir_content[0:5],fps_content[0:5]):
    for (s, j) in zip(ir_content, fps_content):

        ir_dict = json.loads(s)
        fps_dict = json.loads(j)
        # print(ir)
        # 每一行
        for key,ir_value in ir_dict.items():
            fps_value = fps_dict.get(key)
            if not (isinstance(ir_value, list)) and not (isinstance(fps_value, list)):
                with open(tie_file, 'a') as t:
                    t.write(json.dumps({key: 'this pull number no result info!'}))
                    t.write('\n')
                continue
            else:
                if not (isinstance(ir_value,list)):
                    y = Counter(dict(fps_value))
                    z = list(y)
                    # print(ir[0])
                    #
                    reviewer = all_reviewer.get(int(key))
                    # print(reviewer)
                    if not reviewer:
                        with open(tie_file, 'a') as t:
                            t.write(json.dumps({key: 'this pull number no comment info!'}))
                            t.write('\n')
                        continue
                    if not z:
                        with open(tie_file, 'a') as t:
                            t.write(json.dumps({key: 'this pull number no result info!'}))
                            t.write('\n')
                        continue
                    top1, top3, top5 = [z[0]], z[0:3], z[0:5]
                    print(key, top1, top3, top5)

                    with open(tie_file, 'a') as t:
                        t.write(json.dumps({key: z[0:5]}))
                        t.write('\n')

                    compare_result1 = set(top1) & set(reviewer)
                    compare_result3 = set(top3) & set(reviewer)
                    compare_result5 = set(top5) & set(reviewer)

                    print('result1', compare_result1)
                    print('result3', compare_result3)
                    print('result5', compare_result5)

                    if compare_result1:
                        judge_dict1['right'] += 1
                    else:
                        judge_dict1['wrong'] += 1

                    if compare_result3:
                        judge_dict3['right'] += 1
                    else:
                        judge_dict3['wrong'] += 1

                    if compare_result5:
                        judge_dict5['right'] += 1
                    else:
                        judge_dict5['wrong'] += 1
                elif not (isinstance(fps_value,list)):
                    y = Counter(dict(ir_value))
                    z = list(y)
                    # print(ir[0])
                    #
                    reviewer = all_reviewer.get(int(key))
                    # print(reviewer)
                    if not reviewer:
                        with open(tie_file, 'a') as t:
                            t.write(json.dumps({key: 'this pull number no comment info!'}))
                            t.write('\n')
                        continue
                    if not z:
                        with open(tie_file, 'a') as t:
                            t.write(json.dumps({key: 'this pull number no result info!'}))
                            t.write('\n')
                        continue
                    top1, top3, top5 = [z[0]], z[0:3], z[0:5]
                    print(key, top1, top3, top5)

                    with open(tie_file, 'a') as t:
                        t.write(json.dumps({key: z[0:5]}))
                        t.write('\n')

                    compare_result1 = set(top1) & set(reviewer)
                    compare_result3 = set(top3) & set(reviewer)
                    compare_result5 = set(top5) & set(reviewer)

                    print('result1', compare_result1)
                    print('result3', compare_result3)
                    print('result5', compare_result5)

                    if compare_result1:
                        judge_dict1['right'] += 1
                    else:
                        judge_dict1['wrong'] += 1

                    if compare_result3:
                        judge_dict3['right'] += 1
                    else:
                        judge_dict3['wrong'] += 1

                    if compare_result5:
                        judge_dict5['right'] += 1
                    else:
                        judge_dict5['wrong'] += 1
                else:
                    x = Counter(dict(ir_value))
                    y = Counter(dict(fps_value))
                    z = list(x+y)
                    # print(ir[0])
                    #
                    reviewer = all_reviewer.get(int(key))
                    # print(reviewer)
                    if not reviewer:
                        with open(tie_file, 'a') as t:
                            t.write(json.dumps({key:'this pull number no comment info!'}))
                            t.write('\n')
                        continue
                    if not z:
                        with open(tie_file, 'a') as t:
                            t.write(json.dumps({key:'this pull number no result info!'}))
                            t.write('\n')
                        continue
                    top1, top3, top5 = [z[0]], z[0:3], z[0:5]
                    print(key,top1, top3, top5)

                    with open(tie_file,'a') as t:
                        t.write(json.dumps({key: z[0:5]}))
                        t.write('\n')

                    compare_result1 = set(top1) & set(reviewer)
                    compare_result3 = set(top3) & set(reviewer)
                    compare_result5 = set(top5) & set(reviewer)

                    print('result1', compare_result1)
                    print('result3', compare_result3)
                    print('result5', compare_result5)

                    if compare_result1:
                        judge_dict1['right'] += 1
                    else:
                        judge_dict1['wrong'] += 1

                    if compare_result3:
                        judge_dict3['right'] += 1
                    else:
                        judge_dict3['wrong'] += 1

                    if compare_result5:
                        judge_dict5['right'] += 1
                    else:
                        judge_dict5['wrong'] += 1

    rate1 = '{:.2%}'.format(judge_dict1['right'] / (judge_dict1['right'] + judge_dict1['wrong']))
    rate3 = '{:.2%}'.format(judge_dict3['right'] / (judge_dict3['right'] + judge_dict3['wrong']))
    rate5 = '{:.2%}'.format(judge_dict5['right'] / (judge_dict5['right'] + judge_dict5['wrong']))
    with open(tie_file, 'a') as f:
        f.write(json.dumps(rate1))
        f.write(json.dumps(rate3))
        f.write(json.dumps(rate5))
        f.write('\n')
        f.write('spend_time is {}'.format(ir_spend_time+fps_spend_time))
        f.write('\n')
        # f.write(json.dumps(judge_dict1))
        # f.write(json.dumps(judge_dict3))
        # f.write(json.dumps(judge_dict5))

