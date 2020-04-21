import json

predict_list = []
with open('angular.js.txt','r+') as f:
    line = f.readline().strip()
    predict_list.append(line)
    while line:
        line = f.readline().strip()
        predict_list.append(line)

print(predict_list)
print(predict_list[-1])
print(len(predict_list))
result_json = []
flag = 1
