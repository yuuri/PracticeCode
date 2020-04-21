import json
with open('vue.txt','r') as f:
    content = f.read()

# print(content)
content_list = content.split('\n')[:-1]
print(len(content_list))
line = 1

# for i in content_list:
#     print(f'now line is {line}')
#     i = json.loads(i)
#     print(i)
#     print(type(i))
#     line += 1

s = content_list[254]
print(s)
s = json.loads(s)