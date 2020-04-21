import os
import time
import json
from collections import Counter
from contextlib import ExitStack
from itertools import chain
from collections import defaultdict
from functools import wraps

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB,MultinomialNB,BernoulliNB
from sklearn import preprocessing
from sklearn.model_selection import TimeSeriesSplit

from storage.utils import get_request_user_for_bayes
from storage.utils import get_comment_user, get_comment_times
from storage.utils import get_pull_request_file_count
from storage.utils import get_request_after_open
from storage.utils import get_module_name
from storage.utils import get_file_name
from storage.utils import get_project_name


parent_dir = os.path.abspath(os.pardir)
result_path = os.path.join(parent_dir, 'result', 'bayes_network_test')
if not os.path.exists(result_path):
    os.makedirs(result_path)

file_list = ['request.json','module_name.json', 'reviewer.json']

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_columns', None)

# Cause the code reviewer not only one ,
# so we can consider this is a multi-label question


# feature [full_filename,module_name,Patch Attacher,line_of_code(added deleted updated)]
# [boolean, break, catch, char, class, continue,
# do, double, extends, final, finally,
# float, for, if, else, implements, interface,
# new, package, private, protected, public,
# return, static, super, synchronized, this,
# throw, try, while, null, get, set]


class Test(object):
    def __init__(self):
        self.bayes = GaussianNB()

    def result(self):
        x = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2],[-3, -2], [1, 1]])
        y = np.array([1, 1, 1, 2, 2, 2, 5, 9])
        self.bayes.fit(x, y)
        result = self.bayes.predict([[-1, -1]])
        print(result)
        result2 = self.bayes.predict_proba([[-1, -1]])
        print(result2)
        sort_index = np.argsort(result2)
        print(sort_index)
        print(self.bayes.classes_)
        # result = self.bayes.coef_
        # print(result)


class Pmf(Counter):

    def normalizer(self):
        total = float(sum(self.values()))
        for key in self:
            self[key] /= total

    def __hash__(self):
        return id

    def __eq__(self, other):
        return self is other

    def render(self):
        return zip(*sorted(self.items()))


class PrepareData(object):
    def __init__(self, project_name):
        self.project_name = project_name
        project_str = self.project_name.split("/")[1]
        self.save_path = os.path.join(result_path, project_str)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def get_requester(self):
        requester_info = get_request_user_for_bayes(self.project_name)
        return requester_info

    def get_reviewer(self):
        reviewer_info = get_comment_user(self.project_name)
        return reviewer_info

    def get_reviewer_times(self):
        reviewer_times = get_comment_times(self.project_name)
        return reviewer_times

    def get_file_count(self):
        file_count = get_pull_request_file_count(self.project_name)
        return file_count

    def get_time_after_open(self):
        time_period = get_request_after_open(self.project_name)
        return time_period

    def get_request_module(self):
        module_name = get_module_name(self.project_name)
        return module_name

    def get_file_name(self):
        file_name = get_file_name(self.project_name)
        return file_name

    def get_probability(self, info_dict):
        # 如果是int 类型，不需要转为 tuple 类型，转为tuple 类型是为了将
        # 一个key中的多个Value 合并为一个然后计算概率。

        source = [i if isinstance(i, int) else tuple(i) for i in info_dict.values()]
        pmf = Pmf(source)
        pmf.normalizer()

        pmf_dict = dict(pmf)
        # 这里是将pmf转换为字典，然后将字典中的key元组中的元素取出来
        # pmf 结构是 {info_dict.value: probability}

        probability_dict = {}
        for i in info_dict.items():
            key = i[0]
            value = i[1] if isinstance(i[1], int) else tuple(i[1])
            new_value = pmf_dict.get(value)
            probability_dict.update({key: new_value})

        return probability_dict

    def save_map_info(self, data):

        with ExitStack() as stack:
            io_list = [stack.enter_context(open(f'{os.path.join(self.save_path,file)}', 'w+')) for file in file_list]
            for index, io in enumerate(io_list):
                io.write(json.dumps(data[index],indent=2))

    def build_corpus(self):
        # 获取原始字典
        request = self.get_requester()
        file_count = self.get_file_count()
        reviewers = self.get_reviewer()
        time_after_open = self.get_time_after_open()
        module_name = self.get_request_module()

        df = pd.DataFrame({
            "requester": {i[0]: ','.join(i[1]) for i in request.items()},
            # "requester_probability": request_map,
            "module_name": {i[0]: ','.join(i[1]) for i in module_name.items()},
            # 'module_probability': module_map,
            'time_after_open': time_after_open,
            # "time_probility":time_map,
            "file_num": file_count,
            # "file_num_probility": file_count_map
            "reviewer": {i[0]: ','.join(i[1]) for i in reviewers.items()},
            # "reviewer_probability": review_map
        })
        # fill nan as 0
        df = df.dropna()

        # print(fillna)
        # 将存在多值的列分割为多行，分割符为 ","
        new_df = df.drop('reviewer', axis=1).join(df['reviewer'].str.split(',',expand=True).stack().reset_index(level=1,drop=True).rename("reviewer"))
        new_df = new_df.drop('module_name', axis=1).join(df['module_name'].str.split(',',expand=True).stack().reset_index(level=1,drop=True).rename("module_name"))
        # 调整列顺序
        new_df = new_df[['requester', 'module_name', 'time_after_open', 'file_num', 'reviewer']]

        new_df = new_df.dropna()

        print(new_df)
        save_path = os.path.join(self.save_path, 'data.pkl')
        new_df.to_pickle(save_path)
        return new_df

    def load_pickle(self):
        data_path = os.path.join(self.save_path,'data.pkl')
        pkl_data = pd.read_pickle(data_path)
        return pkl_data


class BuildBayesModel(PrepareData):

    def __init__(self, project_name, corpus):
        super().__init__(project_name)
        self.data = corpus
        self.model = GaussianNB()
        self.label = preprocessing.LabelEncoder()
        self.feature = ['requester', 'module_name', 'time_after_open', 'file_number']

    def feature_prepare(self):
        requester = self.label.fit_transform(self.data.requester)
        r_label = dict(enumerate(self.label.classes_))

        module_name = self.label.fit_transform(self.data.module_name)
        m_label = dict(enumerate(self.label.classes_))

        reviewer = self.label.fit_transform(self.data.reviewer)
        rv_label = dict(enumerate(self.label.classes_))

        map_list = [r_label, m_label, rv_label]
        self.save_map_info(map_list)

        result_data = pd.DataFrame({
            'requester': requester,
            'module_name': module_name,
            'time_after_open': self.data.time_after_open,
            'file_number': self.data.file_num,
            'reviewer': reviewer
        }, index=self.data.index)

        # print(result_data)
        return result_data

    def bayes_model(self, bayes_data):
        self.model.fit(bayes_data[self.feature], bayes_data['reviewer'])
        return self.model

    # def predict_func(self, predict_data):
    #     bayes_data = self.feature_prepare()
    #     bayes_model = self.bayes_model(bayes_data)
    #
    #     predict_pro = bayes_model.predict_proba(predict_data)
    #
    #     top5 = self.get_predict_review(predict_pro)
    #     return top5

    def get_predict_review(self, probability):

        sort_index = np.argsort(probability)
        top5 = sort_index[0][-1:-6:-1]

        return top5

    def data_group(self):
        bayes_data = self.feature_prepare()
        # print(bayes_data)
        # 数据按 index 分组
        pd_new = bayes_data.groupby(bayes_data.index)
        # 获取分组 index 信息 (index 为 pull request id)
        index_list = []
        for index, group in pd_new:
            index_list.append(index)

        # print(index_list)
        # 数据切分
        index = 0
        model = self.bayes_model(bayes_data)

        t_split = TimeSeriesSplit(n_splits=10)
        print(pd_new)
        for train, test in t_split.split(pd_new):
            index += 1
            predict_path = os.path.join(self.save_path, str(index))

            print(f'now is {index} times')
            train_index_list = index_list[:len(train)]
            test_index_list = index_list[len(train):len(train) + len(test)]
            print(train_index_list)
            print(test_index_list)

            train_data = bayes_data.loc[train_index_list]
            # model = self.bayes_model(train_data)

            predict_total = len(test_index_list)
            top1_right = 0
            top3_right = 0
            top5_right = 0
            for i in test_index_list:
                print('====')
                print(i)
                # 获取index 对应信息
                index_data = bayes_data.loc[i].values
                # 获取reviewer 信息

                # 判断numpy array 维度 如果是多个值 维度是 2
                index_ndim = index_data.ndim
                print(index_data)
                top1_list = np.empty([0, 1], dtype=int)
                top3_list = np.empty([0, 3], dtype=int)
                top5_list = np.empty([0, 5], dtype=int)
                if index_ndim > 1:
                    # reviewer 去重
                    reviewer = np.unique(index_data[..., 4])
                    # reviewer.astype(dtype=int)
                    true_reviewer = [self.label.inverse_transform([int(i)]).tolist()[0] for i in reviewer]
                    # predict data 去重
                    unique_index_date = np.unique(index_data[..., :4], axis=0)
                    for sub_data in unique_index_date:
                        predict_pro = model.predict_proba([sub_data])
                        top5 = self.get_predict_review(predict_pro)
                        top1, top3 = top5[0], top5[0:3]
                        print(f'top1 is {top1} top3 is {top3}')
                        top1_list = np.unique(np.append(top1_list, top1)).tolist()
                        top3_list = np.unique(np.append(top3_list, top3)).tolist()
                        top5_list = np.unique(np.append(top5_list, top5)).tolist()

                else:
                    print('single data')
                    predict = index_data[:4]
                    reviewer = index_data[4]
                    true_reviewer = self.label.inverse_transform([int(reviewer)])
                    print(f'reviewer is {reviewer} true reviewer is {true_reviewer}')
                    predict_pro = model.predict_proba([predict])
                    #
                    top5 = self.get_predict_review(predict_pro)
                    top1, top3 = top5[0], top5[0:3]

                    top1_list = np.append(top1_list, top1)
                    top3_list = np.append(top3_list, top3)
                    top5_list = np.append(top5_list, top5)
                predict_top1 = [self.label.inverse_transform([int(i)]).tolist()[0] for i in top1_list]
                predict_top3 = [self.label.inverse_transform([int(i)]).tolist()[0] for i in top3_list]
                predict_top5 = [self.label.inverse_transform([int(i)]).tolist()[0] for i in top5_list]

                with open(predict_path, 'a') as f:
                    f.write(json.dumps('{}'.format({i: predict_top5})))
                    f.write('\n')

                if set(true_reviewer) & set(predict_top1):
                    top1_right += 1
                if set(true_reviewer) & set(predict_top3):
                    top3_right += 1
                if set(true_reviewer) & set(predict_top5):
                    top5_right += 1
                print(f'pull request {i} true review is {true_reviewer} predict review number is {predict_top1,predict_top3,predict_top5}')
            top1_rate = '{:.2%}'.format(top1_right / predict_total)
            top3_rate = '{:.2%}'.format(top3_right / predict_total)
            top5_rate = '{:.2%}'.format(top5_right / predict_total)
            with open(predict_path,'a') as f:
                f.write(json.dumps(f'project is {project}  top1 rate is {top1_rate} right {top1_right} total {predict_total}'))
                f.write('\n')
                f.write(json.dumps(f'project is {project}  top3 rate is {top3_rate} right {top3_right} total {predict_total}'))
                f.write('\n')
                f.write(json.dumps(f'project is {project}  top5 rate is {top5_rate} right {top5_right} total {predict_total}'))
                f.write('\n')


project_list = get_project_name()
time_start = time.time()

for project in project_list:
    print(f'project is {project}')
    # project = 'sindresorhus/awesome'
    ppd = PrepareData(project)
    data = ppd.build_corpus()
    # data = ppd.load_pickle()
#
    app = BuildBayesModel(project, data)
    app.data_group()
end_time = time.time()
#
print(f'spend time is {end_time - time_start}')

