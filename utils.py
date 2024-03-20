# Copyright (c) 2022 Huawei Technologies Co.,Ltd.
#
# openGauss is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#          http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.

import re
from collections import defaultdict
from enum import Enum
from functools import lru_cache
from typing import List, Tuple, Sequence, Any
from contextlib import contextmanager
import logging

import sqlparse
from sqlparse.tokens import Name
from sqlparse.sql import Function, Parenthesis, IdentifierList
from sql_metadata import Parser

from HyperQO.sql_feature.bag_of_predicates import BagOfPredicates
from HyperQO.sql_feature.utils import build_similarity_index, embed_queries_and_plans

COLUMN_DELIMITER = ', '
QUERY_PLAN_SUFFIX = 'QUERY PLAN'
EXPLAIN_SUFFIX = 'EXPLAIN'
ERROR_KEYWORD = 'ERROR'
PREPARE_KEYWORD = 'PREPARE'


class QueryType(Enum):
    INEFFECTIVE = 0
    POSITIVE = 1
    NEGATIVE = 2


class IndexType(Enum):
    ADVISED = 1
    REDUNDANT = 2
    INVALID = 3


def replace_function_comma(statement):
    """Replace the ? in function to the corresponding value to ensure that prepare execution can be executed properly"""
    function_value = {'count': '1', 'decode': "'1'"}
    new_statement = ''
    for token in get_tokens(statement):
        value = token.value
        if token.ttype is Name.Placeholder and token.value == '?':
            function_token = None
            if isinstance(token.parent, Parenthesis) and isinstance(token.parent.parent, Function):
                function_token = token.parent.parent
            elif isinstance(token.parent, IdentifierList) \
                    and isinstance(token.parent.parent, Parenthesis) \
                    and isinstance(token.parent.parent.parent, Function):
                function_token = token.parent.parent.parent
            if function_token:
                replaced_value = function_value.get(function_token.get_name().lower(), None)
                value = replaced_value if replaced_value else value
        new_statement += value
    return new_statement


class UniqueList(list):

    def append(self, item: Any) -> None:
        if item not in self:
            super().append(item)

    def extend(self, items: Sequence[Any]) -> None:
        for item in items:
            self.append(item)


class ExistingIndex:

    def __init__(self, schema, table, indexname, columns, indexdef):
        self.__schema = 'public'
        self.__table = table
        self.__indexname = indexname
        self.__columns = columns
        self.__indexdef = indexdef
        self.__primary_key = False
        self.__is_unique = False
        self.__index_type = ''
        self.redundant_objs = []

    def set_is_unique(self):
        self.__is_unique = True

    def get_is_unique(self):
        return self.__is_unique

    def set_index_type(self, index_type):
        self.__index_type = index_type

    def get_index_type(self):
        return self.__index_type

    def get_table(self):
        return self.__table

    def get_schema(self):
        return 'public'

    def get_indexname(self):
        return self.__indexname

    def get_columns(self):
        return self.__columns

    def get_indexdef(self):
        return self.__indexdef

    def is_primary_key(self):
        return self.__primary_key

    def set_is_primary_key(self, is_primary_key: bool):
        self.__primary_key = is_primary_key

    def get_schema_table(self):
        return self.__schema + '.' + self.__table

    def __str__(self):
        return f'{self.__schema}, {self.__table}, {self.__indexname}, {self.__columns}, {self.__indexdef})'

    def __repr__(self):
        return self.__str__()


class AdvisedIndex:
    def __init__(self, tbl, cols, index_type=None):
        self.__table = tbl
        self.__columns = cols
        self.benefit = 0
        self.__storage = 0
        self.__index_type = index_type
        self.association_indexes = defaultdict(list)
        self.__positive_queries = []
        self.__source_index = None
        self.__query_pos = {}
        self.__index_potential = None

    def add_query_pos(self, pos, queries_potential):
        if pos not in self.__query_pos:
            self.__query_pos[pos] = queries_potential[pos]

    def set_query_pos(self, query_pos):
        self.__query_pos = query_pos

    def get_index_query_potential_dict(self):
        return self.__query_pos

    def get_index_potential_average(self):
        potential = 0
        potential_dict = self.get_index_query_potential_dict()
        for key, value in potential_dict.items():
            potential += value
        return potential / len(potential_dict)

    def set_index_potential(self):
        self.__index_potential = self.get_index_potential_average()

    def set_source_index(self, source_index: ExistingIndex):
        self.__source_index = source_index

    def get_source_index(self):
        return self.__source_index

    def append_positive_query(self, query):
        self.__positive_queries.append(query)

    def get_positive_queries(self):
        return self.__positive_queries

    def set_storage(self, storage):
        self.__storage = storage

    def get_storage(self):
        return self.__storage

    def get_table(self):
        return self.__table

    def get_schema(self):
        # return self.__table.split('.')[0]
        return 'public'

    def get_columns(self):
        return self.__columns

    def get_columns_num(self):
        return len(self.get_columns().split(COLUMN_DELIMITER))

    def get_index_type(self):
        return self.__index_type

    def get_index_statement(self):
        table_name = self.get_table().split('.')[-1]
        index_name = 'idx_%s_%s%s' % (table_name, (self.get_index_type() + '_' if self.get_index_type() else ''),
                                      '_'.join(self.get_columns().split(COLUMN_DELIMITER))
                                      )
        statement = 'CREATE INDEX %s ON %s%s%s;' % (index_name, self.get_table(),
                                                    '(' + self.get_columns() + ')',
                                                    (' ' + self.get_index_type() if self.get_index_type() else '')
                                                    )
        return statement

    def set_association_indexes(self, association_indexes_name, association_benefit):
        self.association_indexes[association_indexes_name].append(association_benefit)

    def match_index_name(self, index_name):
        # match <13382>btree_date_dim_d_month_seq
        schema = self.get_schema()
        if schema == 'public':
            return index_name[1:-1].endswith(f'btree_{self.get_index_type() + "_" if self.get_index_type() else ""}'
                                             f'{self.get_table().split(".")[-1]}_'
                                             f'{"_".join(self.get_columns().split(COLUMN_DELIMITER))}')
        # else:
        #     return index_name.endswith(f'btree_{self.get_index_type() + "_" if self.get_index_type() else ""}'
        #                                f'{self.get_table().replace(".", "_")}_'
        #                                f'{"_".join(self.get_columns().split(COLUMN_DELIMITER))}')

    def __str__(self):
        return f'table: {self.__table} columns: {self.__columns} index_type: ' \
               f'{self.__index_type} storage: {self.__storage}'

    def __repr__(self):
        return self.__str__()


def singleton(cls):
    instances = {}

    def _singleton(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return _singleton


@singleton
class IndexItemFactory:
    def __init__(self):
        self.indexes = {}

    def get_index(self, tbl, cols, index_type):
        if COLUMN_DELIMITER not in cols:
            cols = cols.replace(',', COLUMN_DELIMITER)
        if not (tbl, cols, index_type) in self.indexes:
            self.indexes[(tbl, cols, index_type)] = AdvisedIndex(tbl, cols, index_type=index_type)
        return self.indexes[(tbl, cols, index_type)]


def match_table_name(table_name, tables):
    for elem in tables:
        item_tmp = '_'.join(elem.split('.'))
        if table_name == item_tmp:
            table_name = elem
            break
        elif 'public_' + table_name == item_tmp:
            table_name = 'public.' + table_name
            break
    else:
        return False, table_name
    return True, table_name


class QueryItem:
    __valid_index_list: List[AdvisedIndex]

    def __init__(self, sql: str, freq: float):
        self.__statement = sql
        self.__frequency = freq
        self.__valid_index_list = []
        self.__benefit = 0

    def get_statement(self):
        return self.__statement

    def get_frequency(self):
        return self.__frequency

    def append_index(self, index):
        self.__valid_index_list.append(index)

    def get_indexes(self):
        return self.__valid_index_list

    def reset_opt_indexes(self):
        self.__valid_index_list = []

    def get_sorted_indexes(self):
        return sorted(self.__valid_index_list, key=lambda x: (x.get_table(), x.get_columns(), x.get_index_type()))

    def set_benefit(self, benefit):
        self.__benefit = benefit

    def get_benefit(self):
        return self.__benefit

    def __str__(self):
        return f'statement: {self.get_statement()} frequency: {self.get_frequency()} ' \
               f'index_list: {self.__valid_index_list} benefit: {self.__benefit}'

    def __repr__(self):
        return self.__str__()


def get_indexable_columns(parser):
    columns = []
    for position, _columns in parser.columns_dict.items():
        if position.upper() not in ['SELECT', 'INSERT', 'UPDATE']:
            columns.extend(_columns)
    flatten_columns = UniqueList()
    for column in flatten(columns):
        flatten_columns.append(column)
    return flatten_columns


class WorkLoad:
    def __init__(self, queries: List[QueryItem], plans: List[str]):
        self.__indexes_list = []
        self.__queries = queries
        self.__tables = set()
        self.__index_names_list = [[] for _ in range(len(self.__queries))]
        self.__indexes_costs = [[] for _ in range(len(self.__queries))]
        self.__plans = plans
        self.__plan_list = [[] for _ in range(len(self.__queries))]
        self.__query_potential = []
        self.__query_index_cost_cache = {}
        self.__origin_cost = 0
        self.__queries_origin_cost = []
        self.__sim_index = None
        self.__dictionary = None
        self.__query_to_benefit = defaultdict()

    def get_query_index_cost_cache(self):
        return self.__query_index_cost_cache

    def set_workload_similarity_with_predicate_feature(self, sql_embedder):
        queries = [self.__queries[i].get_statement() for i in range(len(self.__queries))]
        workload_embeddings, workload_predicates, self.__dictionary = embed_queries_and_plans(sql_embedder, queries,
                                                                                              self.__plans)
        self.__sim_index = build_similarity_index(sql_embedder.model, workload_embeddings, workload_predicates,
                                                  self.__dictionary)

    def get_similarity_with_predicate(self, plan):
        sim = self.__sim_index[self.__dictionary.doc2bow(
            BagOfPredicates().extract_predicates_from_plan(plan["Plan"]))]
        return sim

    def get_similarity_with_indexable_column(self, query1: str, query2: str):
        query1_indexable_columns = get_indexable_columns(Parser(query1))
        query2_indexable_columns = get_indexable_columns(Parser(query2))

        query_weights = defaultdict(dict)

        # 计算 query1 的权重
        for column in query1_indexable_columns:
            for table in self.get_tables():
                if column not in table.columns:
                    continue
                query_weights[query1][column] = (1 - table.get_n_distinct(column)) * table.size_weight

        # 添加 query2 中不存在于 query1 的索引列，并将权重设置为0
        for column in query2_indexable_columns:
            if column not in query1_indexable_columns:
                query_weights[query1][column] = 0

        # 计算 query2 的权重
        for column in query2_indexable_columns:
            for table in self.get_tables():
                if column not in table.columns:
                    continue
                query_weights[query2][column] = (1 - table.get_n_distinct(column)) * table.size_weight

        # 添加 query1 中不存在于 query2 的索引列，并将权重设置为0
        for column in query1_indexable_columns:
            if column not in query2_indexable_columns:
                query_weights[query2][column] = 0

        intersection = sum(min(query_weights[query1][col], query_weights[query2][col]) for col in
                           set(query_weights[query1]) & set(query_weights[query2]))
        union = sum(max(query_weights[query1][col], query_weights[query2][col]) for col in
                    set(query_weights[query1]) & set(query_weights[query2]))
        return intersection / union if union > 0 else 0

    def set_query_benefit(self):
        for num, query in enumerate(self.__queries):
            predicate_sims = self.get_similarity_with_predicate(self.__plans[num])
            benefit_of_similar_queries = 0

            for id, predicate_sim in predicate_sims:
                if id != num:
                    # 获取除自身之外相似的查询
                    column_sim = self.get_similarity_with_indexable_column(self.__queries[id].get_statement(),
                                                                           query.get_statement())
                    propagation = (predicate_sim + column_sim) / 2
                    benefit_of_similar_queries += propagation * self.__query_potential[id]

            self.__query_to_benefit[num] = self.__query_potential[num] + benefit_of_similar_queries

    def get_query_benefit(self) -> dict:
        return self.__query_to_benefit

    def get_m_largest_sum_with_indices(self, threshold=0.8):
        nums_with_indices = list(enumerate(self.get_query_potential()))  # 列表中每个数和其对应的索引
        nums_with_indices_sorted = sorted(nums_with_indices, key=lambda x: x[1], reverse=True)  # 按数值降序排序
        half_max_indices_num = len(nums_with_indices) // 5
        half_max_indices = [x[0] for x in nums_with_indices_sorted[:half_max_indices_num]]
        print('nums_with_indices :', nums_with_indices_sorted)
        total_sum = sum(num for _, num in nums_with_indices_sorted)  # 计算列表中所有数的总和
        print('total_sum :', total_sum)
        target_sum = total_sum * threshold  # 计算80%阈值的目标和
        current_sum = 0
        m = 0
        indices = []

        # 遍历排序后的数值列表，累加直到达到80%的阈值
        for index, num in nums_with_indices_sorted:
            current_sum += num
            m += 1
            indices.append(index)
            if current_sum >= target_sum:
                break
        # 至少返回1/4的query
        if m > half_max_indices_num:
            return m, indices
        else:
            return half_max_indices_num, half_max_indices

    def set_workload_origin_cost(self, executor):
        from index_advisor_workload import calculate_cost
        origin_cost = 0
        for sql in self.get_queries():
            origin_cost += calculate_cost(executor, sql.get_statement(), None)
        self.__origin_cost = origin_cost

    def get_workload_origin_cost(self):
        return self.__origin_cost

    def get_final_state_reward(self, executor, query_list, indexes):
        # 检查缓存中是否有已经计算过的成本和收益
        indexes = sorted(indexes, key=lambda x: (x.get_table(), x.get_columns()))
        query_cost_index = 0
        origin_cost = 0
        for query_num in query_list:
            cache_key = (query_num, tuple(indexes))
            if cache_key in self.__query_index_cost_cache:
                cost_with_indexes = self.__query_index_cost_cache[cache_key]
            else:
                from index_advisor_workload import calculate_cost
                # 计算索引后的成本
                cost_with_indexes = calculate_cost(executor, self.get_queries()[query_num].get_statement(), indexes)

                # 更新缓存
                self.__query_index_cost_cache[cache_key] = cost_with_indexes

            query_cost_index += cost_with_indexes
        origin_cost = sum([self.__queries_origin_cost[x] for x in query_list])

        return origin_cost - query_cost_index

    def set_query_origin_cost(self, queries_cost_list):
        self.__queries_origin_cost = queries_cost_list

    def set_query_potential(self, query_potential):
        for num in range(len(query_potential)):
            self.__query_potential.append(query_potential[num] * self.__queries_origin_cost[num])

    def get_query_potential(self):
        return self.__query_potential

    def get_queries(self) -> List[QueryItem]:
        return self.__queries

    def get_plans(self) -> List[str]:
        return self.__plans

    def has_indexes(self, indexes: Tuple[AdvisedIndex]):
        return indexes in self.__indexes_list

    def get_used_index_names(self):
        used_indexes = set()
        for index_names in self.get_workload_used_indexes(None):
            for index_name in index_names:
                used_indexes.add(index_name)
        return used_indexes

    @lru_cache(maxsize=None)
    def get_workload_used_indexes(self, indexes: (Tuple[AdvisedIndex], None)):
        return list([index_names[self.__indexes_list.index(indexes if indexes else None)]
                     for index_names in self.__index_names_list])

    def get_query_advised_indexes(self, indexes, query):
        query_idx = self.__queries.index(query)
        indexes_idx = self.__indexes_list.index(indexes if indexes else None)
        used_index_names = self.__index_names_list[indexes_idx][query_idx]
        used_advised_indexes = []
        for index in indexes:
            for index_name in used_index_names:
                if index.match(index_name):
                    used_advised_indexes.append(index)
        return used_advised_indexes

    def set_index_benefit(self):
        for indexes in self.__indexes_list:
            if indexes and len(indexes) == 1:
                indexes[0].benefit = self.get_index_benefit(indexes[0])

    def add_table(self, table_context):
        self.__tables.add(table_context)

    def get_tables(self):
        return self.__tables

    def replace_indexes(self, origin, new):
        if not new:
            new = None
        self.__indexes_list[self.__indexes_list.index(origin if origin else None)] = new

    @lru_cache(maxsize=None)
    def get_total_index_cost(self, indexes: (Tuple[AdvisedIndex], None)):
        return sum(
            query_index_cost[self.__indexes_list.index(indexes if indexes else None)] for query_index_cost in
            self.__indexes_costs)

    @lru_cache(maxsize=None)
    def get_total_origin_cost(self):
        return self.get_total_index_cost(None)

    @lru_cache(maxsize=None)
    def get_indexes_benefit(self, indexes: Tuple[AdvisedIndex]):
        return self.get_total_origin_cost() - self.get_total_index_cost(indexes)

    @lru_cache(maxsize=None)
    def get_index_benefit(self, index: AdvisedIndex):
        return self.get_indexes_benefit(tuple([index]))

    @lru_cache(maxsize=None)
    def get_indexes_cost_of_query(self, query: QueryItem, indexes: (Tuple[AdvisedIndex], None)):
        return self.__indexes_costs[self.__queries.index(query)][
            self.__indexes_list.index(indexes if indexes else None)]

    @lru_cache(maxsize=None)
    def get_indexes_plan_of_query(self, query: QueryItem, indexes: (Tuple[AdvisedIndex], None)):
        return self.__plan_list[self.__queries.index(query)][
            self.__indexes_list.index(indexes if indexes else None)]

    @lru_cache(maxsize=None)
    def get_origin_cost_of_query(self, query: QueryItem):
        return self.get_indexes_cost_of_query(query, None)

    @lru_cache(maxsize=None)
    def is_positive_query(self, index: AdvisedIndex, query: QueryItem):
        logging.info(f'index ：{index}，query :{query}')
        logging.info(
            f'self.get_origin_cost_of_query(query ：{self.get_origin_cost_of_query(query)}，self.get_indexes_cost_of_query(query, tuple([index])) :{self.get_indexes_cost_of_query(query, tuple([index]))}')
        return self.get_origin_cost_of_query(query) > self.get_indexes_cost_of_query(query, tuple([index]))

    def add_indexes(self, indexes: (Tuple[AdvisedIndex], None), costs, index_names, plan_list):
        if not indexes:
            indexes = None
        self.__indexes_list.append(indexes)
        if len(costs) != len(self.__queries):
            raise
        for i, cost in enumerate(costs):
            self.__indexes_costs[i].append(cost)
            self.__index_names_list[i].append(index_names[i])
            self.__plan_list[i].append(plan_list[i])

    @lru_cache(maxsize=None)
    def get_index_related_queries(self, index: AdvisedIndex):
        insert_queries = []
        delete_queries = []
        update_queries = []
        select_queries = []
        positive_queries = []
        ineffective_queries = []
        negative_queries = []

        cur_table = index.get_table()
        for query in self.get_queries():
            if cur_table not in query.get_statement().lower() and \
                    not re.search(r'((\A|[\s(,])%s[\s),])' % cur_table.split('.')[-1],
                                  query.get_statement().lower()):
                continue

            if any(re.match(r'(insert\s+into\s+%s\s)' % table, query.get_statement().lower())
                   for table in [cur_table, cur_table.split('.')[-1]]):
                insert_queries.append(query)
                if not self.is_positive_query(index, query):
                    negative_queries.append(query)
            elif any(re.match(r'(delete\s+from\s+%s\s)' % table, query.get_statement().lower()) or
                     re.match(r'(delete\s+%s\s)' % table, query.get_statement().lower())
                     for table in [cur_table, cur_table.split('.')[-1]]):
                delete_queries.append(query)
                if not self.is_positive_query(index, query):
                    negative_queries.append(query)
            elif any(re.match(r'(update\s+%s\s)' % table, query.get_statement().lower())
                     for table in [cur_table, cur_table.split('.')[-1]]):
                update_queries.append(query)
                if not self.is_positive_query(index, query):
                    negative_queries.append(query)
            else:
                select_queries.append(query)
                if not self.is_positive_query(index, query):
                    ineffective_queries.append(query)
            positive_queries = [query for query in insert_queries + delete_queries + update_queries + select_queries
                                if query not in negative_queries + ineffective_queries]
        return insert_queries, delete_queries, update_queries, select_queries, \
               positive_queries, ineffective_queries, negative_queries

    @lru_cache(maxsize=None)
    def get_index_sql_num(self, index: AdvisedIndex):
        insert_queries, delete_queries, update_queries, \
        select_queries, positive_queries, ineffective_queries, \
        negative_queries = self.get_index_related_queries(index)
        insert_sql_num = sum(query.get_frequency() for query in insert_queries)
        delete_sql_num = sum(query.get_frequency() for query in delete_queries)
        update_sql_num = sum(query.get_frequency() for query in update_queries)
        select_sql_num = sum(query.get_frequency() for query in select_queries)
        positive_sql_num = sum(query.get_frequency() for query in positive_queries)
        ineffective_sql_num = sum(query.get_frequency() for query in ineffective_queries)
        negative_sql_num = sum(query.get_frequency() for query in negative_queries)
        return {'insert': insert_sql_num, 'delete': delete_sql_num, 'update': update_sql_num, 'select': select_sql_num,
                'positive': positive_sql_num, 'ineffective': ineffective_sql_num, 'negative': negative_sql_num}


def get_statement_count(queries: List[QueryItem]):
    return int(sum(query.get_frequency() for query in queries))


def is_subset_index(indexes1: Tuple[AdvisedIndex], indexes2: Tuple[AdvisedIndex]):
    existing = False
    if len(indexes1) > len(indexes2):
        return existing
    for index1 in indexes1:
        existing = False
        for index2 in indexes2:
            # Example indexes1: [table1 col1 global] belong to indexes2:[table1 col1, col2 global].
            if index2.get_table() == index1.get_table() \
                    and match_columns(index1.get_columns(), index2.get_columns()) \
                    and index2.get_index_type() == index1.get_index_type():
                existing = True
                break
        if not existing:
            break
    return existing


def lookfor_subsets_configs(config: List[AdvisedIndex], atomic_config_total: List[Tuple[AdvisedIndex]]):
    """ Look for the subsets of a given config in the atomic configs. """
    contained_atomic_configs = []
    for atomic_config in atomic_config_total:
        if len(atomic_config) == 1:
            continue
        if not is_subset_index(atomic_config, tuple(config)):
            continue
        # Atomic_config should contain the latest candidate_index.
        if not any(is_subset_index((atomic_index,), (config[-1],)) for atomic_index in atomic_config):
            continue
        # Filter redundant config in contained_atomic_configs.
        for contained_atomic_config in contained_atomic_configs[:]:
            if is_subset_index(contained_atomic_config, atomic_config):
                contained_atomic_configs.remove(contained_atomic_config)

        contained_atomic_configs.append(atomic_config)

    return contained_atomic_configs


def match_columns(column1, column2):
    return re.match(column1 + ',', column2 + ',')


def infer_workload_benefit(workload: WorkLoad, config: List[AdvisedIndex],
                           atomic_config_total: List[Tuple[AdvisedIndex]]):
    """ Infer the most important queries for a config according to the model1 """
    total_benefit = 0
    atomic_subsets_configs = lookfor_subsets_configs(config, atomic_config_total)  # 查找给定配置中的子集---是否在原子配置列表中存在
    is_recorded = [True] * len(atomic_subsets_configs)
    for query in workload.get_queries():
        origin_cost_of_query = workload.get_origin_cost_of_query(query)
        if origin_cost_of_query == 0:
            continue
        # When there are multiple indexes, the benefit is the total benefit
        # of the multiple indexes minus the benefit of every single index.
        total_benefit += \
            origin_cost_of_query - workload.get_indexes_cost_of_query(query, (config[-1],))
        for k, sub_config in enumerate(atomic_subsets_configs):
            single_index_total_benefit = sum(origin_cost_of_query -
                                             workload.get_indexes_cost_of_query(query, (index,))
                                             for index in sub_config)
            portfolio_returns = \
                origin_cost_of_query \
                - workload.get_indexes_cost_of_query(query, sub_config) \
                - single_index_total_benefit
            total_benefit += portfolio_returns
            if portfolio_returns / origin_cost_of_query <= 0.01:
                continue
            # Record the portfolio returns of the index.
            association_indexes = ';'.join([str(index) for index in sub_config])
            association_benefit = (query.get_statement(), portfolio_returns / origin_cost_of_query)
            if association_indexes not in config[-1].association_indexes:
                is_recorded[k] = False
                config[-1].set_association_indexes(association_indexes, association_benefit)
                continue
            if not is_recorded[k]:
                config[-1].set_association_indexes(association_indexes, association_benefit)

    return total_benefit


@lru_cache(maxsize=None)
def get_tokens(query):
    return list(sqlparse.parse(query)[0].flatten())


@lru_cache(maxsize=None)
def has_dollar_placeholder(query):
    tokens = get_tokens(query)
    return any(item.ttype is Name.Placeholder for item in tokens)


@lru_cache(maxsize=None)
def get_placeholders(query):
    placeholders = set()
    for item in get_tokens(query):
        if item.ttype is Name.Placeholder:
            placeholders.add(item.value)
    return placeholders


@lru_cache(maxsize=None)
def generate_placeholder_indexes(table_cxt, column):
    indexes = []
    schema_table = f'{table_cxt.schema}.{table_cxt.table}'
    if table_cxt.is_partitioned_table:
        indexes.append(IndexItemFactory().get_index(schema_table, column, 'global'))
        indexes.append(IndexItemFactory().get_index(schema_table, column, 'local'))
    else:
        indexes.append(IndexItemFactory().get_index(schema_table, column, ''))
    return indexes


def replace_comma_with_dollar(query):
    """
    Replacing '?' with '$+Numbers' in SQL:
      input: UPDATE bmsql_customer SET c_balance = c_balance + $1, c_delivery_cnt = c_delivery_cnt + ?
      WHERE c_w_id = $2 AND c_d_id = $3 AND c_id = $4 and c_info = ?;
      output: UPDATE bmsql_customer SET c_balance = c_balance + $1, c_delivery_cnt = c_delivery_cnt + $5
      WHERE c_w_id = $2 AND c_d_id = $3 AND c_id = $4 and c_info = $6;
    note: if track_stmt_parameter is off, all '?' in SQL need to be replaced
    """
    if '?' not in query:
        return query
    max_dollar_number = 0
    dollar_parts = re.findall(r'(\$\d+)', query)
    if dollar_parts:
        max_dollar_number = max(int(item.strip('$')) for item in dollar_parts)
    while '?' in query:
        dollar = "$%s" % (max_dollar_number + 1)
        query = query.replace('?', dollar, 1)
        max_dollar_number += 1
    return query


@lru_cache(maxsize=None)
def is_multi_node(executor):
    # sql = "select pg_catalog.count(*) from pgxc_node where node_type='C';"
    # for cur_tuple in executor.execute_sqls([sql]):
    #     if str(cur_tuple[0]).isdigit():
    #         return int(cur_tuple[0]) > 0
    return False


@contextmanager
def hypo_index_ctx(executor):
    yield
    executor.execute_sqls(['SELECT hypopg_reset_index();'])


def split_integer(m, n):
    quotient = int(m / n)
    remainder = m % n
    if m < n:
        return [1] * m
    if remainder > 0:
        return [quotient] * (n - remainder) + [quotient + 1] * remainder
    if remainder < 0:
        return [quotient - 1] * -remainder + [quotient] * (n + remainder)
    return [quotient] * n


def split_iter(iterable, n):
    size_list = split_integer(len(iterable), n)
    index = 0
    res = []
    for size in size_list:
        res.append(iterable[index:index + size])
        index += size
    return res


def flatten(iterable):
    for _iter in iterable:
        if hasattr(_iter, '__iter__') and not isinstance(_iter, str):
            for item in flatten(_iter):
                yield item
        else:
            yield _iter
