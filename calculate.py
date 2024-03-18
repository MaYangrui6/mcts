from .sql_generator import get_index_check_sqls
import re

def parse_plan_cost(line):
    """ Parse the explain plan to get the estimated cost by database optimizer. """
    cost = -1
    # like "Limit  (cost=19932.04..19933.29 rows=100 width=17)"
    pattern = re.compile(r'\(cost=([^)]*)\)', re.S)
    matched_res = re.search(pattern, line)
    if matched_res and len(matched_res.group(1).split()) == 3:
        _cost, _rows, _width = matched_res.group(1).split()
        # like cost=19932.04..19933.29
        cost = float(_cost.split('..')[-1])
    return cost

class Calculate:
    def __init__(self, executor,query,indexes):
        self.executor=executor
        self.query=query
        self.indexes=indexes
    def calculate_cost(self):
        exe_sqls, _, _ = get_index_check_sqls(self.query, self.indexes, False)  # 这个query创建的hypopg与其他query不交叉
        index_check_results = self.executor.execute_sqls(exe_sqls)
        cost = None
        for res in index_check_results:
            if '(cost' in res[0]:
                cost = parse_plan_cost(res[0])
                break
        return cost
