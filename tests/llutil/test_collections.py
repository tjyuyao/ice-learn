import dill
from ice.llutil.collections import Counter, Dict, ConfigDict

def test_dict_pickable():
    i = Dict(a=1, b="2")
    i.__reduce__()
    buf = dill.dumps(i)
    _ = dill.loads(buf)
    
def test_counter_pickable():
    i = Counter()
    i.__reduce__()
    buf = dill.dumps(i)
    _ = dill.loads(buf)