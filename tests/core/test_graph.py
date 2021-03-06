import pytest
from torch import device
from ice.core.graph import InvalidURIError, Node, GraphOutputCache
from ice.core.hypergraph import HyperGraph, Task
from ice.llutil.collections import ConfigDict

def assertTrue(expr, msg):
    assert expr, msg

class SimpleNode(Node):
    
    def __freeze__(self) -> None:
        return super().__freeze__(value=0)
    
    def forward_impl(self, graph:GraphOutputCache):
        self.value += 1
        return self.value
    
    def backward(self):
        self.value += 10
        
    def update(self):
        self.value += 100


_C = ConfigDict()
_C.GRAPH.G1 = HyperGraph()
_C.GRAPH.G1.add("n1", SimpleNode())
_C.GRAPH.G1.add("n2", Node(forward=lambda n, x: x['n1'] * 2))

def test_simple_node():
    assert isinstance(_C.GRAPH.G1, HyperGraph)
    _C.GRAPH.G1.run(
        [
            Task(train=True, steps=1),
            lambda g: assertTrue(g["*/n1"].value == 111, f"test_simple_node: expecting 111, get {g['*/n1'].value}")
        ],
        devices="cpu"
    )


def test_two_nodes():
    _C.GRAPH.G1.run(
        [
            Task(train=True, steps=2),
            lambda g: assertTrue(g["*/n2"].forward() == (111+1)*2, f"test_two_nodes: expecting 224, get {g['*/n2'].forward()}")
        ],
        devices="cpu"
    )


def test_parse_uri():
    HyperGraph._parse_uri("node_name") == ("*/", "node_name")
    HyperGraph._parse_uri("*/n1") == ("*/", "n1")