import ice

def test_dictprocess():
    @ice.dictprocess
    def Sub1(x, y): return {"z": x-y}

    @ice.dictprocess
    def Sub2(x, y): return

    @ice.dictprocess
    def Sub3(x, y): return x-y

    @ice.dictprocess
    def Sub4(x, y): return {"z": x-y, "w": x + y}

    @ice.dictprocess
    def Sub5(x, y): return x - y, x + y

    assert Sub1(x=2, y=1)({"any": 0}) == {"any": 0, 'z': 1}, "test 1: dst not specified, original return is a dict, update the state dict."
    assert Sub2(x=2, y=1)({"any": 0}) == {"any": 0}, "test 2: dst not specified, original return is None, do nothing to the state dict."
    assert Sub3(x=2, y=1)({"any": 0}) == 1, "test 3: dst not specified, original return is a value, behavior as calling the original function."
    assert Sub1(x=2, y=1, dst="z")({}) == {'z': 1}, "test 4: dst specified, original return is a dict of which dst is a valid key, update corresponding value."
    assert Sub3(x=2, y=1, dst="z1")({}) == {'z1': 1}, "test 5: dst specified, original return is a value, update using dst as the key."
    assert Sub4(x=2, y=1, dst="z1")() == {'z1': {'z': 1, 'w': 3}}, "test 5: dst specified, original return is a dict of which dst is not a valid key, update using dst as the key and the entire dict as the value."
    assert Sub4(x=2, y=1, dst=["w"])() == {'w': 3}, "test 6"
    assert Sub5(dst=["z", "w"])(dict(x=2, y=1)) == {'x': 2, 'y': 1, 'z': 1, 'w': 3}, "test 7"
    assert Sub4(dst={"z1":"z", "w1":"w"})(dict(x=2, y=1)) == {'x': 2, 'y': 1, 'z1': 1, 'w1': 3}, "test 8"
    assert Sub1(x=2, y=1, dst="z1")() == {'z1': 1}, "test 10"