import ice
import subprocess

def repr_args():
    print(ice.args, end="")

def test_cmdline():
    tests = [
        (f'python {__file__} kernel_size="(3,3)" batch_size=4 path/to/cfg.yml',
         b'args: kernel_size=\'(3,3)\' batch_size=4 path/to/cfg.yml'),
    ]
    for cmd, out in tests:
        assert out == subprocess.check_output(cmd, shell=True)

def test_setget():
    ice.args.parse_args(["2", "k1=4"])
    assert len(ice.args) == 2
    assert 2 == ice.args.get(0, int, 4)
    assert 4 == ice.args.get("k1", int, 8)
    assert 4 == int(ice.args["k1"])
    assert 4 == int(ice.args.k1)
    ice.args.setdefault("k2", 8)
    
    assert 8 == int(ice.args.k2)

    ice.args.setdefault("k1", 8)
    assert 4 == int(ice.args.k1)

    del ice.args["k1"]
    assert "k1" not in ice.args
    ice.args.setdefault("k1", 8)
    assert "k1" in ice.args
    assert 8 == int(ice.args.k1)

    ice.args.update(k2=0)
    ice.args.update({0: 0})
    assert 0 == ice.args.get(0, int, 4)
    assert 0 == ice.args.get("k2", int, 4)
    

if __name__ == "__main__":
    repr_args()