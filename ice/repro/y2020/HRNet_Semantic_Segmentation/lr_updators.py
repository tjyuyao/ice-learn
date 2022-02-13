from ice.llutil.dictprocess import dictprocess


@dictprocess
def Poly(param_group, trigger, epochs, steps, *, power=1., min_lr=0., max_updates:int):
    if trigger == "epoch_start":
        progress = epochs / max_updates
    elif trigger == "step":
        progress = steps / max_updates
    else:
        assert False
    if progress > 1: return
    coeff = (1 - progress)**power
    param_group["lr"] = (param_group["initial_lr"] - min_lr) * coeff + min_lr
