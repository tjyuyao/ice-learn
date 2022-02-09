from ice.llutil.dictprocess import dictprocess


@dictprocess
def Poly(param_group, by_epoch, epochs, steps, *, power=1., min_lr=0., max_updates:int):
    if by_epoch:
        progress = epochs / max_updates
    else:
        progress = steps / max_updates
    if progress > 1: return
    coeff = (1 - progress)**power
    param_group["lr"] = (param_group["initial_lr"] - min_lr) * coeff + min_lr
