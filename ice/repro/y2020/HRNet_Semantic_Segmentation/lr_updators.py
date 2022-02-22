#%%

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

@dictprocess
def PolyWarmUp(param_group, trigger, epochs, steps, *, power=1., min_lr=0., max_updates:int):
    if trigger == "epoch_start":
        progress = epochs / max_updates
    elif trigger == "step":
        progress = steps / max_updates
    else:
        assert False
    if progress > 1: return
    coeff = (1 - progress)**power
    lr = (param_group["initial_lr"] - min_lr) * coeff + min_lr
    if progress < 0.01:
        lr = lr * progress * 100
    param_group["lr"] = lr

#%%
if __name__ == "__main__":
    state_dict = {"param_group":{"initial_lr": 1e-2}}
    max_updates = 40000
    poly = Poly(trigger="step", epochs=0, power=3., min_lr=1e-4, max_updates=max_updates)
    lr = [None] * max_updates
    for steps in range(max_updates):
        state_dict["steps"] = steps
        lr[steps] = poly(state_dict)["param_group"]["lr"]

    import matplotlib.pyplot as plt
    import matplotlib as mpl
    import numpy as np
    plt.plot(np.arange(max_updates), np.array(lr))
    plt.show()
# %%
