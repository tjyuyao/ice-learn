
# Common tips for debugging ICE programs

## Debug by print

## Debug by breakpoint

set device to cuda or cpu at first

debugging ddp:

```python
if torch.cuda.current_device() == 0:
    breakpoint()
```
