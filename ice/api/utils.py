from ice.llutil.launcher.launcher import _parse_devices_and_backend

def parse_devices(devices:str):
    return _parse_devices_and_backend(devices)[0]