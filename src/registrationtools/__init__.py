__version__ = "0.4.0"

from .time_registration import TimeRegistration, time_registration
from .spatial_registration import SpatialRegistration, spatial_registration
import sys

if 8<sys.version_info.minor:
    import importlib.resources as importlib_resources

    pkg = importlib_resources.files("registrationtools") / "data"
else:
    from importlib_resources._legacy import path

    pkg = path("registrationtools", "data").args[0]

image_path = pkg / "images"
json_path_spatial = pkg / "JSON_spatial"
json_path_time = pkg / "JSON_time"
