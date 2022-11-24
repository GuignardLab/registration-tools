__version__ = "0.1.1"

from .time_registration import TimeRegistration, time_registration
from .spatial_registration import SpatialRegistration, spatial_registration

try:
    import importlib.resources as importlib_resources

    pkg = importlib_resources.files("registrationtools") / "data"
except Exception as e:
    from importlib_resources._legacy import path

    pkg = path("regsitrationtools", "data").args[0]

image_path = pkg / "images"
json_path_spatial = pkg / "JSON_spatial"
json_path_time = pkg / "JSON_time"
