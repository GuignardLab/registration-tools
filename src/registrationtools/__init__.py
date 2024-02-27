__version__ = "0.4.3"

from .utils import (
    get_paths,
    dimensions,
    get_channels_name,
    reference_channel,
    sort_by_channels_and_timepoints,
    cut_timesequence,
    get_voxel_sizes,
    get_trsf_type,
    data_preparation,
    run_registration,
    save_sequences_as_stacks,
    save_jsonfile,
)
import sys

if 8 < sys.version_info.minor:
    import importlib.resources as importlib_resources

    pkg = importlib_resources.files("registrationtools") / "data"
else:
    from importlib_resources._legacy import path

    pkg = path("registrationtools", "data").args[0]

image_path = pkg / "images"
json_path_spatial = pkg / "JSON_spatial"
json_path_time = pkg / "JSON_time"

__all__ = [
    "get_paths",
    "dimensions",
    "get_channels_name",
    "reference_channel",
    "sort_by_channels_and_timepoints",
    "cut_timesequence",
    "get_voxel_sizes",
    "get_trsf_type",
    "data_preparation",
    "run_registration",
    "save_sequences_as_stacks",
    "save_jsonfile",
    "image_path",
    "json_path_spatial",
    "json_path_time",
]
