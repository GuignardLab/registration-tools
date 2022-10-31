from .time_registration import TimeRegistration
try:
    import importlib.resources as importlib_resources
    pkg = importlib_resources.files('registrationtools')/'data'
except Exception as e:
    from importlib_resources._legacy import path
    pkg = path('regsitrationtools', 'data').args[0]

image_path = pkg/'images'
json_path = pkg/'JSON'