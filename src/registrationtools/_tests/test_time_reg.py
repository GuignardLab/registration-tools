from registrationtools import TimeRegistration, image_path, json_path
from pathlib import Path
from shutil import rmtree


def test_registration():
    tr = TimeRegistration(str(json_path))
    for p in tr.params:
        p.add_path_prefix(str(image_path))
    tr.run_trsf()
    for file in image_path.iterdir():
        if file.name[0] == "_":
            if file.is_dir():
                rmtree(file)
            else:
                file.unlink()
