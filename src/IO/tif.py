# -*- python -*-
#
#       adapted from C. Pradal
#
#       Copyright 2011 INRIA - CIRAD - INRA
#
#       File author(s): Christophe Pradal <christophe.pradal@cirad.fr>
#                       Daniel Barbeau    <daniel.barbeau@cirad.fr>
#
#       Distributed under the Cecill-C License.
#       See accompanying file LICENSE.txt or copy at
#           http://www.cecill.info/licences/Licence_CeCILL-C_V1-en.html
#
#       OpenAlea WebSite : http://openalea.gforge.inria.fr
################################################################################
"""
This module reads 3D tiff format
"""


__license__ = "Cecill-C"
__revision__ = " $Id$ "

import numpy as np
from .spatial_image import SpatialImage


__all__ = []

import decimal

import numpy as np
from tifffile import TiffFile, imread, imwrite
import os, os.path, sys, time

__all__ += ["read_tif", "write_tif", "mantissa"]


def read_tif(filename, channel=0):
    """Read a tif image

    :Parameters:
    - `filename` (str) - name of the file to read
    """

    def format_digit(v):
        try:
            v = eval(v)
        except Exception as e:
            pass
        return v

    with TiffFile(filename) as tif:
        if "ImageDescription" in tif.pages[0].tags:
            description = (
                tif.pages[0].tags["ImageDescription"].value.split("\n")
            )
            separator = set.intersection(*[set(k) for k in description]).pop()
            info_dict = {
                v.split(separator)[0]: format_digit(v.split(separator)[1])
                for v in description
            }
        else:
            info_dict = {}

        if "XResolution" in tif.pages[0].tags:
            vx = tif.pages[0].tags["XResolution"].value
            if vx[0] != 0:
                vx = vx[1] / vx[0]
                if isinstance(vx, list):
                    vx = vx[1] / vx[0]
            else:
                vx = 1.0
        else:
            vx = 1.0

        if "YResolution" in tif.pages[0].tags:
            vy = tif.pages[0].tags["YResolution"].value
            if vy[0] != 0:
                vy = vy[1] / vy[0]
                if isinstance(vy, list):
                    vy = vy[1] / vy[0]
            else:
                vy = 1.0
        else:
            vy = 1.0

        if "ZResolution" in tif.pages[0].tags:
            vz = tif.pages[0].tags["ZResolution"].value
            if vz[0] != 0:
                if isinstance(vz, list):
                    vz = vz[1] / vz[0]
            else:
                vz = 1.0
        elif "spacing" in info_dict:
            vz = info_dict["spacing"]
        else:
            vz = 1.0

        im = tif.asarray()
        if len(im.shape) == 3:
            im = np.transpose(im, (2, 1, 0))
        elif len(im.shape) == 2:
            im = np.transpose(im, (1, 0))
        if 3 <= len(im.shape):
            im = SpatialImage(im, (vx, vy, vz))
        else:
            print(im.shape, vx, vy)
            im = SpatialImage(im, (vx, vy))
        im.resolution = (vx, vy, vz)
    return im


def mantissa(value):
    """Convert value to [number, divisor] where divisor is power of 10"""
    # -- surely not the nicest thing around --
    d = decimal.Decimal(str(value))  # -- lovely...
    sign, digits, exp = d.as_tuple()
    n_digits = len(digits)
    dividend = int(
        sum(v * (10 ** (n_digits - 1 - i)) for i, v in enumerate(digits))
        * (1 if sign == 0 else -1)
    )
    divisor = int(10**-exp)
    return dividend, divisor


def write_tif(filename, obj):
    if len(obj.shape) > 3:
        raise IOError(
            "Vectorial images are currently unsupported by tif writer"
        )
    is3D = len(obj.shape) == 3

    if hasattr(obj, "resolution"):
        res = obj.resolution
    elif hasattr(obj, "voxelsize"):
        res = obj.resolution
    else:
        res = (1, 1, 1) if is3D else (1, 1)

    vx = 1.0 / res[0]
    vy = 1.0 / res[1]
    if is3D:
        spacing = res[2]
        metadata = {"spacing": spacing, "axes": "ZYX"}
    else:
        metadata = {}
    extra_info = {
        "XResolution": res[0],
        "YResolution": res[1],
        "spacing": spacing
        if is3D
        else None,  # : no way to save the spacing (no specific tag)
    }
    print(extra_info)

    if obj.dtype.char in "BHhf":
        return imwrite(
            filename,
            obj.T,
            imagej=True,
            resolution=(vx, vy),
            metadata=metadata,
        )
    else:
        return imwrite(filename, obj.T)
