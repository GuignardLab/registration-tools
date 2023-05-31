#!/usr/bin/python
# This file is subject to the terms and conditions defined in
# file 'LICENCE', which is part of this source code package.
# Author: Leo Guignard (leo.guignard...@AT@...univ-amu.fr)

from pathlib import Path
from time import time
import os
from subprocess import call
import scipy as sp
from scipy import interpolate
import json
import numpy as np
from IO import imread, imsave, SpatialImage
from typing import List, Tuple
from statsmodels.nonparametric.smoothers_lowess import lowess
import sys
import xml.etree.ElementTree as ET
from xml.dom import minidom
from transforms3d.affines import decompose
from transforms3d.euler import mat2euler

if sys.version_info[0] < 3:
    from future.builtins import input

try:
    from pyklb import readheader

    pyklb_found = True
except Exception as e:
    pyklb_found = False
    print("pyklb library not found, klb files will not be generated")


class trsf_parameters(object):
    """
    Read parameters for the registration function from a preformated json file
    """

    def check_parameters_consistancy(self) -> bool:
        """
        Function that should check parameter consistancy
        """
        return True

    def __str__(self) -> str:
        max_key = (
            max([len(k) for k in self.__dict__.keys() if k != "param_dict"])
            + 1
        )
        max_tot = (
            max(
                [
                    len(str(v))
                    for k, v in self.__dict__.items()
                    if k != "param_dict"
                ]
            )
            + 2
            + max_key
        )
        output = "The registration will run with the following arguments:\n"
        output += "\n" + " File format ".center(max_tot, "-") + "\n"
        output += "path_to_data".ljust(max_key, " ") + ": {}\n".format(
            self.path_to_data
        )
        output += "file_name".ljust(max_key, " ") + ": {}\n".format(
            self.file_name
        )
        output += "trsf_folder".ljust(max_key, " ") + ": {}\n".format(
            self.trsf_folder
        )
        output += "output_format".ljust(max_key, " ") + ": {}\n".format(
            self.output_format
        )
        output += "check_TP".ljust(max_key, " ") + ": {:d}\n".format(
            self.check_TP
        )
        output += "\n" + " Time series properties ".center(max_tot, "-") + "\n"
        output += "voxel_size".ljust(
            max_key, " "
        ) + ": {:f}x{:f}x{:f}\n".format(*self.voxel_size)
        output += "first".ljust(max_key, " ") + ": {:d}\n".format(self.first)
        output += "last".ljust(max_key, " ") + ": {:d}\n".format(self.last)
        output += "\n" + " Registration ".center(max_tot, "-") + "\n"
        output += "compute_trsf".ljust(max_key, " ") + ": {:d}\n".format(
            self.compute_trsf
        )
        if self.compute_trsf:
            if self.ref_path is not None:
                output += "ref_path".ljust(max_key, " ") + ": {}\n".format(
                    self.ref_path
                )
            output += "trsf_type".ljust(max_key, " ") + ": {}\n".format(
                self.trsf_type
            )
            if self.trsf_type == "vectorfield":
                output += "sigma".ljust(max_key, " ") + ": {:.2f}\n".format(
                    self.sigma
                )
            output += "recompute".ljust(max_key, " ") + ": {:d}\n".format(
                self.recompute
            )
        output += "apply_trsf".ljust(max_key, " ") + ": {:d}\n".format(
            self.apply_trsf
        )

        return output

    def add_path_prefix(self, prefix: str):
        """
        Add a prefix to all the relevant paths (this is mainly for the unitary tests)

        Args:
            prefix (str): The prefix to add in front of the path
        """
        self.projection_path = (
            os.path.join(prefix, self.projection_path) + os.path.sep
        )
        self.path_to_data = (
            os.path.join(prefix, self.path_to_data) + os.path.sep
        )
        self.trsf_folder = os.path.join(prefix, self.trsf_folder) + os.path.sep
        self.output_format = (
            os.path.join(prefix, self.output_format) + os.path.sep
        )

    def __init__(self, file_name: str):
        if not isinstance(file_name, dict):
            with open(file_name) as f:
                param_dict = json.load(f)
                f.close()
        else:
            param_dict = {}
            for k, v in file_name.items():
                if isinstance(v, Path):
                    param_dict[k] = str(v)
                else:
                    param_dict[k] = v

        # Default parameters
        self.check_TP = None
        self.not_to_do = []
        self.compute_trsf = True
        self.ref_path = None
        self.padding = 1
        self.lowess = False
        self.window_size = 5
        self.step_size = 100
        self.recompute = True
        self.apply_trsf = True
        self.sigma = 2.0
        self.keep_vectorfield = False
        self.trsf_type = "rigid"
        self.image_interpolation = "linear"
        self.path_to_bin = ""
        self.spline = 1
        self.trsf_interpolation = False
        self.sequential = True
        self.time_tag = "TM"
        self.do_bdv = 0
        self.bdv_voxel_size = None
        self.bdv_unit = "microns"
        self.pre_2D = False
        self.low_th = None
        self.plot_trsf = False

        self.param_dict = param_dict
        if "registration_depth" in param_dict:
            self.__dict__["registration_depth_start"] = 6
            self.__dict__["registration_depth_end"] = param_dict[
                "registration_depth"
            ]
        elif not "registration_depth_start" in param_dict:
            self.__dict__["registration_depth_start"] = 6
            self.__dict__["registration_depth_end"] = 3

        self.__dict__.update(param_dict)
        self.voxel_size = tuple(self.voxel_size)
        if not hasattr(self, "voxel_size_out"):
            self.voxel_size_out = self.voxel_size
        else:
            self.voxel_size_out = tuple(self.voxel_size_out)
        self.origin_file_name = file_name
        self.path_to_data = os.path.join(self.path_to_data, "")
        self.trsf_folder = os.path.join(self.trsf_folder, "")
        self.path_to_bin = os.path.join(self.path_to_bin, "")
        if 0 < len(self.path_to_bin) and not os.path.exists(self.path_to_bin):
            print("Binary path could not be found, will try with global call")
            self.path_to_bin = ""


class VectorFlow:
    @staticmethod
    def read_trsf(path: str) -> np.ndarray:
        """
        Read a transformation from a text file

        Args:
            path (str): path to a transformation

        Returns
            (np.ndarray): 4x4 ndarray matrix
        """
        f = open(path)
        if f.read()[0] == "(":
            f.close()
            f = open(path)
            lines = f.readlines()[2:-1]
            f.close()
            return np.array([[float(v) for v in l.split()] for l in lines])
        f.close()
        return np.loadtxt(path)

    def __produce_trsf(self, params: Tuple[trsf_parameters, int, int, bool]):
        """
        Given a parameter object a reference time and two time points
        compute the transformation that registers together two consecutive in time images.

        Args:
            params (tuple): a tuple with the parameter object, two consecutive time points
                to register and a boolean to detemine wether to apply the trsf or not
        """
        (p, t1, t2, make) = params
        if p.forward:
            t_ref = max(t1, t2)
            t_flo = min(t1, t2)
        else:
            t_ref = min(t1, t2)
            t_flo = max(t1, t2)
        p_im_ref = p.A0.format(t=t_ref)
        p_im_flo = p.A0.format(t=t_flo)
        if t_flo != t_ref and (
            p.recompute
            or not os.path.exists(
                os.path.join(p.trsf_folder, "t%06d-%06d.tif" % (t_flo, t_ref))
            )
        ):
            if p.low_th is not None and 0 < p.low_th:
                th = " -ref-lt {lt:f} -flo-lt {lt:f} -no-norma ".format(
                    lt=p.low_th
                )
            else:
                th = ""
            if p.apply_trsf:
                res = " -res " + p.A0_out.format(t=t_flo)
            else:
                res = ""
            if p.keep_vectorfield:
                res_trsf = (
                    " -no-composition-with-left -res-trsf "
                    + os.path.join(
                        p.trsf_folder, "t%06d-%06d.tif" % (t_flo, t_ref)
                    )
                )
            if p.pre_registration:
                call(
                    self.path_to_bin
                    + "blockmatching -ref "
                    + p_im_ref
                    + " -flo "
                    + p_im_flo
                    + " -reference-voxel %f %f %f" % p.voxel_size
                    + " -floating-voxel %f %f %f" % p.voxel_size
                    + " -trsf-type affine -py-hl %d -py-ll %d"
                    % (p.registration_depth_start, p.registration_depth_end)
                    + " -res-trsf "
                    + os.path.join(
                        p.trsf_folder, "init-t%06d-%06d.txt" % (t_flo, t_ref)
                    )
                    + th,
                    shell=True,
                )
                init_trsf = " -init-trsf " + os.path.join(
                    p.trsf_folder, "init-t%06d-%06d.txt" % (t_flo, t_ref)
                )
            else:
                init_trsf = ""
            call(
                self.path_to_bin
                + "blockmatching -ref "
                + p_im_ref
                + " -flo "
                + p_im_flo
                + init_trsf
                + res
                + " -reference-voxel %f %f %f" % p.voxel_size
                + " -floating-voxel %f %f %f" % p.voxel_size
                + " -trsf-type vectorfield -py-hl %d -py-ll %d"
                % (
                    p.registration_depth_start,
                    p.registration_depth_end,
                )
                + res_trsf
                + (
                    " -elastic-sigma {s:.1f} {s:.1f} {s:.1f} "
                    + " -fluid-sigma {s:.1f} {s:.1f} {s:.1f}"
                ).format(s=p.sigma)
                + th,
                shell=True,
            )

    def run_produce_trsf(self, p: trsf_parameters):
        """
        Parallel processing of the transformations from t to t-1/t+1 to t (depending on t<r)
        The transformation is computed using blockmatching algorithm

        Args:
            p (trsf_paraneters): a trsf_parameter object
        """
        mapping = [
            (p, t1, t2, not (t1 in p.not_to_do or t2 in p.not_to_do))
            for t1, t2 in zip(p.to_register[:-1], p.to_register[1:])
        ]
        tic = time()
        for mi in mapping:
            self.__produce_trsf(mi)
        tac = time()
        whole_time = tac - tic
        secs = whole_time % 60
        whole_time = whole_time // 60
        mins = whole_time % 60
        hours = whole_time // 60
        print("%dh:%dmin:%ds" % (hours, mins, secs))

    @classmethod
    def read_param_file(clf, p_param: str = None) -> List[trsf_parameters]:
        """
        Asks for, reads and formats the parameter file

        Args:
            p_params (str | None): path to the json parameter files.
                It can either be a json or a folder containing json files.
                If it is None (default), a path is asked to the user.

        Returns:
            (list): list of `trsf_parameters` objects
        """
        if not isinstance(p_param, dict):
            if p_param is None:
                if len(sys.argv) < 2 or sys.argv[1] == "-f":
                    p_param = input(
                        "\nPlease inform the path to the json config file:\n"
                    )
                else:
                    p_param = sys.argv[1]
            stable = False or isinstance(p_param, Path)
            while not stable:
                tmp = p_param.strip('"').strip("'").strip(" ")
                stable = tmp == p_param
                p_param = tmp
            if os.path.isdir(p_param):
                f_names = [
                    os.path.join(p_param, f)
                    for f in os.listdir(p_param)
                    if ".json" in f and not "~" in f
                ]
            else:
                f_names = [p_param]
        else:
            f_names = [p_param]
        params = []
        for file_name in f_names:
            if isinstance(file_name, str):
                print("")
                print("Extraction of the parameters from file %s" % file_name)
            p = trsf_parameters(file_name)
            if not p.check_parameters_consistancy():
                print("\n%s Failed the consistancy check, it will be skipped")
            else:
                params += [p]
            print("")
        return params

    @staticmethod
    def prepare_paths(p: trsf_parameters):
        """
        Prepare the paths in a format that is usable by the algorithm

        Args:
            p (trsf_parameters): parameter object
        """
        image_formats = [
            ".tiff",
            ".tif",
            ".inr",
            ".gz",
            ".klb",
            ".h5",
            ".hdf5",
        ]
        ### Check the file names and folders:
        p.im_ext = p.file_name.split(".")[-1]  # Image type
        p.A0 = os.path.join(p.path_to_data, p.file_name)  # Image path
        # Output format
        if p.output_format is not None:
            # If the output format is a file alone
            if os.path.split(p.output_format)[0] == "":
                p.A0_out = os.path.join(p.path_to_data, p.output_format)
            # If the output format is a folder alone
            elif not os.path.splitext(p.output_format)[-1] in image_formats:
                p.A0_out = os.path.join(p.output_format, p.file_name)
            else:
                p.A0_out = p.output_format
        else:
            p.A0_out = os.path.join(
                p.path_to_data,
                p.file_name.replace(p.im_ext, p.suffix + "." + p.im_ext),
            )

        # Time points to work with
        p.time_points = np.array(
            [i for i in np.arange(p.first, p.last + 1) if not i in p.not_to_do]
        )
        if p.apply_trsf:
            for t in sorted(p.time_points):
                folder_tmp = os.path.split(p.A0_out.format(t=t))[0]
                if not os.path.exists(folder_tmp):
                    os.makedirs(folder_tmp)

        max_t = max(p.time_points)
        p.to_register = p.time_points
        if p.bdv_voxel_size is None:
            p.bdv_voxel_size = p.voxel_size

    def compute_trsfs(self, p: trsf_parameters):
        """
        Compute all the transformations from a given set of parameters

        Args:
            p (trsf_parameters): Parameter object
        """
        # Create the output folder for the transfomrations
        if not os.path.exists(p.trsf_folder):
            os.makedirs(p.trsf_folder)

        # trsf_fmt = "t{flo:06d}-{ref:06d}.tif"
        try:
            self.run_produce_trsf(p)
        except Exception as e:
            print(p.trsf_folder)
            print(e)

    def run_trsf(self):
        """
        Start the Spatial registration after having informed the parameter files
        """
        for p in self.params:
            try:
                print("Starting experiment")
                print(p)
                self.prepare_paths(p)
                if p.compute_trsf:
                    self.compute_trsfs(p)
            except Exception as e:
                print("Failure of %s" % p.origin_file_name)
                print(e)

    def __init__(self, params=None):
        if params is None:
            self.params = self.read_param_file()
        elif (
            isinstance(params, str)
            or isinstance(params, Path)
            or isinstance(params, dict)
        ):
            self.params = VectorFlow.read_param_file(params)
        else:
            self.params = params
        if self.params is not None and 0 < len(self.params):
            self.path_to_bin = self.params[0].path_to_bin


def vectorflow():
    reg = VectorFlow()
    reg.run_trsf()
