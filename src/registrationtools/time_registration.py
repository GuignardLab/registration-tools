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

        correct = True
        if not "path_to_data" in self.__dict__:
            print('\n\t"path_to_data" is required')
            correct = False
        if not "file_name" in self.__dict__:
            print('\n\t"file_name" is required')
            correct = False
        if not "trsf_folder" in self.__dict__:
            print('\n\t"trsf_folder" is required')
            correct = False
        if not "voxel_size" in self.__dict__:
            print('\n\t"voxel_size" is required')
            correct = False
        if not "ref_TP" in self.__dict__:
            print('\n\t"ref_TP" is required')
            correct = False
        if not "projection_path" in self.__dict__:
            print('\n\t"projection_path" is required')
            correct = False
        if self.apply_trsf and not "output_format" in self.__dict__:
            print('\n\t"output_format" is required')
            correct = False
        if self.trsf_type != "translation" and (
            self.lowess or self.trsf_interpolation
        ):
            print("\n\tLowess or transformation interplolation")
            print("\tonly work with translation")
            correct = False
        if self.lowess and not "window_size" in self.param_dict:
            print('\n\tLowess smoothing "window_size" is missing')
            print("\tdefault value of 5 will be used\n")
        if self.trsf_interpolation and not "step_size" in self.param_dict:
            print('\n\tTransformation interpolation "step_size" is missing')
            print("\tdefault value of 100 will be used\n")
        if self.trsf_type == "vectorfield" and self.ref_path is None:
            print("\tNon-linear transformation asked with propagation.")
            print("\tWhile working it is highly not recommended")
            print(
                "\tPlease consider not doing a registration from propagation"
            )
            print("\tTHIS WILL LITERALLY TAKE AGES!! IF IT WORKS AT ALL ...")
        if (
            not isinstance(self.spline, int)
            or self.spline < 1
            or 5 < self.spline
        ):
            out = ("{:d}" if isinstance(self.spline, int) else "{:s}").format(
                self.spline
            )
            print("The degree of smoothing for the spline interpolation")
            print("Should be an Integer between 1 and 5, you gave " + out)
        return correct

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
            output += "ref_TP".ljust(max_key, " ") + ": {:d}\n".format(
                self.ref_TP
            )
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
            output += "padding".ljust(max_key, " ") + ": {:d}\n".format(
                self.padding
            )
            output += "lowess".ljust(max_key, " ") + ": {:d}\n".format(
                self.lowess
            )
            if self.lowess:
                output += "window_size".ljust(
                    max_key, " "
                ) + ": {:d}\n".format(self.window_size)
            output += "trsf_interpolation".ljust(
                max_key, " "
            ) + ": {:d}\n".format(self.trsf_interpolation)
            if self.trsf_interpolation:
                output += "step_size".ljust(max_key, " ") + ": {:d}\n".format(
                    self.step_size
                )
            output += "recompute".ljust(max_key, " ") + ": {:d}\n".format(
                self.recompute
            )
        output += "apply_trsf".ljust(max_key, " ") + ": {:d}\n".format(
            self.apply_trsf
        )
        if self.apply_trsf:
            if self.projection_path is not None:
                output += "projection_path".ljust(
                    max_key, " "
                ) + ": {}\n".format(self.projection_path)
            output += "image_interpolation".ljust(
                max_key, " "
            ) + ": {}\n".format(self.image_interpolation)

        return output

    def add_path_prefix(self, prefix: str):
        """
        Add a prefix to all the relevant paths (this is mainly for the unitary tests)

        Args:
            prefix (str): The prefix to add in front of the path
        """
        self.projection_path = os.path.join(prefix, self.projection_path) + os.path.sep
        self.path_to_data = os.path.join(prefix, self.path_to_data) + os.path.sep
        self.trsf_folder = os.path.join(prefix, self.trsf_folder) + os.path.sep
        self.output_format = os.path.join(prefix, self.output_format) + os.path.sep

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
        self.projection_path = None
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

        if 0 < len(self.projection_path) and self.projection_path[-1] != os.path.sep:
            self.projection_path = self.projection_path + os.path.sep
        if 0 < len(self.path_to_data) and self.path_to_data[-1] != os.path.sep:
            self.path_to_data = self.path_to_data + os.path.sep
        if 0 < len(self.trsf_folder) and self.trsf_folder[-1] != os.path.sep:
            self.trsf_folder = self.trsf_folder + os.path.sep

class TimeRegistration:
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
        if not p.sequential:
            p_im_flo = p.A0.format(t=t2)
            p_im_ref = p.ref_path
            t_ref = p.ref_TP
            t_flo = t2
            if t1 == p.to_register[0]:
                self.__produce_trsf((p, t2, t1, make))
        else:
            if t1 < p.ref_TP:
                t_ref = t2
                t_flo = t1
            else:
                t_ref = t1
                t_flo = t2
            p_im_ref = p.A0.format(t=t_ref)
            p_im_flo = p.A0.format(t=t_flo)
        if not make:
            print("trsf tp %d-%d not done" % (t1, t2))
            np.savetxt(
                os.path.join(p.trsf_folder, "t%06d-%06d.txt" % (t_flo, t_ref)),
                np.identity(4),
            )
        elif t_flo != t_ref and (
            p.recompute
            or not os.path.exists(
                os.path.join(p.trsf_folder, "t%06d-%06d.txt" % (t_flo, t_ref))
            )
        ):
            if p.low_th is not None and 0<p.low_th:
                th = " -ref-lt {lt:f} -flo-lt {lt:f} -no-norma ".format(
                    lt=p.low_th
                )
            else:
                th = ""
            if p.trsf_type != "vectorfield":
                if p.pre_2D == 1:
                    print(self.path_to_bin
                        + "blockmatching -ref "
                        + p_im_ref
                        + " -flo "
                        + p_im_flo
                        + " -reference-voxel %f %f %f" % p.voxel_size
                        + " -floating-voxel %f %f %f" % p.voxel_size
                        + " -trsf-type rigid2D -py-hl %d -py-ll %d"
                        % (
                            p.registration_depth_start,
                            p.registration_depth_end,
                        )
                        + " -res-trsf "
                        + os.path.join(p.trsf_folder, "t%06d-%06d-tmp.txt" % (t_flo, t_ref))
                        + th)
                    call(
                        self.path_to_bin
                        + "blockmatching -ref "
                        + p_im_ref
                        + " -flo "
                        + p_im_flo
                        + " -reference-voxel %f %f %f" % p.voxel_size
                        + " -floating-voxel %f %f %f" % p.voxel_size
                        + " -trsf-type rigid2D -py-hl %d -py-ll %d"
                        % (
                            p.registration_depth_start,
                            p.registration_depth_end,
                        )
                        + " -res-trsf "
                        + os.path.join(p.trsf_folder, "t%06d-%06d-tmp.txt" % (t_flo, t_ref))
                        + th,
                        shell=True,
                    )
                    pre_trsf = (
                        " -init-trsf "
                        + os.path.join(p.trsf_folder, "t%06d-%06d-tmp.txt" % (t_flo, t_ref))
                        + " -composition-with-initial "
                    )
                else:
                    pre_trsf = ""
                call(
                    self.path_to_bin
                    + "blockmatching -ref "
                    + p_im_ref
                    + " -flo "
                    + p_im_flo
                    + pre_trsf
                    + " -reference-voxel %f %f %f" % p.voxel_size
                    + " -floating-voxel %f %f %f" % p.voxel_size
                    + " -trsf-type %s -py-hl %d -py-ll %d"
                    % (
                        p.trsf_type,
                        p.registration_depth_start,
                        p.registration_depth_end,
                    )
                    + " -res-trsf "
                    + os.path.join(p.trsf_folder, "t%06d-%06d.txt" % (t_flo, t_ref))
                    + th,
                    shell=True,
                )
            else:
                if p.apply_trsf:
                    res = " -res " + p.A0_out.format(t=t_flo)
                else:
                    res = ""
                if p.keep_vectorfield and pyklb_found:
                    res_trsf = (
                        " -composition-with-initial -res-trsf "
                        + os.path.join(p.trsf_folder, "t%06d-%06d.klb" % (t_flo, t_ref))
                    )
                else:
                    res_trsf = ""
                    if p.keep_vectorfield:
                        print(
                            "The vectorfield cannot be stored without pyklb being installed"
                        )
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
                    + os.path.join(p.trsf_folder, "t%06d-%06d.txt" % (t_flo, t_ref))
                    + th,
                    shell=True,
                )
                call(
                    self.path_to_bin
                    + "blockmatching -ref "
                    + p_im_ref
                    + " -flo "
                    + p_im_flo
                    + " -init-trsf "
                    + os.path.join(p.trsf_folder, "t%06d-%06d.txt" % (t_flo, t_ref))
                    + res
                    + " -reference-voxel %f %f %f" % p.voxel_size
                    + " -floating-voxel %f %f %f" % p.voxel_size
                    + " -trsf-type %s -py-hl %d -py-ll %d"
                    % (
                        p.trsf_type,
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
        tmp = []
        for mi in mapping:
            tmp += [self.__produce_trsf(mi)]
        tac = time()
        whole_time = tac - tic
        secs = whole_time % 60
        whole_time = whole_time // 60
        mins = whole_time % 60
        hours = whole_time // 60
        print("%dh:%dmin:%ds" % (hours, mins, secs))

    def compose_trsf(
        self, flo_t: int, ref_t: int, trsf_p: str, tp_list: List[int]
    ) -> str:
        """
        Recusrively build the transformation that allows
        to register time `flo_t` onto the frame of time `ref_t`
        assuming that it exists the necessary intermediary transformations

        Args:
            flo_t (int): time of the floating image
            ref_t (int): time of the reference image
            trsf_p (str): path to folder containing the transformations
            tp_list (list): list of time points that have been processed

        Returns:
            out_trsf (str): path to the result composed transformation
        """
        out_trsf = trsf_p + "t%06d-%06d.txt" % (flo_t, ref_t)
        if not os.path.exists(out_trsf):
            flo_int = tp_list[tp_list.index(flo_t) + np.sign(ref_t - flo_t)]
            # the call is recursive, to build `T_{flo\leftarrow ref}`
            # we need `T_{flo+1\leftarrow ref}` and `T_{flo\leftarrow ref-1}`
            trsf_1 = self.compose_trsf(flo_int, ref_t, trsf_p, tp_list)
            trsf_2 = self.compose_trsf(flo_t, flo_int, trsf_p, tp_list)
            call(
                self.path_to_bin
                + "composeTrsf "
                + out_trsf
                + " -trsfs "
                + trsf_2
                + " "
                + trsf_1,
                shell=True,
            )
        return out_trsf

    @staticmethod
    def __lowess_smooth(
        X: np.ndarray, T: np.ndarray, frac: float
    ) -> np.ndarray:
        """
        Smooth a curve using the lowess algorithm.
        See:
            Cleveland, W.S. (1979)
            “Robust Locally Weighted Regression and Smoothing Scatterplots”.
            Journal of the American Statistical Association 74 (368): 829-836.

        Args:
            X (np.ndarray): 1D array for the point positions
            T (np.ndarray): 1D array for the times corresponding to the `X` positions
            frac (float): Between 0 and 1. The fraction of the data used when estimating each value within X

        Returns:
            (np.ndarray): the smoothed X values
        """
        return lowess(X, T, frac=frac, is_sorted=True, return_sorted=False)

    @staticmethod
    def interpolation(
        p: trsf_parameters, X: np.ndarray, T: np.ndarray
    ) -> np.ndarray:
        """
        Interpolate positional values between missing timepoints using spline interplolation

        Args:
            p (trsf_parameters): parameters for the function
            X (np.ndarray): 1D array for the point positions
            T (np.ndarray): 1D array for the times corresponding to the `X` position

        Returns:
            (np.ndarray): The interoplated function
        """
        return sp.interpolate.InterpolatedUnivariateSpline(T, X, k=p.spline)

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
                if len(sys.argv) < 2 or sys.argv[1] == '-f':
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
        ### Check the file names and folders:
        p.im_ext = p.file_name.split(".")[-1]  # Image type
        p.A0 = os.path.join(p.path_to_data, p.file_name)  # Image path
        # Output format
        if p.output_format is not None:
            if os.path.split(p.output_format)[0] == "":
                p.A0_out = os.path.join(p.path_to_data, p.output_format)
            elif os.path.split(p.output_format)[1] == "":
                p.A0_out = os.path.join(p.output_format, p.file_name)
            else:
                p.A0_out = p.output_format
        else:
            p.A0_out = os.path.join(p.path_to_data, p.file_name.replace(
                p.im_ext, p.suffix + "." + p.im_ext
            ))

        # Time points to work with
        p.time_points = np.array(
            [i for i in np.arange(p.first, p.last + 1) if not i in p.not_to_do]
        )
        if p.check_TP:
            missing_time_points = []
            for t in p.time_points:
                if not os.path.exists(p.A0.format(t=t)):
                    missing_time_points += [t]
            if len(missing_time_points) != 0:
                print("The following time points are missing:")
                print("\t" + missing_time_points)
                print("Aborting the process")
                exit()
        if not p.sequential:
            p.ref_path = p.A0.format(t=p.ref_TP)
        if p.apply_trsf:
            for t in sorted(p.time_points):
                folder_tmp = os.path.split(p.A0_out.format(t=t))[0]
                if not os.path.exists(folder_tmp):
                    os.makedirs(folder_tmp)

        max_t = max(p.time_points)
        if p.trsf_interpolation:
            p.to_register = sorted(p.time_points)[:: p.step_size]
            if not max_t in p.to_register:
                p.to_register += [max_t]
        else:
            p.to_register = p.time_points
        if not hasattr(p, "bdv_im") or p.bdv_im is None:
            p.bdv_im = p.A0
        if not hasattr(p, "out_bdv") or p.out_bdv is None:
            p.out_bdv = os.path.join(p.trsf_folder, "bdv.xml")
        if p.bdv_voxel_size is None:
            p.bdv_voxel_size = p.voxel_size

    def lowess_filter(self, p: trsf_parameters, trsf_fmt: str) -> str:
        """
        Apply a lowess filter on a set of ordered transformations (only works for translations).
        It does it from already computed transformations and write new transformations on disk.

        Args:
            p (trsf_parameters): parameter object
            trsf_fmt (srt): the string format for the initial transformations

        Returns:
            (str): output transformation string
        """
        X_T = []
        Y_T = []
        Z_T = []
        T = []
        for t in p.to_register:
            trsf_p = p.trsf_folder + trsf_fmt.format(flo=t, ref=p.ref_TP)
            trsf = self.read_trsf(trsf_p)
            T += [t]
            X_T += [trsf[0, -1]]
            Y_T += [trsf[1, -1]]
            Z_T += [trsf[2, -1]]

        frac = float(p.window_size) / len(T)
        X_smoothed = self.__lowess_smooth(p, X_T, T, frac=frac)
        Y_smoothed = self.__lowess_smooth(p, Y_T, T, frac=frac)
        Z_smoothed = self.__lowess_smooth(p, Z_T, T, frac=frac)

        new_trsf_fmt = "t{flo:06d}-{ref:06d}-filtered.txt"
        for i, t in enumerate(T):
            mat = np.identity(4)
            mat[0, -1] = X_smoothed[i]
            mat[1, -1] = Y_smoothed[i]
            mat[2, -1] = Z_smoothed[i]
            np.savetxt(
                p.trsf_folder + new_trsf_fmt.format(flo=t, ref=p.ref_TP), mat
            )
        return new_trsf_fmt

    def interpolate(
        self, p: trsf_parameters, trsf_fmt: str, new_trsf_fmt: str = None
    ) -> str:
        """
        Interpolate a set of ordered transformations (only works for translations).
        It does it from already computed transformations and write new transformations on disk.

        Args:
            p (trsf_parameters): parameter object
            trsf_fmt (srt): the string format for the initial transformations
            new_trsf_fmt (str | None): path to the interpolated transformations. If None,
                the format will be "t{flo:06d}-{ref:06d}-interpolated.txt"

        Returns:
            (str): output transformation format
        """
        if new_trsf_fmt is None:
            new_trsf_fmt = "t{flo:06d}-{ref:06d}-interpolated.txt"
        X_T = []
        Y_T = []
        Z_T = []
        T = []
        for t in p.to_register:
            trsf_p = os.path.join(p.trsf_folder, trsf_fmt.format(flo=t, ref=p.ref_TP))
            trsf = self.read_trsf(trsf_p)
            T += [t]
            X_T += [trsf[0, -1]]
            Y_T += [trsf[1, -1]]
            Z_T += [trsf[2, -1]]

        X_interp = self.interpolation(p, X_T, T)
        Y_interp = self.interpolation(p, Y_T, T)
        Z_interp = self.interpolation(p, Z_T, T)
        for t in p.time_points:
            mat = np.identity(4)
            mat[0, -1] = X_interp(t)
            mat[1, -1] = Y_interp(t)
            mat[2, -1] = Z_interp(t)
            np.savetxt(
                os.path.join(p.trsf_folder, new_trsf_fmt.format(flo=t, ref=p.ref_TP), mat)
            )
        trsf_fmt = new_trsf_fmt
        return trsf_fmt

    @staticmethod
    def pad_trsfs(p: trsf_parameters, trsf_fmt: str):
        """
        Pad transformations

        Args:
            p (trsf_parameters): parameter object
            trsf_fmt (srt): the string format for the initial transformations
        """
        if not p.sequential and p.ref_path.split(".")[-1] == "klb":
            im_shape = readheader(p.ref_path)["imagesize_tczyx"][-1:-4:-1]
        elif not p.sequential:
            im_shape = imread(p.ref_path).shape
        elif p.A0.split(".")[-1] == "klb":
            im_shape = readheader(p.A0.format(t=p.ref_TP))["imagesize_tczyx"][
                -1:-4:-1
            ]
        else:
            im_shape = imread(p.A0.format(t=p.ref_TP)).shape
        im = SpatialImage(np.ones(im_shape), dtype=np.uint8)
        im.voxelsize = p.voxel_size
        if pyklb_found:
            template = os.path.join(p.trsf_folder, "tmp.klb")
            res_t = "template.klb"
        else:
            template = os.path.join(p.trsf_folder, "tmp.tif")
            res_t = "template.tif"

        imsave(template, im)
        identity = np.identity(4)

        trsf_fmt_no_flo = trsf_fmt.replace("{flo:06d}", "%06d")
        new_trsf_fmt = "t{flo:06d}-{ref:06d}-padded.txt"
        new_trsf_fmt_no_flo = new_trsf_fmt.replace("{flo:06d}", "%06d")
        for t in p.not_to_do:
            np.savetxt(
                os.path.join(p.trsf_folder, trsf_fmt.format(flo=t, ref=p.ref_TP), identity)
            )

        call(
            p.path_to_bin
            + "changeMultipleTrsfs -trsf-format "
            + os.path.join(p.trsf_folder, trsf_fmt_no_flo.format(ref=p.ref_TP))
            + " -index-reference %d -first %d -last %d "
            % (p.ref_TP, min(p.time_points), max(p.time_points))
            + " -template "
            + template
            + " -res "
            + os.path.join(p.trsf_folder, new_trsf_fmt_no_flo.format(ref=p.ref_TP))
            + " -res-t "
            + os.path.join(p.trsf_folder, res_t)
            + " -trsf-type %s -vs %f %f %f" % ((p.trsf_type,) + p.voxel_size),
            shell=True,
        )

    def compute_trsfs(self, p: trsf_parameters):
        """
        Compute all the transformations from a given set of parameters

        Args:
            p (trsf_parameters): Parameter object
        """
        # Create the output folder for the transfomrations
        if not os.path.exists(p.trsf_folder):
            os.makedirs(p.trsf_folder)

        trsf_fmt = "t{flo:06d}-{ref:06d}.txt"
        try:
            self.run_produce_trsf(p)
            if p.sequential:
                if min(p.to_register) != p.ref_TP:
                    self.compose_trsf(
                        min(p.to_register),
                        p.ref_TP,
                        p.trsf_folder,
                        list(p.to_register),
                    )
                if max(p.to_register) != p.ref_TP:
                    self.compose_trsf(
                        max(p.to_register),
                        p.ref_TP,
                        p.trsf_folder,
                        list(p.to_register),
                    )
            np.savetxt(
                os.path.join("{:s}", trsf_fmt).format(
                    p.trsf_folder, flo=p.ref_TP, ref=p.ref_TP
                ),
                np.identity(4),
            )
        except Exception as e:
            print(p.trsf_folder)
            print(e)

        if p.lowess:
            trsf_fmt = self.lowess_filter(p, trsf_fmt)
        if p.trsf_interpolation:
            trsf_fmt = interpolate(p, trsf_fmt)
        if p.padding:
            self.pad_trsfs(p, trsf_fmt)

    @staticmethod
    def apply_trsf(p: trsf_parameters):
        """
        Apply transformations to a movie from a set of computed transformations

        Args:
            p (trsf_parameters): Parameter object
        """
        trsf_fmt = "t{flo:06d}-{ref:06d}.txt"
        if p.lowess:
            trsf_fmt = "t{flo:06d}-{ref:06d}-filtered.txt"
        if p.trsf_interpolation:
            trsf_fmt = "t{flo:06d}-{ref:06d}-interpolated.txt"
        if p.padding:
            trsf_fmt = "t{flo:06d}-{ref:06d}-padded.txt"
            if pyklb_found:
                template = os.path.join(p.trsf_folder, "template.klb")
                X, Y, Z = readheader(template)["imagesize_tczyx"][-1:-4:-1]
            else:
                template = os.path.join(p.trsf_folder, "template.tif")
                X, Y, Z = imread(template).shape
        elif p.A0.split(".")[-1] == "klb":
            X, Y, Z = readheader(p.A0.format(t=p.ref_TP))["imagesize_tczyx"][
                -1:-4:-1
            ]
            template = p.A0.format(t=p.ref_TP)
        else:
            X, Y, Z = imread(p.A0.format(t=p.ref_TP)).shape
            template = p.A0.format(t=p.ref_TP)

        xy_proj = np.zeros((X, Y, len(p.time_points)), dtype=np.uint16)
        xz_proj = np.zeros((X, Z, len(p.time_points)), dtype=np.uint16)
        yz_proj = np.zeros((Y, Z, len(p.time_points)), dtype=np.uint16)
        for i, t in enumerate(sorted(p.time_points)):
            folder_tmp = os.path.split(p.A0_out.format(t=t))[0]
            if not os.path.exists(folder_tmp):
                os.makedirs(folder_tmp)
            call(
                p.path_to_bin
                + "applyTrsf '%s' '%s' -trsf "
                % (p.A0.format(t=t), p.A0_out.format(t=t))
                + os.path.join(p.trsf_folder, trsf_fmt.format(flo=t, ref=p.ref_TP))
                + " -template "
                + template
                + " -floating-voxel %f %f %f " % p.voxel_size
                + " -reference-voxel %f %f %f " % p.voxel_size_out
                + " -interpolation %s" % p.image_interpolation,
                shell=True,
            )
            im = imread(p.A0_out.format(t=t))
            if p.projection_path is not None:
                xy_proj[..., i] = SpatialImage(np.max(im, axis=2))
                xz_proj[..., i] = SpatialImage(np.max(im, axis=1))
                yz_proj[..., i] = SpatialImage(np.max(im, axis=0))
        if p.projection_path is not None:
            if not os.path.exists(p.projection_path):
                os.makedirs(p.projection_path)
            p_to_data = p.projection_path
            num_s = p.file_name.find("{")
            num_e = p.file_name.find("}") + 1
            f_name = p.file_name.replace(p.file_name[num_s:num_e], "")
            if not os.path.exists(p_to_data.format(t=-1)):
                os.makedirs(p_to_data.format(t=-1))
            imsave(
                os.path.join(p_to_data, f_name.replace(p.im_ext, "xyProjection.tif")),
                SpatialImage(xy_proj),
            )
            imsave(
                os.path.join(p_to_data, f_name.replace(p.im_ext, "xzProjection.tif")),
                SpatialImage(xz_proj),
            )
            imsave(
                os.path.join(p_to_data, f_name.replace(p.im_ext, "yzProjection.tif")),
                SpatialImage(yz_proj),
            )

    @staticmethod
    def inv_trsf(trsf: np.ndarray) -> np.ndarray:
        """
        Inverse a transformation

        Args:
            trsf (np.ndarray): 4x4 transformation matrix

        Returns
            (np.ndarray): the inverse of the input transformation
        """
        return np.linalg.lstsq(trsf, np.identity(4))[0]

    @staticmethod
    def prettify(elem: ET.Element) -> str:
        """
        Return a pretty-printed XML string for the Element.

        Args:
            elem (xml.etree.ElementTree.Element): xml element

        Returns:
            (str): a nice version of our xml file
        """
        rough_string = ET.tostring(elem, "utf-8")
        reparsed = minidom.parseString(rough_string)
        return reparsed.toprettyxml(indent="  ")

    @staticmethod
    def do_viewSetup(
        ViewSetup: ET.SubElement,
        p: trsf_parameters,
        im_size: Tuple[int, int, int],
        i: int,
    ):
        """
        Setup xml elements for BigDataViewer

        Args:
            ViewSetup (ET.SubElement): ...
            p (trsf_parameters): Parameter object
            im_size (tuple): tuple of 3 integers with the dimension of the image
            i (int): angle id
        """
        id_ = ET.SubElement(ViewSetup, "id")
        id_.text = "%d" % i
        name = ET.SubElement(ViewSetup, "name")
        name.text = "%d" % i
        size = ET.SubElement(ViewSetup, "size")
        size.text = "%d %d %d" % tuple(im_size)
        voxelSize = ET.SubElement(ViewSetup, "voxelSize")
        unit = ET.SubElement(voxelSize, "unit")
        unit.text = p.bdv_unit
        size = ET.SubElement(voxelSize, "size")
        size.text = "%f %f %f" % tuple(p.bdv_voxel_size)
        attributes = ET.SubElement(ViewSetup, "attributes")
        illumination = ET.SubElement(attributes, "illumination")
        illumination.text = "0"
        channel = ET.SubElement(attributes, "channel")
        channel.text = "0"
        tile = ET.SubElement(attributes, "tile")
        tile.text = "0"
        angle = ET.SubElement(attributes, "angle")
        angle.text = "%d" % i

    def do_ViewRegistration(
        self,
        ViewRegistrations: ET.SubElement,
        p: trsf_parameters,
        t: int,
        a: int,
    ):
        """
        Write the view registration for BigDataViewer

        Args:
            ViewSetup (ET.SubElement): ...
            p (trsf_parameters): Parameter object
            t (int): time point to treat
            a (int): angle to treat (useless for time regisrtation)
        """
        ViewRegistration = ET.SubElement(ViewRegistrations, "ViewRegistration")
        ViewRegistration.set("timepoint", "%d" % t)
        ViewRegistration.set("setup", "0")
        ViewTransform = ET.SubElement(ViewRegistration, "ViewTransform")
        ViewTransform.set("type", "affine")
        affine = ET.SubElement(ViewTransform, "affine")
        f = os.path.join(p.trsf_folder, "t%06d-%06d.txt" % (t, p.ref_TP))
        trsf = self.read_trsf(f)
        trsf = self.inv_trsf(trsf)
        formated_trsf = tuple(trsf[:-1, :].flatten())
        affine.text = ("%f " * 12) % formated_trsf
        ViewTransform = ET.SubElement(ViewRegistration, "ViewTransform")
        ViewTransform.set("type", "affine")
        affine = ET.SubElement(ViewTransform, "affine")
        affine.text = (
            "%f 0.0 0.0 0.0 0.0 %f 0.0 0.0 0.0 0.0 %f 0.0" % p.voxel_size
        )

    def build_bdv(self, p: trsf_parameters):
        """
        Build the BigDataViewer xml

        Args:
            p (trsf_parameters): Parameter object
        """
        if not p.im_ext in ["klb"]:  # ['tif', 'klb', 'tiff']:
            print("Image format not adapted for BigDataViewer")
            return
        SpimData = ET.Element("SpimData")
        SpimData.set("version", "0.2")
        SpimData.set("encoding", "UTF-8")

        base_path = ET.SubElement(SpimData, "BasePath")
        base_path.set("type", "relative")
        base_path.text = "."

        SequenceDescription = ET.SubElement(SpimData, "SequenceDescription")

        ImageLoader = ET.SubElement(SequenceDescription, "ImageLoader")
        ImageLoader.set("format", p.im_ext)
        Resolver = ET.SubElement(ImageLoader, "Resolver")
        Resolver.set(
            "type", "org.janelia.simview.klb.bdv.KlbPartitionResolver"
        )
        ViewSetupTemplate = ET.SubElement(Resolver, "ViewSetupTemplate")
        template = ET.SubElement(ViewSetupTemplate, "template")
        template.text = p.bdv_im.format(t=p.to_register[0])
        timeTag = ET.SubElement(ViewSetupTemplate, "timeTag")
        timeTag.text = p.time_tag

        ViewSetups = ET.SubElement(SequenceDescription, "ViewSetups")
        ViewSetup = ET.SubElement(ViewSetups, "ViewSetup")
        if p.im_ext == "klb":
            im_size = tuple(
                readheader(p.A0.format(t=p.ref_TP))["imagesize_tczyx"][
                    -1:-4:-1
                ]
            )
        else:
            im_size = tuple(imread(p.A0.format(t=p.ref_TP)).shape)
        self.do_viewSetup(ViewSetup, p, im_size, 0)

        Attributes = ET.SubElement(ViewSetups, "Attributes")
        Attributes.set("name", "illumination")
        Illumination = ET.SubElement(Attributes, "Illumination")
        id_ = ET.SubElement(Illumination, "id")
        id_.text = "0"
        name = ET.SubElement(Illumination, "name")
        name.text = "0"

        Attributes = ET.SubElement(ViewSetups, "Attributes")
        Attributes.set("name", "channel")
        Channel = ET.SubElement(Attributes, "Channel")
        id_ = ET.SubElement(Channel, "id")
        id_.text = "0"
        name = ET.SubElement(Channel, "name")
        name.text = "0"

        Attributes = ET.SubElement(ViewSetups, "Attributes")
        Attributes.set("name", "tile")
        Tile = ET.SubElement(Attributes, "Tile")
        id_ = ET.SubElement(Tile, "id")
        id_.text = "0"
        name = ET.SubElement(Tile, "name")
        name.text = "0"

        Attributes = ET.SubElement(ViewSetups, "Attributes")
        Attributes.set("name", "angle")
        Angle = ET.SubElement(Attributes, "Angle")
        id_ = ET.SubElement(Angle, "id")
        id_.text = "0"
        name = ET.SubElement(Angle, "name")
        name.text = "0"

        TimePoints = ET.SubElement(SequenceDescription, "Timepoints")
        TimePoints.set("type", "range")
        first = ET.SubElement(TimePoints, "first")
        first.text = "%d" % min(p.to_register)
        last = ET.SubElement(TimePoints, "last")
        last.text = "%d" % max(p.to_register)
        ViewRegistrations = ET.SubElement(SpimData, "ViewRegistrations")
        b = min(p.to_register)
        e = max(p.to_register)
        for t in range(b, e + 1):
            self.do_ViewRegistration(ViewRegistrations, p, t)

        with open(p.out_bdv, "w") as f:
            f.write(self.prettify(SpimData))
            f.close()

    def plot_transformations(self, p: trsf_parameters):

        trsf_fmt = "t{flo:06d}-{ref:06d}.txt"
        if p.lowess:
            trsf_fmt = "t{flo:06d}-{ref:06d}-filtered.txt"
        if p.trsf_interpolation:
            trsf_fmt = "t{flo:06d}-{ref:06d}-interpolated.txt"
        if p.padding:
            trsf_fmt = "t{flo:06d}-{ref:06d}-padded.txt"

        import matplotlib.pyplot as plt
        tX, tY, tZ = [], [], []
        rX, rY, rZ = [], [], []
        for t in sorted(p.time_points):
            trsf = self.read_trsf(os.path.join(p.trsf_folder, trsf_fmt.format(flo=t, ref=p.ref_TP)))
            (tx, ty, tz), M, *_ = decompose(trsf)
            rx, ry, rz = mat2euler(M)
            tX.append(tx)
            tY.append(ty)
            tZ.append(tz)
            rX.append(np.rad2deg(rx))
            rY.append(np.rad2deg(ry))
            rZ.append(np.rad2deg(rz))
        fig, ax = plt.subplots(3, 1, figsize=(8, 5), sharex=True, sharey=True)
        ax[0].plot(p.time_points, tX, 'o-')
        ax[1].plot(p.time_points, tY, 'o-')
        ax[2].plot(p.time_points, tZ, 'o-')
        for axis, axi in zip(['X', 'Y', 'Z'], ax):
            axi.set_ylabel(f'{axis} Translation [µm]')
        ax[2].set_xlabel('Time')
        fig.suptitle('Translations')
        fig.tight_layout()

        fig, ax = plt.subplots(3, 1, figsize=(8, 5), sharex=True, sharey=True)
        ax[0].plot(p.time_points, rX, 'o-')
        ax[1].plot(p.time_points, rY, 'o-')
        ax[2].plot(p.time_points, rZ, 'o-')
        for axis, axi in zip(['X', 'Y', 'Z'], ax):
            axi.set_ylabel(f'{axis} Rotation\nin degree')
        ax[2].set_xlabel('Time')
        fig.suptitle('Rotations')
        fig.tight_layout()

        plt.show()

        
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
                if p.plot_trsf:
                    self.plot_transformations(p)
                if p.apply_trsf and p.trsf_type != "vectorfield":
                    self.apply_trsf(p)
                if p.do_bdv:
                    self.build_bdv(p)
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
            self.params = TimeRegistration.read_param_file(params)
        else:
            self.params = params
        if self.params is not None and 0 < len(self.params):
            self.path_to_bin = self.params[0].path_to_bin


def time_registration():
    reg = TimeRegistration()
    reg.run_trsf()
