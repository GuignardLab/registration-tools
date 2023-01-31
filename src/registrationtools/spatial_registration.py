#!/usr/bin/python
# This file is subject to the terms and conditions defined in
# file 'LICENCE', which is part of this source code package.
# Author: Leo Guignard (leo.guignard...@AT@...univ-amu.fr)

import numpy as np
import os
import sys
import json
from subprocess import call
import xml.etree.ElementTree as ET
from xml.dom import minidom
from shutil import copyfile
from pathlib import Path
from typing import Union, List, Tuple
from IO import imread, imsave, SpatialImage


class trsf_parameters(object):
    """
    Read parameters for the registration function from a preformated json file
    """

    def check_parameters_consistancy(self) -> bool:
        """
        Function that should check parameter consistancy

        @TODO:
        write something that actually do check
        """
        correct = True
        if not (
            hasattr(self, "out_pattern") or hasattr(self, "output_format")
        ):
            print("The output pattern cannot be an empty string")
            correct = False
        return correct

    def __str__(self) -> str:
        max_key = (
            max([len(k) for k in self.__dict__.keys() if k != "param_dict"])
            + 1
        )
        output = "The registration will run with the following arguments:\n"
        output += "\n" + " File format \n"
        output += "path_to_data".ljust(max_key, " ") + ": {:s}\n".format(
            self.path_to_data
        )
        output += "ref_im".ljust(max_key, " ") + ": {:s}\n".format(self.ref_im)

        output += "flo_ims".ljust(max_key, " ") + ": "
        tmp_just_len = len("flo_ims".ljust(max_key, " ") + ": ")
        already = tmp_just_len + 1
        for flo in self.flo_ims:
            output += (" " * (tmp_just_len - already)) + "{:s}\n".format(flo)
            already = 0

        output += "init_trsfs".ljust(max_key, " ") + ": "
        tmp_just_len = len("init_trsfs".ljust(max_key, " ") + ": ")
        already = tmp_just_len + 1
        for init_trsfs in self.init_trsfs:
            if init_trsfs is not None and 0 < len(init_trsfs):
                output += (" " * (tmp_just_len - already)) + "{}\n".format(
                    init_trsfs
                )
            already = 0

        output += "trsf_types".ljust(max_key, " ") + ": "
        tmp_just_len = len("trsf_types".ljust(max_key, " ") + ": ")
        already = tmp_just_len + 1
        for trsf_type in self.trsf_types:
            output += (" " * (tmp_just_len - already)) + "{:s}\n".format(
                trsf_type
            )
            already = 0

        output += "ref_voxel".ljust(
            max_key, " "
        ) + ": {:f} x {:f} x {:f}\n".format(*self.ref_voxel)

        output += "flo_voxels".ljust(max_key, " ") + ": "
        tmp_just_len = len("flo_voxels".ljust(max_key, " ") + ": ")
        already = tmp_just_len + 1
        for flo_voxel in self.flo_voxels:
            output += (
                " " * (tmp_just_len - already)
            ) + "{:f} x {:f} x {:f}\n".format(*flo_voxel)
            already = 0

        output += "out_voxel".ljust(
            max_key, " "
        ) + ": {:f} x {:f} x {:f}\n".format(*self.out_voxel)

        output += "out_pattern".ljust(max_key, " ") + ": {:s}\n".format(
            self.out_pattern
        )

        return output

    def add_path_prefix(self, prefix: str):
        """
        Add a prefix to all the relevant paths (this is mainly for the unitary tests)

        Args:
            prefix (str): The prefix to add in front of the path
        """
        self.path_to_data = str(Path(prefix) / self.path_to_data)
        self.trsf_paths = [
            str(Path(prefix) / trsf) for trsf in self.trsf_paths
        ]

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
        self.param_dict = param_dict
        self.init_trsfs = [[], [], []]
        self.path_to_bin = ""
        self.registration_depth = 3
        self.init_trsf_real_unit = True
        self.image_interpolation = "linear"
        self.apply_trsf = True
        self.compute_trsf = True
        self.test_init = False
        self.begin = None
        self.end = None
        self.trsf_types = []
        self.time_tag = "TM"
        self.bdv_unit = "microns"
        self.bdv_voxel_size = None
        self.do_bdv = 0
        self.flo_im_sizes = None
        self.copy_ref = False
        self.bbox_out = False

        self.__dict__.update(param_dict)
        self.ref_voxel = tuple(self.ref_voxel)
        self.flo_voxels = [tuple(vox) for vox in self.flo_voxels]
        self.out_voxel = tuple(self.out_voxel)
        self.origin_file_name = file_name
        self.path_to_bin = os.path.join(self.path_to_bin, "")
        if 0 < len(self.path_to_bin) and not os.path.exists(self.path_to_bin):
            print("Binary path could not be found, will try with global call")
            self.path_to_bin = ""


class SpatialRegistration:
    @staticmethod
    def axis_rotation_matrix(
        axis: str,
        angle: float,
        min_space: Tuple[int, int, int] = None,
        max_space: Tuple[int, int, int] = None,
    ) -> np.ndarray:
        """Return the transformation matrix from the axis and angle necessary

        Args:
            axis (str): axis of rotation ("X", "Y" or "Z")
            angle (float) : angle of rotation (in degree)
            min_space (tuple(int, int, int)): coordinates of the bottom point (usually (0, 0, 0))
            max_space (tuple(int, int, int)): coordinates of the top point (usually im shape)

        Returns:
            (ndarray): 4x4 rotation matrix
        """
        import math

        I = np.linalg.inv
        D = np.dot
        if axis not in ["X", "Y", "Z"]:
            raise Exception(f"Unknown axis: {axis}")
        rads = math.radians(angle)
        s = math.sin(rads)
        c = math.cos(rads)

        centering = np.identity(4)
        if min_space is None and max_space is not None:
            min_space = np.array([0.0, 0.0, 0.0])

        if max_space is not None:
            space_center = (max_space - min_space) / 2.0
            offset = -1.0 * space_center
            centering[:3, 3] = offset

        rot = np.identity(4)
        if axis == "X":
            rot = np.array(
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, c, -s, 0.0],
                    [0.0, s, c, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )
        elif axis == "Y":
            rot = np.array(
                [
                    [c, 0.0, s, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [-s, 0.0, c, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )

        elif axis == "Z":
            rot = np.array(
                [
                    [c, -s, 0.0, 0.0],
                    [s, c, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ]
            )

        return D(I(centering), D(rot, centering))

    @staticmethod
    def flip_matrix(axis: str, im_size: tuple) -> np.ndarray:
        """
        Build a matrix to flip an image according to a given axis

        Args:
            axis (str): axis along the flip is done ("X", "Y" or "Z")
            im_size (tuple(int, int, int)): coordinates of the top point (usually im shape)

        Returns:
            (ndarray): 4x4 flipping matrix
        """
        out = np.identity(4)
        if axis == "X":
            out[0, 0] = -1
            out[0, -1] = im_size[0]
        if axis == "Y":
            out[1, 1] = -1
            out[1, -1] = im_size[1]
        if axis == "Z":
            out[2, 2] = -1
            out[2, -1] = im_size[2]
        return out

    @staticmethod
    def translation_matrix(axis: str, tr: float) -> np.ndarray:
        """
        Build a matrix to flip an image according to a given axis

        Args:
            axis (str): axis along the flip is done ("X", "Y" or "Z")
            tr (float): translation value

        Returns:
            (ndarray): 4x4 flipping matrix
        """
        out = np.identity(4)
        if axis == "X":
            out[0, -1] = tr
        if axis == "Y":
            out[1, -1] = tr
        if axis == "Z":
            out[2, -1] = tr
        return out

    @classmethod
    def read_param_file(
        clf, p_param: trsf_parameters = None
    ) -> trsf_parameters:
        """
        Asks for, reads and formats the parameter file
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
    def read_trsf(path: Union[str, Path]) -> np.ndarray:
        """
        Read a transformation from a text file

        Args:
            path (str | Path): path to a transformation

        Returns:
            (ndarray): 4x4 transformation matrix
        """
        f = open(path)
        if f.read()[0] == "(":
            f.close()
            f = open(path)
            lines = f.readlines()[2:-1]
            f.close()
            return np.array([[float(v) for v in l.split()] for l in lines])
        else:
            f.close()
            return np.loadtxt(path)

    @staticmethod
    def prepare_paths(p: trsf_parameters):
        """
        Prepare the paths in a format that is usable by the algorithm

        Args:
            p (trsf_parameters): path to a transformation
        """
        p.ref_A = os.path.join(p.path_to_data, p.ref_im)
        p.flo_As = []
        for flo_im in p.flo_ims:
            p.flo_As += [os.path.join(p.path_to_data, flo_im)]
        if os.path.split(p.out_pattern)[0] == "":
            ext = p.ref_im.split(".")[-1]
            p.ref_out = p.ref_A.replace(ext, p.out_pattern + "." + ext)
            p.flo_outs = []
            for flo in p.flo_As:
                p.flo_outs += [flo.replace(ext, p.out_pattern + "." + ext)]
        else:
            if not os.path.exists(p.out_pattern):
                os.makedirs(p.out_pattern)
            p.ref_out = os.path.join(p.out_pattern, p.ref_im)
            p.flo_outs = []
            for flo in p.flo_ims:
                p.flo_outs += [os.path.join(p.out_pattern, flo)]
        if not hasattr(p, "trsf_paths"):
            p.trsf_paths = [os.path.split(pi)[0] for pi in p.flo_outs]
            p.trsf_names = ["A{a:d}-{trsf:s}.trsf" for _ in p.flo_outs]
        else:
            formated_paths = []
            p.trsf_names = []
            for pi in p.trsf_paths:
                path, n = os.path.split(pi)
                if n == "":
                    n = "A{a:d}-{trsf:s}.trsf"
                elif not "{a:" in n:
                    if not "{trsf:" in n:
                        n += "{a:d}-{trsf:s}.trsf"
                    else:
                        n += "{a:d}.trsf"
                elif not "{trsf:" in n:
                    n += "{trsf:s}.trsf"
                formated_paths += [path]
                p.trsf_names += [n]
            p.trsf_paths = formated_paths
        if p.begin is None:
            s = p.ref_A.find(p.time_tag) + len(p.time_tag)
            e = s
            while p.ref_A[e].isdigit() and e < len(p.ref_A):
                e += 1
            p.begin = p.end = int(p.ref_A[s:e])
        if p.flo_im_sizes is None:
            p.flo_im_sizes = []
            for im_p in p.flo_As:
                p.flo_im_sizes.append(imread(im_p).shape)
        if (
            not hasattr(p, "ref_im_size") or p.ref_im_size is None
        ) and p.flo_im_sizes is not None:
            p.ref_im_size = p.flo_im_sizes[0]
        else:
            p.ref_im_size = imread(p.ref_A).shape
        if not hasattr(p, "bdv_im") or p.bdv_im is None:
            p.bdv_im = [p.ref_A] + p.flo_As
        if not hasattr(p, "out_bdv") or p.out_bdv is None:
            p.out_bdv = os.path.join(p.trsf_paths[0], "bdv.xml")
        if p.bdv_voxel_size is None:
            p.bdv_voxel_size = p.ref_voxel

    @staticmethod
    def inv_trsf(trsf: Union[np.ndarray, List[List]]) -> np.ndarray:
        """
        Inverse a given transformation

        Args:
            trsf (np.ndarray): 4x4 ndarray

        Returns:
            (np.ndarray): the 4x4 inverted matrix
        """
        return np.linalg.lstsq(trsf, np.identity(4))[0]

    def vox_to_real(
        self,
        trsf: Union[np.ndarray, List[List]],
        ref_vs: List[float],
    ) -> np.ndarray:
        """
        Transform a transformation for voxel units to physical units

        Args:
            trsf (ndarray): 4x4 matrix
            ref_vs (list(float, float, float)): initial voxel size in each dimension

        Returns:
            (np.ndarray): new matrix in metric size
        """
        H_ref = [
            [ref_vs[0], 0, 0, 0],
            [0, ref_vs[1], 0, 0],
            [0, 0, ref_vs[2], 0],
            [0, 0, 0, 1],
        ]
        H_ref_inv = self.inv_trsf(H_ref)
        return np.dot(trsf, H_ref_inv)

    def compute_trsfs(self, p: trsf_parameters):
        """
        Here is where the magic happens, give as an input the trsf_parameters object, get your transformations computed

        Args:
            p (trsf_parameters): parameters to compute the transformation
        """
        for A_num, flo_A in enumerate(p.flo_As):
            flo_voxel = p.flo_voxels[A_num]
            init_trsf = p.init_trsfs[A_num]
            trsf_path = p.trsf_paths[A_num]
            trsf_name = p.trsf_names[A_num]
            if isinstance(init_trsf, list):
                i = 0
                trsfs = []
                im_size = (
                    np.array(p.flo_im_sizes[A_num], dtype=float) * flo_voxel
                )
                while i < len(init_trsf):
                    t_type = init_trsf[i]
                    i += 1
                    axis = init_trsf[i]
                    i += 1
                    if "rot" in t_type:
                        angle = init_trsf[i]
                        i += 1
                        trsfs += [
                            self.axis_rotation_matrix(
                                axis.upper(), angle, np.zeros(3), im_size
                            )
                        ]
                    elif "flip" in t_type:
                        trsfs += [self.flip_matrix(axis.upper(), im_size)]
                    elif "trans" in t_type:
                        tr = init_trsf[i]
                        i += 1
                        trsfs += [self.translation_matrix(axis, tr)]
                res = np.identity(4)
                for trsf in trsfs:
                    res = np.dot(res, trsf)
                if not os.path.exists(trsf_path):
                    os.makedirs(trsf_path)
                init_trsf = os.path.join(
                    trsf_path, "A{:d}-init.trsf".format(A_num + 1)
                )
                np.savetxt(init_trsf, res)
            elif not p.init_trsf_real_unit and init_trsf is not None:
                tmp = self.vox_to_real(
                    self.inv_trsf(self.read_trsf(init_trsf)),
                    flo_voxel,
                    p.ref_voxel,
                )
                init_ext = init_trsf.split(".")[-1]
                init_trsf = init_trsf.replace(init_ext, "real.txt")
                np.savetxt(init_trsf, self.inv_trsf(tmp))
            if init_trsf is not None:
                init_trsf_command = " -init-trsf {:s}".format(init_trsf)
            else:
                init_trsf_command = ""
            i = 0
            if not p.test_init:
                for i, trsf_type in enumerate(p.trsf_types[:-1]):
                    if i != 0:
                        init_trsf_command = " -init-trsf {:s}".format(
                            os.path.join(trsf_path, res_trsf)
                        )
                    res_trsf = os.path.join(
                        trsf_path,
                        trsf_name.format(a=A_num + 1, trsf=trsf_type),
                    )
                    call(
                        p.path_to_bin
                        + "blockmatching -ref "
                        + p.ref_A.format(t=p.begin)
                        + " -flo "
                        + flo_A.format(t=p.begin)
                        + " -reference-voxel %f %f %f" % p.ref_voxel
                        + " -floating-voxel %f %f %f" % flo_voxel
                        + " -trsf-type %s -py-hl 6 -py-ll %d"
                        % (trsf_type, p.registration_depth)
                        + init_trsf_command
                        + " -res-trsf "
                        + res_trsf
                        + " -composition-with-initial",
                        shell=True,
                    )
                trsf_type = p.trsf_types[-1]
                i = len(p.trsf_types) - 1
                if i != 0:
                    init_trsf_command = " -init-trsf {:s}".format(
                        os.path.join(trsf_path, res_trsf)
                    )
                res_trsf = os.path.join(
                    trsf_path, trsf_name.format(a=A_num + 1, trsf=trsf_type)
                )
                res_inv_trsf = os.path.join(
                    trsf_path,
                    ("inv-" + trsf_name).format(a=A_num + 1, trsf=trsf_type),
                )
                call(
                    p.path_to_bin
                    + "blockmatching -ref "
                    + p.ref_A.format(t=p.begin)
                    + " -flo "
                    + flo_A.format(t=p.begin)
                    + " -reference-voxel %f %f %f" % p.ref_voxel
                    + " -floating-voxel %f %f %f" % flo_voxel
                    + " -trsf-type %s -py-hl 6 -py-ll %d"
                    % (trsf_type, p.registration_depth)
                    + init_trsf_command
                    + " -res-trsf "
                    + res_trsf
                    +  # ' -res-voxel-trsf ' + res_voxel_trsf + \
                    # ' -res ' + flo_out +\
                    " -composition-with-initial",
                    shell=True,
                )
                call(
                    p.path_to_bin + "invTrsf %s %s" % (res_trsf, res_inv_trsf),
                    shell=True,
                )

    @staticmethod
    def pad_trsfs(p: trsf_parameters, t: int = None):
        """
        Pad transformations

        Args:
            p (trsf_parameters): parameter object
            trsf_fmt (srt): the string format for the initial transformations
        """

        out_voxel = p.out_voxel
        trsf_path = p.trsf_paths[0]
        trsf_name = p.trsf_names[0]
        trsf_type = p.trsf_types[-1]

        im_shape = imread(p.ref_A.format(t=t)).shape
        im = SpatialImage(np.ones(im_shape, dtype=np.uint8))
        im.voxelsize = p.ref_voxel
        template = os.path.join(trsf_path, "tmp.tif")
        res_t = "template.tif"
        imsave(template, im)
        identity = np.identity(4)

        where_a = trsf_name.find("{a:d}")
        no_a = (trsf_name[:where_a] + trsf_name[where_a + 5 :]).format(
            trsf=trsf_type
        )
        trsf_name_only_a = no_a[:where_a] + "{a:d}" + no_a[where_a:]
        trsf_fmt = os.path.join(trsf_path, trsf_name_only_a)

        trsf_fmt_no_flo = trsf_fmt.replace("{a:d}", "%d")
        new_trsf_fmt = ".".join(trsf_fmt.split(".")[:-1]) + "-padded.txt"
        new_trsf_fmt_no_flo = new_trsf_fmt.replace("{a:d}", "%d")
        np.savetxt(trsf_fmt.format(a=0, trsf=trsf_type), identity)

        call(
            p.path_to_bin
            + "changeMultipleTrsfs -trsf-format "
            + trsf_fmt_no_flo
            + " -index-reference %d -first %d -last %d "
            % (0, 0, len(p.trsf_paths))
            + " -template "
            + template
            + " -res "
            + new_trsf_fmt_no_flo
            + " -res-t "
            + os.path.join(trsf_path, res_t)
            + " "
            + " -trsf-type %s -vs %f %f %f" % ((trsf_type,) + out_voxel),
            shell=True,
        )
        template = os.path.join(trsf_path, res_t)
        return new_trsf_fmt, template

    def apply_trsf(self, p, t=None):
        """
        Apply the transformation according to `trsf_parameters`

        Args:
            p (trsf_parameters): parameters for the transformation
        """
        if p.bbox_out:
            trsf_fmt, template = self.pad_trsfs(p, t)
            A0_trsf = (
                " -trsf " + trsf_fmt.format(a=0) + " -template " + template
            )
        else:
            A0_trsf = ""
        if p.out_voxel != p.ref_voxel or p.bbox_out:
            call(
                p.path_to_bin
                + "applyTrsf"
                + " -flo "
                + p.ref_A.format(t=t)
                + " -res "
                + p.ref_out.format(t=t)
                + " -floating-voxel %f %f %f" % p.ref_voxel
                + " -vs %f %f %f" % p.out_voxel
                + A0_trsf,
                shell=True,
            )
        elif p.copy_ref:
            copyfile(p.ref_A.format(t=t), p.ref_out.format(t=t))
        else:
            p.ref_out = p.ref_A
        for A_num, flo_A in enumerate(p.flo_As):
            flo_voxel = p.flo_voxels[A_num]
            trsf_path = p.trsf_paths[A_num]
            trsf_name = p.trsf_names[A_num]
            init_trsf = p.init_trsfs[A_num]
            if p.test_init:
                if isinstance(init_trsf, list):
                    trsf = " -trsf " + os.path.join(
                        trsf_path, "A{:d}-init.trsf".format(A_num + 1)
                    )
                else:
                    trsf = " -trsf " + init_trsf
            elif not p.bbox_out:
                t_type = "" if len(p.trsf_types) < 1 else p.trsf_types[-1]
                trsf = " -trsf " + os.path.join(
                    trsf_path, trsf_name.format(a=A_num + 1, trsf=t_type)
                )
            else:
                trsf = (
                    " -trsf "
                    + trsf_fmt.format(a=A_num + 1)
                    + " -template "
                    + template
                )
            flo_out = p.flo_outs[A_num]
            call(
                p.path_to_bin
                + "applyTrsf"
                + " -flo "
                + flo_A.format(t=t)
                + " -floating-voxel %f %f %f" % flo_voxel
                + " -res "
                + flo_out.format(t=t)
                + " -ref "
                + p.ref_out.format(t=t)
                + " -reference-voxel %f %f %f" % p.out_voxel
                + trsf
                + " -interpolation %s" % p.image_interpolation,
                shell=True,
            )

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
            a (int): angle to treat
        """

        ViewRegistration = ET.SubElement(ViewRegistrations, "ViewRegistration")
        ViewRegistration.set("timepoint", "%d" % t)
        ViewRegistration.set("setup", "%d" % a)
        ViewTransform = ET.SubElement(ViewRegistration, "ViewTransform")
        ViewTransform.set("type", "affine")
        if a != 0:
            affine = ET.SubElement(ViewTransform, "affine")
            trsf_path = p.trsf_paths[a - 1]
            trsf_name = p.trsf_names[a - 1]
            trsf_type = p.trsf_types[-1]
            f = os.path.join(
                trsf_path, ("inv-" + trsf_name).format(a=a, trsf=trsf_type)
            )
            trsf = self.read_trsf(f)
            formated_trsf = tuple(trsf[:-1, :].flatten())
            affine.text = ("%f " * 12) % formated_trsf
            ViewTransform = ET.SubElement(ViewRegistration, "ViewTransform")
            ViewTransform.set("type", "affine")
            affine = ET.SubElement(ViewTransform, "affine")
            affine.text = (
                "%f 0.0 0.0 0.0 0.0 %f 0.0 0.0 0.0 0.0 %f 0.0"
                % p.flo_voxels[a - 1]
            )
        else:
            affine = ET.SubElement(ViewTransform, "affine")
            affine.text = (
                "%f 0.0 0.0 0.0 0.0 %f 0.0 0.0 0.0 0.0 %f 0.0" % p.ref_voxel
            )

    def build_bdv(self, p: trsf_parameters):
        """
        Build the BigDataViewer xml

        Args:
            p (trsf_parameters): Parameter object
        """
        SpimData = ET.Element("SpimData")
        SpimData.set("version", "0.2")

        base_path = ET.SubElement(SpimData, "BasePath")
        base_path.set("type", "relative")
        base_path.text = "."

        SequenceDescription = ET.SubElement(SpimData, "SequenceDescription")

        ImageLoader = ET.SubElement(SequenceDescription, "ImageLoader")
        ImageLoader.set("format", "klb")
        Resolver = ET.SubElement(ImageLoader, "Resolver")
        Resolver.set(
            "type", "org.janelia.simview.klb.bdv.KlbPartitionResolver"
        )
        for im_p in p.bdv_im:
            ViewSetupTemplate = ET.SubElement(Resolver, "ViewSetupTemplate")
            template = ET.SubElement(ViewSetupTemplate, "template")
            template.text = im_p
            timeTag = ET.SubElement(ViewSetupTemplate, "timeTag")
            timeTag.text = p.time_tag

        ViewSetups = ET.SubElement(SequenceDescription, "ViewSetups")
        ViewSetup = ET.SubElement(ViewSetups, "ViewSetup")
        i = 0
        self.do_viewSetup(ViewSetup, p, p.ref_im_size, i)
        for i, pi in enumerate(p.flo_voxels):
            ViewSetup = ET.SubElement(ViewSetups, "ViewSetup")
            self.do_viewSetup(ViewSetup, p, p.flo_im_sizes[i], i + 1)

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
        for i in range(len(p.flo_voxels) + 1):
            Angle = ET.SubElement(Attributes, "Angle")
            id_ = ET.SubElement(Angle, "id")
            id_.text = "%d" % i
            name = ET.SubElement(Angle, "name")
            name.text = "%d" % i

        TimePoints = ET.SubElement(SequenceDescription, "Timepoints")
        TimePoints.set("type", "range")
        first = ET.SubElement(TimePoints, "first")
        first.text = "%d" % p.begin
        last = ET.SubElement(TimePoints, "last")
        last.text = "%d" % p.end
        ViewRegistrations = ET.SubElement(SpimData, "ViewRegistrations")
        for t in range(p.begin, p.end + 1):
            self.do_ViewRegistration(ViewRegistrations, p, t, 0)
            for a in range(len(p.flo_voxels)):
                self.do_ViewRegistration(ViewRegistrations, p, t, a + 1)

        with open(p.out_bdv, "w") as f:
            f.write(self.prettify(SpimData))
            f.close()

    def run_trsf(self):
        """
        Start the Spatial registration after having informed the parameter files
        """
        for p in self.params:
            try:
                print("Starting experiment")
                print(p)
                self.prepare_paths(p)
                if p.compute_trsf or p.test_init:
                    self.compute_trsfs(p)
                if p.apply_trsf or p.test_init:
                    if not (p.begin is None and p.end is None):
                        for t in range(p.begin, p.end + 1):
                            self.apply_trsf(p, t)
                    else:
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
            self.params = SpatialRegistration.read_param_file(params)
        else:
            self.params = params
        if self.params is not None and 0 < len(self.params):
            self.path_to_bin = self.params[0].path_to_bin


def spatial_registration():
    reg = SpatialRegistration()
    reg.run_trsf()
