#!/usr/bin/python
# This file is subject to the terms and conditions defined in
# file 'LICENCE', which is part of this source code package.
# Author: Leo Guignard (guignardl...@AT@...janelia.hhmi.org)

from time import time
import os
from subprocess import call
import scipy as sp
from scipy import interpolate
import json
import numpy as np
from IO import imread, imsave, SpatialImage
from pyklb import readheader
from statsmodels.nonparametric.smoothers_lowess import lowess
import sys
if sys.version_info[0]<3:
    from future.builtins import input
import xml.etree.ElementTree as ET
from xml.dom import minidom

class trsf_parameters(object):
    """docstring for trsf_parameters"""
    def check_parameters_consistancy(self):
        correct = True
        if not 'path_to_data' in self.__dict__:
            print('\n\t"path_to_data" is required')
            correct = False
        if not 'file_name' in self.__dict__:
            print('\n\t"file_name" is required')
            correct = False
        if not 'trsf_folder' in self.__dict__:
            print('\n\t"trsf_folder" is required')
            correct = False
        if not 'voxel_size' in self.__dict__:
            print('\n\t"voxel_size" is required')
            correct = False
        if not 'ref_TP' in self.__dict__:
            print('\n\t"ref_TP" is required')
            correct = False
        if not 'projection_path' in self.__dict__:
            print('\n\t"projection_path" is required')
            correct = False
        if (self.apply_trsf and
            not 'output_format' in self.__dict__):
            print('\n\t"output_format" is required')
            correct = False
        if (self.trsf_type!='translation' and 
            (self.lowess or self.trsf_interpolation)):
            print('\n\tLowess or transformation interplolation')
            print('\tonly work with translation')
            correct = False
        if (self.lowess and
            not 'window_size' in self.param_dict):
            print('\n\tLowess smoothing "window_size" is missing')
            print('\tdefault value of 5 will be used\n')
        if (self.trsf_interpolation and
            not 'step_size' in self.param_dict):
            print('\n\tTransformation interpolation "step_size" is missing')
            print('\tdefault value of 100 will be used\n')
        if self.trsf_type=='vectorfield' and self.ref_path is None:
            print('\tNon-linear transformation asked with propagation.')
            print('\tWhile working it is highly not recommended')
            print('\tPlease consider not doing a registration from propagation')
            print('\tTHIS WILL LITERALLY TAKE AGES!! IF IT WORKS AT ALL ...')
        if not isinstance(self.spline, int) or self.spline<1 or 5<self.spline:
            out = ('{:d}' if isinstance(self.spline, int) else '{:s}').format(self.spline)
            print('The degree of smoothing for the spline interpolation')
            print('Should be an Integer between 1 and 5, you gave ' + out)
        return correct

    def __str__(self):
        max_key = max([len(k) for k in self.__dict__.keys() if k!="param_dict"]) + 1
        max_tot = max([len(str(v)) for k, v in self.__dict__.items()
                       if k!="param_dict"]) + 2 + max_key
        output  = 'The registration will run with the following arguments:\n'
        output += "\n" + " File format ".center(max_tot, '-') + "\n"
        output += "path_to_data".ljust(max_key, ' ') + ": {:s}\n".format(self.path_to_data)
        output += "file_name".ljust(max_key, ' ') + ": {:s}\n".format(self.file_name)
        output += "trsf_folder".ljust(max_key, ' ') + ": {:s}\n".format(self.trsf_folder)
        output += "output_format".ljust(max_key, ' ') + ": {:s}\n".format(self.output_format)
        output += "check_TP".ljust(max_key, ' ') + ": {:d}\n".format(self.check_TP)
        output += "\n" + " Time series properties ".center(max_tot, '-') + "\n"
        output += "voxel_size".ljust(max_key, ' ') + ": {:f}x{:f}x{:f}\n".format(*self.voxel_size)
        output += "first".ljust(max_key, ' ') + ": {:d}\n".format(self.first)
        output += "last".ljust(max_key, ' ') + ": {:d}\n".format(self.last)
        output += "\n" + " Registration ".center(max_tot, '-') + "\n"
        output += "compute_trsf".ljust(max_key, ' ') + ": {:d}\n".format(self.compute_trsf)
        if self.compute_trsf:
            output += "ref_TP".ljust(max_key, ' ') + ": {:d}\n".format(self.ref_TP)
            if self.ref_path is not None:
                output += "ref_path".ljust(max_key, ' ') + ": {:s}\n".format(self.ref_path)
            output += "trsf_type".ljust(max_key, ' ') + ": {:s}\n".format(self.trsf_type)
            if self.trsf_type == 'vectorfield':
                output += "sigma".ljust(max_key, ' ') + ": {:.2f}\n".format(self.sigma)
            output += "padding".ljust(max_key, ' ') + ": {:d}\n".format(self.padding)
            output += ("lowess".ljust(max_key, ' ') +
                       ": {:d}\n".format(self.lowess))
            if self.lowess:
                output += "window_size".ljust(max_key, ' ') + ": {:d}\n".format(self.window_size)
            output += ("trsf_interpolation".ljust(max_key, ' ') +
                       ": {:d}\n".format(self.trsf_interpolation))
            if self.trsf_interpolation:
                output += "step_size".ljust(max_key, ' ') + ": {:d}\n".format(self.step_size)
            output += "recompute".ljust(max_key, ' ') + ": {:d}\n".format(self.recompute)
        output += "apply_trsf".ljust(max_key, ' ') + ": {:d}\n".format(self.apply_trsf)
        if self.apply_trsf:
            if self.projection_path is not None:
                output += ("projection_path".ljust(max_key, ' ') +
                           ": {:s}\n".format(self.projection_path))
            output += ("image_interpolation".ljust(max_key, ' ') +
                       ": {:s}\n".format(self.image_interpolation))

        return output


    def __init__(self, file_name):
        with open(file_name) as f:
            param_dict = json.load(f)
            f.close()

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
        self.trsf_type = 'rigid'
        self.image_interpolation = 'linear'
        self.path_to_bin = ''
        self.spline = 1
        self.trsf_interpolation = False
        self.sequential = True
        self.time_tag = 'TM'
        self.do_bdv = 0
        self.bdv_voxel_size = None
        self.bdv_unit = 'microns'
        self.projection_path = None

        self.param_dict = param_dict
        if 'registration_depth' in param_dict:
            self.__dict__['registration_depth_start'] = 6
            self.__dict__['registration_depth_end'] = param_dict['registration_depth']

        self.__dict__.update(param_dict)
        self.voxel_size = tuple(self.voxel_size)
        self.origin_file_name = file_name

def read_trsf(path):
    ''' Read a transformation from a text file
        Args:
            path: string, path to a transformation
    '''
    f = open(path)
    if f.read()[0] == '(':
        f.close()
        f = open(path)
        lines = f.readlines()[2:-1]
        f.close()
        return np.array([[float(v) for v in l.split()]  for l in lines])
    f.close()
    return np.loadtxt(path)

def produce_trsf(params):
    ''' Given an output path, an image path format, a reference time, two time points,
        two locks on their respective images, an original voxel size and a voxel size,
        compute the transformation that registers together two consecutive in time images.
        The registration is done by cross correlation of the MIP along the 3 main axes.
        This function is meant to be call by multiprocess.Pool
    '''
    (p, t1, t2, make) = params
    if not p.sequential:
        p_im_flo = p.A0.format(t=t2)
        p_im_ref = p.ref_path
        t_ref = p.ref_TP
        t_flo = t2
        if t1 == p.to_register[0]: #t1<t2 and not os.path.exists(p.trsf_folder + 't%06d-%06d.txt'%(t2, t_ref)):
            produce_trsf((p, t2, t1, make))
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
        print('trsf tp %d-%d not done'%(t1, t2))
        np.savetxt(p.trsf_folder + 't%06d-%06d.txt'%(t_flo, t_ref), np.identity(4))
    elif t_flo !=  t_ref and (p.recompute or
          not os.path.exists(p.trsf_folder + 't%06d-%06d.txt'%(t_flo, t_ref))):
        if p.trsf_type != 'vectorfield':
            call(p.path_to_bin +
                 'blockmatching -ref ' + p_im_ref + ' -flo ' + p_im_flo + \
                 ' -reference-voxel %f %f %f'%p.voxel_size + \
                 ' -floating-voxel %f %f %f'%p.voxel_size + \
                 ' -trsf-type %s -py-hl %d -py-ll %d'%(p.trsf_type,
                                                       p.registration_depth_start,
                                                       p.registration_depth_end) + \
                 ' -res-trsf ' + p.trsf_folder + 't%06d-%06d.txt'%(t_flo, t_ref),
                 shell=True)
        else:
            if p.apply_trsf:
                res = ' -res ' + p.A0_out.format(t=t_flo)
            else:
                res = ''
            if p.keep_vectorfield:
                res_trsf = ' -res-trsf ' + p.trsf_folder + 't%06d-%06d.klb'%(t_flo, t_ref)
            else:
                res_trsf = ''
            call(p.path_to_bin +
                 'blockmatching -ref ' + p_im_ref + ' -flo ' + p_im_flo + \
                 ' -reference-voxel %f %f %f'%p.voxel_size + \
                 ' -floating-voxel %f %f %f'%p.voxel_size + \
                 ' -trsf-type affine -py-hl %d -py-ll %d'%(p.registration_depth_start,
                                                           p.registration_depth_end) + \
                 ' -res-trsf ' + p.trsf_folder + 't%06d-%06d.txt'%(t_flo, t_ref),
                 shell=True)
            call(p.path_to_bin +
                 'blockmatching -ref ' + p_im_ref + \
                 ' -flo ' + p_im_flo + \
                 ' -init-trsf ' + p.trsf_folder + 't%06d-%06d.txt'%(t_flo, t_ref) + \
                 res + \
                 ' -reference-voxel %f %f %f'%p.voxel_size + \
                 ' -floating-voxel %f %f %f'%p.voxel_size + \
                 ' -trsf-type %s -py-hl %d -py-ll %d'%(p.trsf_type,
                                                       p.registration_depth_start,
                                                       p.registration_depth_end) + \
                 res_trsf + \
                 (' -elastic-sigma {s:.1f} {s:.1f} {s:.1f} ' + \
                  ' -fluid-sigma {s:.1f} {s:.1f} {s:.1f}').format(s=p.sigma),
                 shell=True)

def run_produce_trsf(p, nb_cpu=1):
    ''' Parallel processing of the transformations from t to t-1/t-1 to t (depending on t<r)
        The transformation is computed using blockmatching algorithm
        Args:
            p: string, path pattern to the images to register
            r: int, reference time point
            nb_times (not used): int, number of time points on which to apply the transformation
            trsf_p: string, path to the transformation
            tp_list: [int, ], list of time points on which to apply the transformation
            ors: float, original aspect ratio
            first_TP (not used): int, first time point on which to apply the transformation
            vs: float, aspect ratio
            nb_cpy: int, number of cpus to use
    '''
    mapping = [(p, t1, t2, not (t1 in p.not_to_do or t2 in p.not_to_do))
               for t1, t2 in zip(p.to_register[:-1], p.to_register[1:])]
    tic = time()
    tmp = []
    for mi in mapping:
        tmp += [produce_trsf(mi)]
    tac = time()
    whole_time = tac - tic
    secs = whole_time%60
    whole_time = whole_time//60
    mins = whole_time%60
    hours = whole_time//60
    print('%dh:%dmin:%ds'%(hours, mins, secs))

def compose_trsf(flo_t, ref_t, trsf_p, tp_list):
    ''' Recusrively build the transformation that allows
        to register time `flo_t` onto the frame of time `ref_t`
        assuming that it exists the necessary intermediary transformations
        Args:
            flo_t: int, time of the floating image
            ref_t: int, time of the reference image
            trsf_p: string, path to folder containing the transformations
            tp_list: [int, ], list of time points that have been processed
        Returns:
            out_trsf: string, path to the result composed transformation
    '''
    out_trsf = trsf_p + 't%06d-%06d.txt'%(flo_t, ref_t)
    if not os.path.exists(out_trsf):
        flo_int = tp_list[tp_list.index(flo_t) + np.sign(ref_t - flo_t)]
        # the call is recursive, to build `T_{flo\leftarrow ref}`
        # we need `T_{flo+1\leftarrow ref}` and `T_{flo\leftarrow ref-1}`
        trsf_1 = compose_trsf(flo_int, ref_t, trsf_p, tp_list)
        trsf_2 = compose_trsf(flo_t, flo_int, trsf_p, tp_list)
        call(p.path_to_bin + 'composeTrsf ' + out_trsf + ' -trsfs ' + trsf_2 + ' ' + trsf_1, shell=True)
    return out_trsf

def lowess_smooth(p, X, T, frac):
    return lowess(X, T, frac = frac, is_sorted = True, return_sorted = False)

def interpolation(p, X, T):
    return sp.interpolate.InterpolatedUnivariateSpline(T, X, k=p.spline)

def read_param_file():
    ''' Asks for, reads and formats the parameter file
    '''
    if len(sys.argv)<2:
        p_param = input('\nPlease inform the path to the json config file:\n')
    else:
        p_param = sys.argv[1]
    stable = False
    while not stable:
        tmp = p_param.strip('"').strip("'").strip(' ')
        stable = tmp==p_param
        p_param = tmp
    if os.path.isdir(p_param):
        f_names = [os.path.join(p_param, f) for f in os.listdir(p_param)
                   if '.json' in f and not '~' in f]
    else:
        f_names = [p_param]

    params = []
    for file_name in f_names:
        print('')
        print("Extraction of the parameters from file %s"%file_name)
        p = trsf_parameters(file_name)
        if not p.check_parameters_consistancy():
            print("\n%s Failed the consistancy check, it will be skipped")
        else:
            params += [p]
        print('')
    return params

def prepare_paths(p):
    ### Check the file names and folders:
    p.im_ext = p.file_name.split('.')[-1] # Image type
    p.A0 = os.path.join(p.path_to_data, p.file_name) # Image path
    # Output format
    if p.output_format is not None:
        if os.path.split(p.output_format)[0] == '':
            p.A0_out = os.path.join(p.path_to_data, p.output_format)
        elif os.path.split(p.output_format)[1] == '':
            p.A0_out = os.path.join(p.output_format, p.file_name)
        else:
            p.A0_out = p.output_format
    else:
        p.A0_out = p.path_to_data + p.file_name.replace(p.im_ext, p.suffix + '.' + p.im_ext)

    # Time points to work with
    p.time_points = np.array([i for i in np.arange(p.first, p.last + 1)
                                  if not i in p.not_to_do])
    if p.check_TP:
        missing_time_points = []
        for t in p.time_points:
            if not os.path.exists(p.A0.format(t=t)):
                missing_time_points += [t]
        if len(missing_time_points)!=0:
            print("The following time points are missing:")
            print("\t" + missing_time_points)
            print("Aborting the process")
            exit()
    if not p.sequential:
        p.ref_path = p.A0.format(t=p.ref_TP)
    if p.apply_trsf:
        for i, t in enumerate(sorted(p.time_points)):
            folder_tmp = os.path.split(p.A0_out.format(t=t))[0]
            if not os.path.exists(folder_tmp):
                os.makedirs(folder_tmp)

    max_t = max(p.time_points)
    if p.trsf_interpolation:
        p.to_register = sorted(p.time_points)[::p.step_size]
        if not max_t in p.to_register:
            p.to_register += [max_t]
    else:
        p.to_register = p.time_points
    if not hasattr(p, 'bdv_im') or p.bdv_im is None:
        p.bdv_im = p.A0
    if not hasattr(p, 'out_bdv') or p.out_bdv is None:
        p.out_bdv = os.path.join(p.trsf_folder, 'bdv.xml')
    if p.bdv_voxel_size is None:
        p.bdv_voxel_size = p.voxel_size

def lowess_filter(p, trsf_fmt):
    X_T = []
    Y_T = []
    Z_T = []
    T = []
    for t in p.to_register:
        trsf_p = p.trsf_folder + trsf_fmt.format(flo=t, ref=p.ref_TP)
        trsf = read_trsf(trsf_p)
        T += [t]
        X_T += [trsf[0, -1]]
        Y_T += [trsf[1, -1]]
        Z_T += [trsf[2, -1]]

    frac = float(p.window_size)/len(T)
    X_smoothed = lowess_smooth(p, X_T, T, frac = frac)
    Y_smoothed = lowess_smooth(p, Y_T, T, frac = frac)
    Z_smoothed = lowess_smooth(p, Z_T, T, frac = frac)

    new_trsf_fmt = 't{flo:06d}-{ref:06d}-filtered.txt'
    for i, t in enumerate(T):
        mat = np.identity(4)
        mat[0, -1] = X_smoothed[i]
        mat[1, -1] = Y_smoothed[i]
        mat[2, -1] = Z_smoothed[i]
        np.savetxt(p.trsf_folder + new_trsf_fmt.format(flo=t, ref=p.ref_TP), mat)
    return new_trsf_fmt

def interpolate(p, trsf_fmt, new_trsf_fmt=None):
    if new_trsf_fmt is None:
        new_trsf_fmt = 't{flo:06d}-{ref:06d}-interpolated.txt'
    X_T = []
    Y_T = []
    Z_T = []
    T = []
    for t in p.to_register:
        trsf_p = p.trsf_folder + trsf_fmt.format(flo=t, ref=p.ref_TP)
        trsf = read_trsf(trsf_p)
        T += [t]
        X_T += [trsf[0, -1]]
        Y_T += [trsf[1, -1]]
        Z_T += [trsf[2, -1]]

    X_interp = interpolation(p, X_T, T)
    Y_interp = interpolation(p, Y_T, T)
    Z_interp = interpolation(p, Z_T, T)
    for t in p.time_points:
        mat = np.identity(4)
        mat[0, -1] = X_interp(t)
        mat[1, -1] = Y_interp(t)
        mat[2, -1] = Z_interp(t)
        np.savetxt(p.trsf_folder + new_trsf_fmt.format(flo=t, ref=p.ref_TP), mat)
    trsf_fmt = new_trsf_fmt
    return trsf_fmt

def pad_trsfs(p, trsf_fmt):
    if not p.sequential and p.ref_path.split('.')[-1] == 'klb':
        im_shape = readheader(p.ref_path)['imagesize_tczyx'][-1:-4:-1]
    elif not p.sequential:
        im_shape = imread(p.ref_path).shape
    elif p.A0.split('.')[-1] == 'klb':
        im_shape = readheader(p.A0.format(t=p.ref_TP))['imagesize_tczyx'][-1:-4:-1]
    else:
        im_shape = imread(p.A0.format(t=p.ref_TP)).shape
    im = SpatialImage(np.ones(im_shape), dtype=np.uint8)
    im.voxelsize = p.voxel_size
    imsave(p.trsf_folder + 'tmp.klb', im)
    identity = np.identity(4)
   
    trsf_fmt_no_flo = trsf_fmt.replace('{flo:06d}', '%06d')
    new_trsf_fmt = 't{flo:06d}-{ref:06d}-padded.txt'
    new_trsf_fmt_no_flo = new_trsf_fmt.replace('{flo:06d}', '%06d')
    for t in p.not_to_do:
        np.savetxt(p.trsf_folder+trsf_fmt.format(flo=t, ref=p.ref_TP), identity)

    call(p.path_to_bin +
         'changeMultipleTrsfs -trsf-format ' +
         p.trsf_folder + trsf_fmt_no_flo.format(ref=p.ref_TP) + \
         ' -index-reference %d -first %d -last %d '%(p.ref_TP,
                                                     min(p.time_points),
                                                     max(p.time_points)) + \
         ' -template ' + p.trsf_folder + 'tmp.klb ' + \
         ' -res ' + p.trsf_folder + new_trsf_fmt_no_flo.format(ref=p.ref_TP) + \
         ' -res-t ' + p.trsf_folder + 'template.klb ' + \
         ' -trsf-type %s -vs %f %f %f'%((p.trsf_type,)+p.voxel_size),
         shell=True)

def compute_trsfs(p):
    # Create the output folder for the transfomrations
    if not os.path.exists(p.trsf_folder):
        os.makedirs(p.trsf_folder)

    trsf_fmt = 't{flo:06d}-{ref:06d}.txt'
    try:
        run_produce_trsf(p, nb_cpu=1)
        if p.sequential:
            if min(p.to_register) != p.ref_TP:
                compose_trsf(min(p.to_register), p.ref_TP,
                             p.trsf_folder, list(p.to_register))
            if max(p.to_register) != p.ref_TP:
                compose_trsf(max(p.to_register), p.ref_TP,
                             p.trsf_folder, list(p.to_register))
        np.savetxt(('{:s}'+trsf_fmt).format(p.trsf_folder,
                                            flo=p.ref_TP,
                                            ref=p.ref_TP),
                       np.identity(4))
    except Exception as e:
        print(p.trsf_folder)
        print(e)

    if p.lowess:
        trsf_fmt = lowess_filter(p, trsf_fmt)
    if p.trsf_interpolation:
        trsf_fmt = interpolate(p, trsf_fmt)
    if p.padding:
        pad_trsfs(p, trsf_fmt)

def apply_trsf(p):
    trsf_fmt = 't{flo:06d}-{ref:06d}.txt'
    if p.lowess:
        trsf_fmt = 't{flo:06d}-{ref:06d}-filtered.txt'
    if p.trsf_interpolation:
        trsf_fmt = 't{flo:06d}-{ref:06d}-interpolated.txt'
    if p.padding:
        trsf_fmt = 't{flo:06d}-{ref:06d}-padded.txt'
        X, Y, Z = readheader(p.trsf_folder + 'template.klb')['imagesize_tczyx'][-1:-4:-1]
        template = p.trsf_folder + 'template.klb'
    elif p.A0.split('.')[-1] == 'klb':
        X, Y, Z = readheader(p.A0.format(t=p.ref_TP))['imagesize_tczyx'][-1:-4:-1]
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
        call(p.path_to_bin +
             "applyTrsf '%s' '%s' -trsf "%(p.A0.format(t=t), p.A0_out.format(t=t)) + \
             p.trsf_folder + trsf_fmt.format(flo=t, ref=p.ref_TP) + \
             ' -template ' + template + \
             ' -floating-voxel %f %f %f '%p.voxel_size + \
             ' -reference-voxel %f %f %f '%p.voxel_size + \
             ' -interpolation %s'%p.image_interpolation,
             shell=True)
        im = imread(p.A0_out.format(t=t))
        if p.projection_path is not None:
            xy_proj[:, :, i] = SpatialImage(np.max(im, axis=2))
            xz_proj[:, :, i] = SpatialImage(np.max(im, axis=1))
            yz_proj[:, :, i] = SpatialImage(np.max(im, axis=0))
    if p.projection_path is not None:
        if not os.path.exists(p.projection_path):
            os.makedirs(p.projection_path)
        p_to_data = p.projection_path
        num_s = p.file_name.find('{')
        num_e = p.file_name.find('}')+1
        f_name = p.file_name.replace(p.file_name[num_s:num_e], '')
        if not os.path.exists(p_to_data.format(t=-1)):
            os.makedirs(p_to_data.format(t=-1))
        imsave((p_to_data + f_name.replace(p.im_ext, 'xyProjection.tif')),
               SpatialImage(xy_proj))
        imsave((p_to_data + f_name.replace(p.im_ext, 'xzProjection.tif')),
               SpatialImage(xz_proj))
        imsave((p_to_data + f_name.replace(p.im_ext, 'yzProjection.tif')),
               SpatialImage(yz_proj))

def inv_trsf(trsf):
    return np.linalg.lstsq(trsf, np.identity(4))[0]

def prettify(elem):
    """Return a pretty-printed XML string for the Element.
    """
    rough_string = ET.tostring(elem, 'utf-8')
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")

def do_viewSetup(ViewSetup, p, im_size, i):
    id_ = ET.SubElement(ViewSetup, 'id')
    id_.text = '%d'%i
    name = ET.SubElement(ViewSetup, 'name')
    name.text = '%d'%i
    size = ET.SubElement(ViewSetup, 'size')
    size.text = '%d %d %d'%tuple(im_size)
    voxelSize = ET.SubElement(ViewSetup, 'voxelSize')
    unit = ET.SubElement(voxelSize, 'unit')
    unit.text = p.bdv_unit
    size = ET.SubElement(voxelSize, 'size')
    size.text = '%f %f %f'%tuple(p.bdv_voxel_size)
    attributes = ET.SubElement(ViewSetup, 'attributes')
    illumination = ET.SubElement(attributes, 'illumination')
    illumination.text = '0'
    channel = ET.SubElement(attributes, 'channel')
    channel.text = '0'
    tile = ET.SubElement(attributes, 'tile')
    tile.text = '0'
    angle = ET.SubElement(attributes, 'angle')
    angle.text = '%d'%i

def do_ViewRegistration(ViewRegistrations, p, t):
    ViewRegistration = ET.SubElement(ViewRegistrations, 'ViewRegistration')
    ViewRegistration.set('timepoint', '%d'%t)
    ViewRegistration.set('setup', '0')
    ViewTransform = ET.SubElement(ViewRegistration, 'ViewTransform')
    ViewTransform.set('type', 'affine')
    affine = ET.SubElement(ViewTransform, 'affine')
    f = os.path.join(p.trsf_folder, 't%06d-%06d.txt'%(t, p.ref_TP))
    trsf = read_trsf(f)
    trsf = inv_trsf(trsf)
    formated_trsf = tuple(trsf[:-1,:].flatten())
    affine.text = ('%f '*12)%formated_trsf
    ViewTransform = ET.SubElement(ViewRegistration, 'ViewTransform')
    ViewTransform.set('type', 'affine')
    affine = ET.SubElement(ViewTransform, 'affine')
    affine.text = '%f 0.0 0.0 0.0 0.0 %f 0.0 0.0 0.0 0.0 %f 0.0'%p.voxel_size

def build_bdv(p):
    if not p.im_ext in ['klb']:#['tif', 'klb', 'tiff']:
        print('Image format not adapted for BigDataViewer')
        return
    SpimData = ET.Element('SpimData')
    SpimData.set('version', "0.2")
    SpimData.set('encoding', "UTF-8")

    base_path = ET.SubElement(SpimData, 'BasePath')
    base_path.set('type', 'relative')
    base_path.text = '.'

    SequenceDescription = ET.SubElement(SpimData, 'SequenceDescription')
    
    ImageLoader = ET.SubElement(SequenceDescription, 'ImageLoader')
    ImageLoader.set('format', p.im_ext)
    Resolver = ET.SubElement(ImageLoader, 'Resolver')
    Resolver.set('type', "org.janelia.simview.klb.bdv.KlbPartitionResolver")
    ViewSetupTemplate = ET.SubElement(Resolver, 'ViewSetupTemplate')
    template = ET.SubElement(ViewSetupTemplate, 'template')
    template.text = p.bdv_im.format(t=p.to_register[0])
    timeTag = ET.SubElement(ViewSetupTemplate, 'timeTag')
    timeTag.text = p.time_tag

    ViewSetups = ET.SubElement(SequenceDescription, 'ViewSetups')
    ViewSetup = ET.SubElement(ViewSetups, 'ViewSetup')
    if p.im_ext == 'klb':
        im_size = tuple(readheader(p.A0.format(t=p.ref_TP))['imagesize_tczyx'][-1:-4:-1])
    else:
        im_size = tuple(imread(p.A0.format(t=p.ref_TP)).shape)
    do_viewSetup(ViewSetup, p, im_size, 0)

    Attributes = ET.SubElement(ViewSetups, 'Attributes')
    Attributes.set('name', 'illumination')
    Illumination = ET.SubElement(Attributes, 'Illumination')
    id_ = ET.SubElement(Illumination, 'id')
    id_.text = '0'
    name = ET.SubElement(Illumination, 'name')
    name.text = '0'

    Attributes = ET.SubElement(ViewSetups, 'Attributes')
    Attributes.set('name', 'channel')
    Channel = ET.SubElement(Attributes, 'Channel')
    id_ = ET.SubElement(Channel, 'id')
    id_.text = '0'
    name = ET.SubElement(Channel, 'name')
    name.text = '0'

    Attributes = ET.SubElement(ViewSetups, 'Attributes')
    Attributes.set('name', 'tile')
    Tile = ET.SubElement(Attributes, 'Tile')
    id_ = ET.SubElement(Tile, 'id')
    id_.text = '0'
    name = ET.SubElement(Tile, 'name')
    name.text = '0'

    Attributes = ET.SubElement(ViewSetups, 'Attributes')
    Attributes.set('name', 'angle')
    Angle = ET.SubElement(Attributes, 'Angle')
    id_ = ET.SubElement(Angle, 'id')
    id_.text = '0'
    name = ET.SubElement(Angle, 'name')
    name.text = '0'

    TimePoints = ET.SubElement(SequenceDescription, 'Timepoints')
    TimePoints.set('type', 'range')
    first = ET.SubElement(TimePoints, 'first')
    first.text = '%d'%min(p.to_register)
    last = ET.SubElement(TimePoints, 'last')
    last.text = '%d'%max(p.to_register)
    ViewRegistrations = ET.SubElement(SpimData, 'ViewRegistrations')
    b = min(p.to_register)
    e = max(p.to_register)
    for t in range(b, e+1):
        do_ViewRegistration(ViewRegistrations, p, t)

    with open(p.out_bdv, 'w') as f:
        f.write(prettify(SpimData))
        f.close()

if __name__ == '__main__':
    params = read_param_file()
    for p in params:
        try:
            print("Starting experiment")
            print(p)
            prepare_paths(p)

            if p.compute_trsf:
                compute_trsfs(p)

            if p.apply_trsf and p.trsf_type!='vectorfield':
                apply_trsf(p)
            if p.do_bdv:
                build_bdv(p)
        except Exception as e:
            print('Failure of %s'%p.origin_file_name)
            print(e)
