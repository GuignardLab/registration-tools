#!/usr/bin/python
# This file is subject to the terms and conditions defined in
# file 'LICENCE', which is part of this source code package.
# Author: Leo Guignard (guignardl...@AT@...janelia.hhmi.org)

from time import time
import os
import scipy as sp
from scipy import interpolate
import json
import numpy as np
from IO import imread, imsave, SpatialImage
from pyklb import readheader
from statsmodels.nonparametric.smoothers_lowess import lowess
import sys

class trsf_parameters(object):
    """docstring for trsf_parameters"""
    def check_parameters_consistancy(self):
        correct = True
        if not 'path_to_data' in self.__dict__:
            print '\n\t"path_to_data" is required'
            correct = False
        if not 'file_name' in self.__dict__:
            print '\n\t"file_name" is required'
            correct = False
        if not 'trsf_folder' in self.__dict__:
            print '\n\t"trsf_folder" is required'
            correct = False
        if not 'voxel_size' in self.__dict__:
            print '\n\t"voxel_size" is required'
            correct = False
        if not 'ref_TP' in self.__dict__:
            print '\n\t"ref_TP" is required'
            correct = False            
        if (self.apply_trsf and
            self.output_format is None and
            self.suffix is None):
            print '\n\tEither "output_format" or "suffix" has to be specified'
            correct = False
        if (self.lowess_interpolation and
            self.trsf_type!='translation'):
            print '\n\tLowess interpolation only works with translation'
            correct = False
        if (self.lowess_interpolation and
             self.ref_path is None):
            print '\n\tLowess interpolation only works with a defined reference image'
            correct = False
        if (self.lowess_interpolation and
            not 'window_size' in self.param_dict):
            print '\n\tLowess interpolation "window_size" is missing'
            print '\tdefault value of 5 will be used\n'
        if (self.lowess_interpolation and
            not 'step_size' in self.param_dict):
            print '\n\tLowess interpolation "step_size" is missing'
            print '\tdefault value of 100 will be used\n'
        if self.suffix is None and self.output_format is None:
            print '\tAt least of one the following argument has to be specified:'
            print '\t\t"suffix"', '"output_format"'
        if not(self.suffix is None or self.output_format is None):
            print('\tThe parameters "suffix" and "output_format" '+\
                  'have both been defined, "output_format" will be used:')
            print '\t'+self.output_format
        if self.trsf_type=='vectorfield' and self.ref_path is None:
            print 'Non-linear transformation asked with propagation.'
            print 'While working it is highly not recommended'
            print 'Please consider not doing a registration from propagation'
        return correct

    def __str__(self):
        max_key = max([len(k) for k in self.__dict__.iterkeys() if k!="param_dict"]) + 1
        max_tot = max([len(str(v)) for k, v in self.__dict__.iteritems()
                       if k!="param_dict"]) + 2 + max_key
        output  = 'The registration will run with the following arguments:\n'
        output += "\n" + " File format ".center(max_tot, '-') + "\n"
        output += "path_to_data".ljust(max_key, ' ') + ": {:s}\n".format(self.path_to_data)
        output += "file_name".ljust(max_key, ' ') + ": {:s}\n".format(self.file_name)
        output += "trsf_folder".ljust(max_key, ' ') + ": {:s}\n".format(self.trsf_folder)
        output += "suffix".ljust(max_key, ' ') + ": {:s}\n".format(self.suffix)
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
            output += ("lowess_interpolation".ljust(max_key, ' ') +
                       ": {:d}\n".format(self.lowess_interpolation))
            if self.lowess_interpolation:
                output += "window_size".ljust(max_key, ' ') + ": {:d}\n".format(self.window_size)
                output += "step_size".ljust(max_key, ' ') + ": {:d}\n".format(self.step_size)
            output += "recompute".ljust(max_key, ' ') + ": {:d}\n".format(self.recompute)
        output += "apply_trsf".ljust(max_key, ' ') + ": {:d}\n".format(self.apply_trsf)
        if self.apply_trsf:
            output += ("projection_path".ljust(max_key, ' ') +
                       ": {:s}\n".format(self.projection_path))
            output += ("interpolation".ljust(max_key, ' ') +
                       ": {:s}\n".format(self.interpolation))

        return output


    def __init__(self, file_name):
        with open(file_name) as f:
            param_dict = json.load(f)
            f.close()

        # Default parameters
        self.suffix = None
        self.output_format = None
        self.check_TP = None
        self.not_to_do = []
        self.compute_trsf = True
        self.ref_path = None
        self.registration_depth = 3
        self.padding = 1
        self.lowess_interpolation = False
        self.window_size = 5
        self.step_size = 100
        self.recompute = True
        self.apply_trsf = True
        self.projection_path = None
        self.sigma = 2.0
        self.keep_vectorfield = False
        self.trsf_type = 'rigid'
        self.interpolation = 'linear'
        self.path_to_bin = ''

        self.param_dict = param_dict

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
    if p.ref_path is not None:
        p_im_flo = p.A0.format(t=t2)
        p_im_ref = p.ref_path
        t_ref = p.ref_TP
        t_flo = t2
        if t1<t2 and not os.path.exists(p.trsf_folder + 't%06d-%06d.txt'%(t2, t_ref)):
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
        print 'trsf tp %d-%d not done'%(t1, t2)
        np.savetxt(p.trsf_folder + 't%06d-%06d.txt'%(t_flo, t_ref), np.identity(4))
    elif (p.recompute or
          not os.path.exists(p.trsf_folder + 't%06d-%06d.txt'%(t_flo, t_ref))):
        if p.trsf_type != 'vectorfield':
            os.system(self.path_to_bin +
                      'blockmatching -ref ' + p_im_ref + ' -flo ' + p_im_flo + \
                      ' -reference-voxel %f %f %f'%p.voxel_size + \
                      ' -floating-voxel %f %f %f'%p.voxel_size + \
                      ' -trsf-type %s -py-hl 6 -py-ll %d'%(p.trsf_type, p.registration_depth) + \
                      ' -res-trsf ' + p.trsf_folder + 't%06d-%06d.txt'%(t_flo, t_ref))
        else:
            if p.apply_trsf:
                res = ' -res ' + p.A0_out.format(t=t_flo)
            else:
                res = ''
            if p.keep_vectorfield:
                res_trsf = ' -res-trsf ' + p.trsf_folder + 't%06d-%06d.klb'%(t_flo, t_ref)
            else:
                res_trsf = ''
            os.system(self.path_to_bin +
                      'blockmatching -ref ' + p_im_ref + ' -flo ' + p_im_flo + \
                      ' -reference-voxel %f %f %f'%p.voxel_size + \
                      ' -floating-voxel %f %f %f'%p.voxel_size + \
                      ' -trsf-type affine -py-hl 6 -py-ll %d'%(p.registration_depth) + \
                      ' -res-trsf ' + p.trsf_folder + 't%06d-%06d.txt'%(t_flo, t_ref))
            os.system(self.path_to_bin +
                      'blockmatching -ref ' + p_im_ref + \
                      ' -flo ' + p_im_flo + \
                      ' -init-trsf ' + p.trsf_folder + 't%06d-%06d.txt'%(t_flo, t_ref) + \
                      res + \
                      ' -reference-voxel %f %f %f'%p.voxel_size + \
                      ' -floating-voxel %f %f %f'%p.voxel_size + \
                      ' -trsf-type %s -py-hl 6 -py-ll %d'%(p.trsf_type, p.registration_depth) + \
                      res_trsf + \
                      (' -elastic-sigma {s:.1f} {s:.1f} {s:.1f} ' + \
                       ' -fluid-sigma {s:.1f} {s:.1f} {s:.1f}').format(s=p.sigma))

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
    print '%dh:%dmin:%ds'%(hours, mins, secs)

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
        os.system(p.path_to_bin + 'composeTrsf ' + out_trsf + ' -trsfs ' + trsf_2 + ' ' + trsf_1)
    return out_trsf

def lowess_smooth_interp(X, T, frac):
    X_smoothed = lowess(X, T, frac = frac, is_sorted = True, return_sorted = False)
    return sp.interpolate.InterpolatedUnivariateSpline(T, X_smoothed, k=1)
   
def read_param_file():
    ''' Asks for, reads and formats the parameter file
    '''
    if len(sys.argv)<2:
        p_param = raw_input('\nPlease inform the path to the json config file:\n')
        p_param = p_param.replace('"', '')
        p_param = p_param.replace("'", '')
        p_param = p_param.replace(" ", '')
    else:
        p_param = sys.argv[1]
    if os.path.isdir(p_param):
        f_names = [os.path.join(p_param, f) for f in os.listdir(p_param)
                   if '.json' in f and not '~' in f]
    else:
        f_names = [p_param]

    params = []
    for file_name in f_names:
        print ''
        print "Extraction of the parameters from file %s"%file_name
        p = trsf_parameters(file_name)
        if not p.check_parameters_consistancy():
            print "\n%s Failed the consistancy check, it will be skipped"
        else:
            params += [p]
        print ''
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
            print "The following time points are missing:"
            print "\t" + missing_time_points
            print "Aborting the process"
            exit()
    if p.ref_path is not None:
        p.ref_path = p.ref_path.format(t=p.ref_TP)
    if p.apply_trsf:
        for i, t in enumerate(sorted(p.time_points)):
            folder_tmp = os.path.split(p.A0_out.format(t=t))[0]
            if not os.path.exists(folder_tmp):
                os.makedirs(folder_tmp)

def interpolate_trsfs(p, trsf_fmt):
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
    X_smoothed = lowess_smooth_interp(X_T, T, frac = frac)
    Y_smoothed = lowess_smooth_interp(Y_T, T, frac = frac)
    Z_smoothed = lowess_smooth_interp(Z_T, T, frac = frac)

    new_trsf_fmt = 't{flo:06d}-{ref:06d}-filtered.txt'
    for t in range(min(p.time_points), max(p.time_points)+1):
        mat = np.identity(4)
        mat[0, -1] = X_smoothed(t)
        mat[1, -1] = Y_smoothed(t)
        mat[2, -1] = Z_smoothed(t)
        np.savetxt(p.trsf_folder + new_trsf_fmt.format(flo=t, ref=p.ref_TP), mat)
    trsf_fmt = new_trsf_fmt
    return trsf_fmt

def pad_trsfs(p, trsf_fmt):
    if p.ref_path is not None and p.ref_path.split('.')[-1] == 'klb':
        im_shape = readheader(p.ref_path)['imagesize_tczyx'][-1:-4:-1]
    elif p.ref_path is not None:
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

    os.system(self.path_to_bin +
              'changeMultipleTrsfs -trsf-format ' +
              p.trsf_folder + trsf_fmt_no_flo.format(ref=p.ref_TP) + \
              ' -index-reference %d -first %d -last %d '%(p.ref_TP,
                                                          min(p.time_points),
                                                          max(p.time_points)) + \
              ' -template ' + p.trsf_folder + 'tmp.klb ' + \
              ' -res ' + p.trsf_folder + new_trsf_fmt_no_flo.format(ref=p.ref_TP) + \
              ' -res-t ' + p.trsf_folder + 'template.klb ' + \
              ' -trsf-type %s -vs %f %f %f'%((p.trsf_type,)+p.voxel_size))

def compute_trsfs(p):
    # Create the output folder for the transfomrations
    if not os.path.exists(p.trsf_folder):
        os.makedirs(p.trsf_folder)
   
    max_t = max(p.time_points)
    if p.lowess_interpolation:
        p.to_register = sorted(p.time_points)[::p.step_size] + [max_t]
    else:
        p.to_register = p.time_points

    trsf_fmt = 't{flo:06d}-{ref:06d}.txt'
    try:
        run_produce_trsf(p, nb_cpu=1)
        if p.ref_path is None:
            compose_trsf(min(p.to_register), p.ref_TP,
                         p.trsf_folder, list(p.to_register))
            compose_trsf(max(p.to_register), p.ref_TP,
                         p.trsf_folder, list(p.to_register))
            np.savetxt(('{:s}'+trsf_fmt).format(p.trsf_folder,
                                                flo=p.ref_TP,
                                                ref=p.ref_TP),
                       np.identity(4))
    except Exception as e:
        print p.trsf_folder
        print e

    if p.lowess_interpolation:
        trsf_fmt = interpolate_trsfs(p, trsf_fmt)

    if p.padding:
        pad_trsfs(p, trsf_fmt)

def apply_trsf(p):
    trsf_fmt = 't{flo:06d}-{ref:06d}.txt'
    if p.lowess_interpolation:
        trsf_fmt = 't{flo:06d}-{ref:06d}-filtered.txt'
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
        os.system(self.path_to_bin +
                  "applyTrsf '%s' '%s' -trsf "%(p.A0.format(t=t), p.A0_out.format(t=t)) + \
                  p.trsf_folder + trsf_fmt.format(flo=t, ref=p.ref_TP) + \
                  ' -template ' + template + \
                  ' -floating-voxel %f %f %f '%p.voxel_size + \
                  ' -reference-voxel %f %f %f '%p.voxel_size + \
                  ' -interpolation %s'%p.interpolation)
        im = imread(p.A0_out.format(t=t))
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
    else:
        p_to_data = p.A0_out
    if not os.path.exists(p_to_data.format(t=-1)):
        os.makedirs(p_to_data.format(t=-1))
    imsave((p_to_data + f_name.replace(p.im_ext, 'xyProjection.klb')),
           SpatialImage(xy_proj))
    imsave((p_to_data + f_name.replace(p.im_ext, 'xzProjection.klb')),
           SpatialImage(xz_proj))
    imsave((p_to_data + f_name.replace(p.im_ext, 'yzProjection.klb')),
           SpatialImage(yz_proj))

if __name__ == '__main__':
    params = read_param_file()
    for p in params:
        try:
            print "Starting experiment"
            print p
            prepare_paths(p)

            if p.compute_trsf:
                compute_trsfs(p)

            if p.apply_trsf and p.trsf_type!='vectorfield':
                apply_trsf(p)
        except Exception as e:
            print 'Failure of %s'%p.origin_file_name
            print e
