#!/usr/bin/python
# This file is subject to the terms and conditions defined in
# file 'LICENCE', which is part of this source code package.
# Author: Leo Guignard (guignardl...@AT@...janelia.hhmi.org)

import numpy as np
import os
import sys
import json
from subprocess import call
import xml.etree.ElementTree as ET
from xml.dom import minidom
from shutil import copyfile
if sys.version_info[0]<3:
    from future.builtins import input

class trsf_parameters(object):
    """docstring for trsf_parameters"""
    def check_parameters_consistancy(self):
        correct = True
        if self.out_pattern == '':
            print('The output pattern cannot be an empty string')
        return correct

    def __str__(self):
        max_key = max([len(k) for k in self.__dict__.keys() if k!="param_dict"]) + 1
        # max_tot = max([len(str(v)) else max([len(str(vi)) for vi in v])
        #                for k, v in self.__dict__.items() 
        #                if k!="param_dict"]) + 2 + max_key
        output  = 'The registration will run with the following arguments:\n'
        # output += "\n" + " File format ".center(max_tot, '-') + "\n"
        output += "\n" + " File format \n"
        output += "path_to_data".ljust(max_key, ' ') + ": {:s}\n".format(self.path_to_data)
        output += "ref_im".ljust(max_key, ' ') + ": {:s}\n".format(self.ref_im)
        
        output += "flo_ims".ljust(max_key, ' ') + ": "
        tmp_just_len = len("flo_ims".ljust(max_key, ' ') + ": ")
        already = tmp_just_len + 1
        for flo in self.flo_ims:
            output += (" "*(tmp_just_len-already)) + "{:s}\n".format(flo)
            already = 0
        
        output += "init_trsfs".ljust(max_key, ' ') + ": "
        tmp_just_len = len("init_trsfs".ljust(max_key, ' ') + ": ")
        already = tmp_just_len + 1
        for init_trsfs in self.init_trsfs:
            if init_trsfs is not None and 0<len(init_trsfs):
                output += (" "*(tmp_just_len-already)) + "{:s}\n".format(init_trsfs)
            already = 0

        output += "trsf_types".ljust(max_key, ' ') + ": "
        tmp_just_len = len("trsf_types".ljust(max_key, ' ') + ": ")
        already = tmp_just_len + 1
        for trsf_type in self.trsf_types:
            output += (" "*(tmp_just_len-already)) + "{:s}\n".format(trsf_type)
            already = 0

        output += "ref_voxel".ljust(max_key, ' ') + ": {:f} x {:f} x {:f}\n".format(*self.ref_voxel)

        output += "flo_voxels".ljust(max_key, ' ') + ": "
        tmp_just_len = len("flo_voxels".ljust(max_key, ' ') + ": ")
        already = tmp_just_len + 1
        for flo_voxel in self.flo_voxels:
            output += (" "*(tmp_just_len-already)) + "{:f} x {:f} x {:f}\n".format(*flo_voxel)
            already = 0

        output += "out_voxel".ljust(max_key, ' ') + ": {:f} x {:f} x {:f}\n".format(*self.out_voxel)

        output += "out_pattern".ljust(max_key, ' ') + ": {:s}\n".format(self.out_pattern)
        
        return output


    def __init__(self, file_name):
        with open(file_name) as f:
            param_dict = json.load(f)
            f.close()

        # Default parameters
        self.init_trsfs = [[], [], []]
        self.param_dict = param_dict
        self.path_to_bin = ''
        self.registration_depth = 3
        self.init_trsf_real_unit = True
        self.image_interpolation = 'linear'
        self.apply_trsf = True
        self.compute_trsf = True
        self.test_init = False
        self.begin = None
        self.end = None
        self.trsf_types = []
        self.time_tag = 'TM'
        self.bdv_unit = 'microns'
        self.bdv_voxel_size = None
        self.do_bdv = 0
        self.flo_im_sizes = [None, None, None]
        self.copy_ref = False

        self.__dict__.update(param_dict)
        self.ref_voxel = tuple(self.ref_voxel)
        self.flo_voxels = [tuple(vox) for vox in self.flo_voxels]
        self.out_voxel = tuple(self.out_voxel)
        self.origin_file_name = file_name

def axis_rotation_matrix(axis, angle, min_space=None, max_space=None):
    """ Return the transformation matrix from the axis and angle necessary
    axis : axis of rotation ("X", "Y" or "Z")
    angle : angle of rotation (in degree)
    min_space : coordinates of the bottom point (usually (0, 0, 0))
    max_space : coordinates of the top point (usually im shape)
    """
    import math
    I = np.linalg.inv
    D = np.dot
    if axis not in ["X", "Y", "Z"]:
        raise Exception("Unknown axis : "+ str(axis))
    rads = math.radians(angle)
    s = math.sin(rads)
    c = math.cos(rads)

    centering = np.identity(4)
    if min_space is None and max_space is not None:
        min_space = np.array([0.,0.,0.])

    if max_space is not None:
        space_center = (max_space-min_space)/2.
        offset = -1.*space_center
        centering[:3,3] = offset

    rot = np.identity(4)
    if axis=="X":
        rot = np.array([ [1., 0., 0., 0.],
                            [0., c, -s,  0.],
                            [0., s,  c,  0.],
                            [0., 0., 0., 1.] ])
    elif axis=="Y":
        rot = np.array([ [c,   0., s,  0.],
                            [0.,  1., 0., 0.],
                            [-s,  0., c,  0.],
                            [0.,  0., 0., 1.] ])

    elif axis=="Z":
        rot = np.array([ [c, -s,  0., 0.],
                            [s,  c,  0., 0.],
                            [0., 0., 1., 0.],
                            [0., 0., 0., 1.] ])

    return D(I(centering), D(rot, centering))

def flip_matrix(axis, im_size):
    out = np.identity(4)
    if axis == 'X':
        out[0,  0] = -1
        out[0, -1] = im_size[0]
    if axis == 'Y':
        out[1,  1] = -1
        out[1, -1] = im_size[2]
    if axis == 'Z':
        out[2,  2] = -1
        out[2, -1] = im_size[2]
    return out

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
    else:
        f.close()
        return np.loadtxt(path)

def prepare_paths(p):
    p.ref_A = os.path.join(p.path_to_data, p.ref_im)
    p.flo_As = []
    for flo_im in p.flo_ims:
        p.flo_As += [os.path.join(p.path_to_data, flo_im)]
    if os.path.split(p.out_pattern)[0] == '':
        ext = p.ref_im.split('.')[-1]
        p.ref_out = p.ref_A.replace(ext, p.out_pattern+'.'+ext)
        p.flo_outs = []
        for flo in p.flo_As:
            p.flo_outs += [flo.replace(ext, p.out_pattern+'.'+ext)]
    else:
        if not os.path.exists(p.out_pattern):
            os.makedirs(p.out_pattern)
        p.ref_out = os.path.join(p.out_pattern, p.ref_im)
        p.flo_outs = []
        for flo in p.flo_ims:
            p.flo_outs += [os.path.join(p.out_pattern, flo)]
    if not hasattr(p, 'trsf_paths'):
        p.trsf_paths = [os.path.split(pi)[0] for pi in p.flo_outs]
        p.trsf_names = ['A{a:d}-{trsf:s}.trsf' for pi in p.flo_outs]
    else:
        formated_paths = []
        p.trsf_names = []
        for pi in p.trsf_paths:
            path, n = os.path.split(pi)
            if n == '':
                n = 'A{a:d}-{trsf:s}.trsf'
            elif not '{a:' in n:
                n += '{a:d}.trsf'
            formated_paths += [path]
            p.trsf_names += [n]
        p.trsf_paths = formated_paths
    if p.begin is None:
        s = p.ref_A.find(p.time_tag) + len(p.time_tag)
        e = s
        while p.ref_A[e].isdigit() and e<len(p.ref_A):
            e += 1
        p.begin = p.end = int(p.ref_A[s:e])
    if not hasattr(p, 'ref_im_size') or p.ref_im_size is None:
        p.ref_im_size = p.flo_im_sizes[0]
    if not hasattr(p, 'bdv_im') or p.bdv_im is None:
        p.bdv_im = [p.ref_A] + p.flo_As
    if not hasattr(p, 'out_bdv') or p.out_bdv is None:
        p.out_bdv = os.path.join(p.trsf_paths[0], 'bdv.xml')
    if p.bdv_voxel_size is None:
        p.bdv_voxel_size = p.ref_voxel

def build_init_trsf(trsf_type, axis, angle=None):
    trsf_mat = np.identity(4)
    if trsf_type=='flip':
        if axis.lower()=='x':
            trsf_mat[0, 0]=-1
            trsf_mat[0, -1]=-1

def inv_trsf(trsf):
    return np.linalg.lstsq(trsf, np.identity(4))[0]

def vox_to_real(trsf, flo_vs, ref_vs):
    H_ref = [[ref_vs[0], 0        , 0        , 0],
             [0        , ref_vs[1], 0        , 0],
             [0        , 0        , ref_vs[2], 0],
             [0        , 0        , 0        , 1]]
    H_ref_inv = inv_trsf(H_ref)
    return (np.dot(trsf, H_ref_inv))

def compute_trsfs(p):
    for A_num, flo_A in enumerate(p.flo_As):
        flo_voxel = p.flo_voxels[A_num]
        init_trsf = p.init_trsfs[A_num]
        trsf_path = p.trsf_paths[A_num]
        trsf_name = p.trsf_names[A_num]
        flo_out = p.flo_outs[A_num]
        if isinstance(init_trsf, list):
            i = 0
            trsfs = []
            im_size = np.array(p.flo_im_sizes[A_num], dtype=float)*flo_voxel
            while i < len(init_trsf):
                t_type = init_trsf[i]
                i+=1
                axis = init_trsf[i]
                i+=1
                if 'rot' in t_type:
                    angle = init_trsf[i]
                    i+=1
                    trsfs += [axis_rotation_matrix(axis.upper(), angle,
                                                   np.zeros(3), im_size)]
                else:
                    trsfs += [flip_matrix(axis.upper(), im_size)]
            res = np.identity(4)
            for trsf in trsfs:
                res = np.dot(res, trsf)
            if not os.path.exists(trsf_path):
                os.makedirs(trsf_path)
            init_trsf = os.path.join(trsf_path, 'A{:d}-init.trsf'.format(A_num+1))
            np.savetxt(init_trsf, res)
        elif not p.init_trsf_real_unit and init_trsf is not None:
            tmp = vox_to_real(inv_trsf(read_trsf(init_trsf)), flo_voxel, p.ref_voxel)
            init_ext = init_trsf.split('.')[-1]
            init_trsf = init_trsf.replace(init_ext, 'real.txt')
            np.savetxt(init_trsf, inv_trsf(tmp))
        if init_trsf is not None:
            init_trsf_command = ' -init-trsf {:s}'.format(init_trsf)
        else:
            init_trsf_command = ''
        i=0
        if not p.test_init:
            for i, trsf_type in enumerate(p.trsf_types[:-1]):
                if i!=0:
                    init_trsf_command = ' -init-trsf {:s}'.format(os.path.join(trsf_path, res_trsf))
                res_trsf = os.path.join(trsf_path, trsf_name.format(a=A_num+1,
                                                                    trsf=trsf_type))
                call(p.path_to_bin +
                     'blockmatching -ref ' + p.ref_A.format(t=p.begin) + \
                     ' -flo ' + flo_A.format(t=p.begin) + \
                     ' -reference-voxel %f %f %f'%p.ref_voxel + \
                     ' -floating-voxel %f %f %f'%flo_voxel + \
                     ' -trsf-type %s -py-hl 6 -py-ll %d'%(trsf_type, p.registration_depth) + \
                     init_trsf_command + \
                     ' -res-trsf ' + res_trsf +\
                     ' -composition-with-initial',
                     shell=True)
            trsf_type = p.trsf_types[-1]
            i = len(p.trsf_types)-1
            if i!=0:
                init_trsf_command = ' -init-trsf {:s}'.format(os.path.join(trsf_path, res_trsf))
            res_trsf = os.path.join(trsf_path,
                                    trsf_name.format(a=A_num+1,
                                                     trsf=trsf_type))
            res_inv_trsf = os.path.join(trsf_path,
                                        ('inv-' + trsf_name).format(a=A_num+1,
                                                                      trsf=trsf_type))
            call(p.path_to_bin +
                 'blockmatching -ref ' + p.ref_A.format(t=p.begin) + \
                 ' -flo ' + flo_A.format(t=p.begin) + \
                 ' -reference-voxel %f %f %f'%p.ref_voxel + \
                 ' -floating-voxel %f %f %f'%flo_voxel + \
                 ' -trsf-type %s -py-hl 6 -py-ll %d'%(trsf_type, p.registration_depth) + \
                 init_trsf_command + \
                 ' -res-trsf ' + res_trsf +\
                 # ' -res-voxel-trsf ' + res_voxel_trsf + \
                 # ' -res ' + flo_out +\
                 ' -composition-with-initial',
                 shell=True)
            call(p.path_to_bin +\
                 'invTrsf %s %s'%(res_trsf, res_inv_trsf),
                 shell=True)

def apply_trsf(p, t=None):
    if  p.out_voxel != p.ref_voxel:
        call(p.path_to_bin +
             'applyTrsf' +
             ' -flo ' + p.ref_A.format(t=t) +
             ' -res ' + p.ref_out.format(t=t) +
             ' -floating-voxel %f %f %f'%p.ref_voxel+
             ' -vs %f %f %f'%p.out_voxel,
             shell=True)
    elif p.copy_ref:
        copyfile(p.ref_A.format(t=t), p.ref_out.format(t=t))
        # call(p.path_to_bin +
        #     'copy %s %s'%(p.ref_A.format(t=t), p.ref_out.format(t=t)),
        #      shell=True)
    else:
        p.ref_out = p.ref_A
    for A_num, flo_A in enumerate(p.flo_As):
        out_voxel = p.out_voxel
        flo_voxel = p.flo_voxels[A_num]
        trsf_path = p.trsf_paths[A_num]
        trsf_name = p.trsf_names[A_num]
        init_trsf = p.init_trsfs[A_num]
        if p.test_init:
            if isinstance(init_trsf, list):
                trsf = os.path.join(trsf_path, 'A{:d}-init.trsf'.format(A_num+1))
            else:
                trsf = init_trsf
        else:
            t_type = '' if len(p.trsf_types)<1 else p.trsf_types[-1]
            trsf = os.path.join(trsf_path,
                                trsf_name.format(a=A_num+1,
                                                 trsf=t_type))
        flo_out = p.flo_outs[A_num]
        call(p.path_to_bin +
             'applyTrsf' +
             ' -flo ' + flo_A.format(t=t) +
             ' -floating-voxel %f %f %f'%flo_voxel +
             ' -res ' + flo_out.format(t=t) +
             ' -ref ' + p.ref_out.format(t=t) +
             ' -reference-voxel %f %f %f'%p.out_voxel +
             ' -trsf ' + trsf +
             ' -interpolation %s'%p.image_interpolation,
             shell=True)

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

def do_ViewRegistration(ViewRegistrations, p, t, a):
    ViewRegistration = ET.SubElement(ViewRegistrations, 'ViewRegistration')
    ViewRegistration.set('timepoint', '%d'%t)
    ViewRegistration.set('setup', '%d'%a)
    ViewTransform = ET.SubElement(ViewRegistration, 'ViewTransform')
    ViewTransform.set('type', 'affine')
    if a != 0:
        affine = ET.SubElement(ViewTransform, 'affine')
        trsf_path = p.trsf_paths[a-1]
        trsf_name = p.trsf_names[a-1]
        trsf_type = p.trsf_types[-1]
        f = os.path.join(trsf_path, ('inv-' + trsf_name).format(a=a, trsf=trsf_type))
        trsf = read_trsf(f)
        formated_trsf = tuple(trsf[:-1,:].flatten())
        affine.text = ('%f '*12)%formated_trsf
        ViewTransform = ET.SubElement(ViewRegistration, 'ViewTransform')
        ViewTransform.set('type', 'affine')
        affine = ET.SubElement(ViewTransform, 'affine')
        affine.text = '%f 0.0 0.0 0.0 0.0 %f 0.0 0.0 0.0 0.0 %f 0.0'%p.flo_voxels[a-1]
    else:
        affine = ET.SubElement(ViewTransform, 'affine')
        affine.text = '%f 0.0 0.0 0.0 0.0 %f 0.0 0.0 0.0 0.0 %f 0.0'%p.ref_voxel

def build_bdv(p):
    SpimData = ET.Element('SpimData')
    SpimData.set('version', "0.2")

    base_path = ET.SubElement(SpimData, 'BasePath')
    base_path.set('type', 'relative')
    base_path.text = '.'

    SequenceDescription = ET.SubElement(SpimData, 'SequenceDescription')
    
    ImageLoader = ET.SubElement(SequenceDescription, 'ImageLoader')
    ImageLoader.set('format', "klb")
    Resolver = ET.SubElement(ImageLoader, 'Resolver')
    Resolver.set('type', "org.janelia.simview.klb.bdv.KlbPartitionResolver")
    for im_p in p.bdv_im:
        ViewSetupTemplate = ET.SubElement(Resolver, 'ViewSetupTemplate')
        template = ET.SubElement(ViewSetupTemplate, 'template')
        template.text = im_p
        timeTag = ET.SubElement(ViewSetupTemplate, 'timeTag')
        timeTag.text = p.time_tag

    ViewSetups = ET.SubElement(SequenceDescription, 'ViewSetups')
    ViewSetup = ET.SubElement(ViewSetups, 'ViewSetup')
    i = 0
    do_viewSetup(ViewSetup, p, p.ref_im_size, i)
    for i, pi in enumerate(p.flo_voxels):
        ViewSetup = ET.SubElement(ViewSetups, 'ViewSetup')
        do_viewSetup(ViewSetup, p, p.flo_im_sizes[i], i+1)

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
    for i in range(len(p.flo_voxels)+1):
        Angle = ET.SubElement(Attributes, 'Angle')
        id_ = ET.SubElement(Angle, 'id')
        id_.text = '%d'%i
        name = ET.SubElement(Angle, 'name')
        name.text = '%d'%i

    TimePoints = ET.SubElement(SequenceDescription, 'Timepoints')
    TimePoints.set('type', 'range')
    first = ET.SubElement(TimePoints, 'first')
    first.text = '%d'%p.begin
    last = ET.SubElement(TimePoints, 'last')
    last.text = '%d'%p.end
    ViewRegistrations = ET.SubElement(SpimData, 'ViewRegistrations')
    for t in range(p.begin, p.end+1):
        do_ViewRegistration(ViewRegistrations, p, t, 0)
        for a in range(len(p.flo_voxels)):
            do_ViewRegistration(ViewRegistrations, p, t, a+1)

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
            if p.compute_trsf or p.test_init:
                compute_trsfs(p)
            if p.apply_trsf or p.test_init:
                if not(p.begin is None and p.end is None):
                    for t in range(p.begin, p.end+1):
                        apply_trsf(p, t)
                else:
                    apply_trsf(p)
            if p.do_bdv:
                build_bdv(p)
        except Exception as e:
            print('Failure of %s'%p.origin_file_name)
            print(e)
