#updated and clarified 22/03/23 for valentin
#windows version works on registration-env

import tifffile
import numpy as np
import json
import os
from pathlib import Path
import registrationtools
from glob import glob

##PARAMETERS
path_to_bin = "C:/Users/gros/Anaconda3/envs/registration-test_env/Library/bin/" #needs forward slashes everywhere plus one at the end of the path
main_folder = rf"C:\Users\gros\Desktop\DATA\Registration\pulse"
folder_output = rf"C:/Users/gros/Desktop/DATA/Registration/Valentin_output/"
path_to_json = rf'C:/Users/gros/Desktop/CODES/Alice_Registration/json_files/spatial.json' #where the json file is going to be saved (just for safety checks)
ch_ref = "dapi"
### give below the list of the channels that are going to be registered, with the first one being the reference channel
channels = ['dapi']
channels_float=channels.copy()
channels_float.remove(ch_ref)
input_voxel = [0.62, 0.62, 1]
output_voxel = [1, 1, 1] #voxel size of the output images, XYZ
init_trsfs= [["flip", "Y", "flip", "Z", "trans", "Z", -10]]#,"rot","Y",-20]]#careful : translation is positive if the fisrt position is on top of the second position
axis = 0
print(registrationtools)
#number of the samples registered

register_spatial = True
check_napari = True

def register(path_data:str,path_to_bin:str,name:str,channel:str,input_voxel:tuple=[1,1,1],
            output_voxel:tuple=[1,1,1],compute_trsf:int=1) :
    data = {
            "path_to_bin": path_to_bin,
            "path_to_data": rf"{path_data}/stacks_bychannel/",
            "ref_im": rf"{name}_bot_{channel}.tif",
            "flo_ims": [rf"{name}_top_{channel}.tif"
            ],
            "compute_trsf": compute_trsf
            ,
            "init_trsfs":init_trsfs,
            "trsf_paths": [rf"{path_data}/trsf/"],
            "trsf_types": ["rigid3D"],
            "ref_voxel": input_voxel,
            "flo_voxels": [ input_voxel],
            "out_voxel":output_voxel,
            "test_init": 0,
            "apply_trsf": 1,
            "out_pattern": rf"{path_data}/registered",
            "begin" : 1,
            "end":1,
            "bbox_out": 0,
            "registration_depth":1,
        }

    # #saves the json file
    json_string=json.dumps(data)
    with open(path_to_json,'w') as outfile :
        outfile.write(json_string)

    tr = registrationtools.SpatialRegistration(data)
    tr.run_trsf()


list_folder_day = [f.path for f in os.scandir(main_folder) if f.is_dir()]
for folder_day in list_folder_day :
    print('Registering day',folder_day)
    list_folder_experiment = [rf'C:\Users\gros\Desktop\DATA\Registration\pulse\20230101\test_seuils'] ##to select one experiment
    for folder_experiment in list_folder_experiment :
        name_experiment = os.path.basename(folder_experiment)
        print('Name of the experiment :',name_experiment)

        # list_gastruloids=['g'+str(i+1) for i in range(len(list_bottom))]
        list_gastruloids = ['g1']
        for ind_g,name_g in enumerate(list_gastruloids) :
            print('gastruloid',name_g)

            folder_gastruloid = rf'{folder_experiment}/{name_g}/'
            subfolder_output = rf'{folder_output}/{name_experiment}_{name_g}/'

            if register_spatial :
            #registering the reference channel
                register(path_data=folder_gastruloid,path_to_bin=path_to_bin,name=name_g,channel=ch_ref,input_voxel=input_voxel,
                        output_voxel=output_voxel,compute_trsf=1)

                for channel in channels_float : #loop to register every channels in the list one by one
                    print(channel)
                    compute_transf = 0
                    #We just apply the previous transformation for a "secondary" channel with the same transformations, compute_transform=0. 
                    register(path_data=folder_gastruloid,path_to_bin=path_to_bin,name=name_g,channel=channel,input_voxel=input_voxel,
                        output_voxel=output_voxel,compute_trsf=0) 

