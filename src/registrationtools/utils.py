import tifffile
import os
import registrationtools
import json
import numpy as np
from glob import glob
from ensure import check  ##need to pip install !
from pathlib import Path


def get_paths():
    # Ask the user for the folder containing his data, find the movies is they are in tiff format and put them in a list after confirmation
    path_correct = False
    while not path_correct:
        path_folder = input(
            "Path to the folder of the movie(s) (in tiff format only) : \n "
        )
        paths_movies = sorted(glob(rf"{path_folder}/*.tif"))
        if len(paths_movies) > 0:
            print("You have", len(paths_movies), "movie(s), which is (are) : ")
            for path_movie in paths_movies:
                print(
                    "", Path(path_movie).stem
                )  # the empty str at the begining is to align the answers one space after the questions, for readability
                path_correct = int(
                    input("Correct ? (1 for yes, 0 for no) \n ")
                )
        else:
            print("There is no tiff file in the folder. \n")

    return paths_movies


def dimensions(list_paths: list):
    # Take the path to the data and returns its size in the dimensions CTZXY. Raise error if the other movies do not have the same C ot T dimensions.
    movie = tifffile.imread(list_paths[0])
    name_movie = Path(list_paths[0]).stem
    dim_correct = False
    while not dim_correct:
        print("\nThe dimensions of ", name_movie, "are ", movie.shape, ". \n")
        dimensions = input(
            "What is the order of the dimensions (for example TZCYX or XYZT) ? T stands for Time, C for channels if your image has multiple channels, Z for depth (or number or plans) and XY is your field of view. \n "
        )
        sorted_axes = sorted(dimensions)
        if "".join(sorted_axes) == "CTXYZ":
            number_channels = movie.shape[dimensions.find("C")]
            dim_correct = True
        elif "".join(sorted_axes) == "TXYZ":
            number_channels = 1
            dim_correct = True
        elif len(sorted_axes) != len(movie.shape):
            print("Error : Number of dimensions is incorrect. \n")
            dim_correct = False
        else:
            print(
                " \n The letters you choose has to be among these letters : X,Y,Z,C,T, with XYZT mandatory, and no other letters are allowed. \nEvery letter can be included only once. \n"
            )

        number_timepoints = movie.shape[dimensions.find("T")]

        if "C" in dimensions:
            number_channels = movie.shape[dimensions.find("C")]
        else:
            number_channels = 1

        if dim_correct:
            depth = movie.shape[dimensions.find("Z")]
            size_X = movie.shape[dimensions.find("X")]
            size_Y = movie.shape[dimensions.find("Y")]

            print(
                "\nSo",
                name_movie,
                "has",
                number_channels,
                "channels, ",
                number_timepoints,
                "timepoints, the depth in z is",
                depth,
                "pixels and the XY plane measures ",
                size_X,
                "x",
                size_Y,
                "pixels.",
            )
            dim_correct = int(input("Correct ? (1 for yes, 0 for no) \n "))

    # checking if the movies have the same dimensions in C and T, otherwise the registration cannot be computed.
    if (
        "C" in dimensions
    ):  # have to check beforehand if multichannels, otherwise, dimensions.find('C') will return -1 and mov.shape[dimensions.find('C')] will probably return the dimension of the X axis
        for path in list_paths:
            mov = tifffile.imread(path)
            check(mov.shape[dimensions.find("T")]).equals(
                number_timepoints
            ).or_raise(
                Exception,
                "These movies do not all have the same number of timepoints. \n",
            )
            check(mov.shape[dimensions.find("C")]).equals(
                number_channels
            ).or_raise(
                Exception,
                "These movies do not all have the same number of channels. \n",
            )
            # if XYZ are not the same thats ok ?
    return (number_channels, number_timepoints, depth, size_X, size_Y)


def get_channels_name(number_channels: int):
    # Ask for the name of the channels, return a list of str containing all the channels
    channels = []
    if number_channels == 1:
        ch = str(input("Name of the channel : \n "))
        channels.append(ch)
    else:  # if multichannels
        for n in range(number_channels):
            ch = str(input("Name of channel n°" + str(n + 1) + " : \n "))
            channels.append(ch)
    return channels


def reference_channel(channels: list):
    # Ask for the reference channel among the channels given (with safety check)
    if len(channels) == 1:
        ch_ref = channels[0]
    else:
        channel_correct = False
        print(
            "\nAmong the channels"
            + str(channels)
            + ", you need a reference channel to compute the registration. A good option is generally a marker that is expressed ubiquitously\n"
        )
        while not channel_correct:
            ch_ref = input("Name of the reference channel : \n ")
            if ch_ref not in channels:
                print(
                    "The reference channel is not in",
                    channels,
                    "(do not put any other character than the name itself)",
                )
                channel_correct = False
            else:
                channel_correct = True
    channels_float = channels.copy()
    channels_float.remove(ch_ref)
    return (channels_float, ch_ref)


def sort_by_channels_and_timepoints(
    list_paths: str, channels: str, number_timepoints: int
):
    # Take a list of movies and cut them into a timesequence of 3D stacks, one timesequence per channel. Works for one channel or multiple channels.
    for path in list_paths:
        for n in range(len(channels)):
            name_movie = Path(path).stem
            directory = name_movie + "_" + channels[n]
            cut_timesequence(
                path_to_movie=path,
                directory=directory,
                ind_current_channel=n,
                number_timepoints=number_timepoints,
                number_channels=len(channels),
            )


def cut_timesequence(
    path_to_movie: str,
    directory: str,
    ind_current_channel: int,
    number_timepoints: int,
    number_channels: int,
):
    # take a movie and cut it into a timesequence in the given directory. Creates the folder structure for the registration.
    movie = tifffile.imread(path_to_movie)
    position_t = np.argwhere(np.array(movie.shape) == number_timepoints)[0][0]
    movie = np.moveaxis(
        movie, source=position_t, destination=0
    )  # i did not test this
    path_to_data = os.path.dirname(path_to_movie)
    path_dir = rf"{path_to_data}\{directory}"
    check(os.path.isdir(path_dir)).is_(False).or_raise(
        Exception, "Please delete the folder " + directory + " and run again."
    )
    os.mkdir(os.path.join(path_to_data, directory))
    os.mkdir(os.path.join(path_to_data + "/" + directory, "trsf"))
    os.mkdir(os.path.join(path_to_data + "/" + directory, "output"))
    os.mkdir(os.path.join(path_to_data + "/" + directory, "proj_output"))
    os.mkdir(os.path.join(path_to_data + "/" + directory, "stackseq"))

    if number_channels > 1:
        position_c = np.argwhere(np.array(movie.shape) == number_channels)[0][
            0
        ]
        movie = np.moveaxis(
            movie, source=position_c, destination=2
        )  # we artificially change to the format TZCXY (reference in Fiji). Its just to cut into timesequence. does not modify the data
        for t in range(number_timepoints):
            stack = movie[t, :, ind_current_channel, :, :]
            tifffile.imwrite(
                path_to_data
                + "/"
                + directory
                + "/stackseq/movie_t"
                + str(format(t, "03d") + ".tif"),
                stack,
            )
    else:
        for t in range(number_timepoints):
            stack = movie[t, :, :, :]
            tifffile.imwrite(
                path_to_data
                + "/"
                + directory
                + "/stackseq/movie_t"
                + str(format(t, "03d") + ".tif"),
                stack,
            )


def get_voxel_sizes():
    # ask the user for the voxel size of his input and what voxel size does he want in output. Returns 2 tuples for this values.
    print(
        "\nTo register properly, you need to specify the voxel size of your input image. This can be found in Fiji, Image>Show Info. "
    )
    print("Voxel size of your original image (XYZ successively) :\n ")
    x = float(input("X :"))
    y = float(input("Y :"))
    z = float(input("Z :"))
    voxel_size_input = [x, y, z]
    print("Initial voxel size =", voxel_size_input)

    change_voxel_size = int(
        input(
            " \nYou can choose to have another voxel size on the registered image , for example to have an isotropic output image (voxel size [1,1,1]), Or you can also choose to keep the same voxel size. \nDo you want to change the voxel size of your movies ? (1 for yes, 0 for no) : \n "
        )
    )
    if change_voxel_size == 1:
        print("\nVoxel size of your image after transformation (XYZ): \n ")
        x = float(input("X :"))
        y = float(input("Y :"))
        z = float(input("Z :"))
        voxel_size_output = [x, y, z]
    elif change_voxel_size == 0:
        voxel_size_output = voxel_size_input
    print("\nVoxel size after transformation =", voxel_size_output)

    return (voxel_size_input, voxel_size_output)


def get_trsf_type():
    # Ask for what transformation the user want and return a string
    list_trsf_types = [
        "rigid2D",
        "rigid3D",
        "translation2D",
        "translation3D",
    ]  # needs to be completed
    print(
        "\nYou can choose to apply different transformation types depending on your data : ",
        list_trsf_types,
    )
    trsf_correct = False
    while not trsf_correct:
        trsf_type = str(
            input(
                "\nWhich one do you want to use ? (please enter the name of the transformation only, no other character) \n "
            )
        )
        if trsf_type in list_trsf_types:
            trsf_correct = True
        else:
            print(
                (
                    "You can only choose a transformation that is in this list :",
                    list_trsf_types,
                )
            )

    return trsf_type


def data_preparation():
    # Gather all the functions asking the user for the parameters. Returns the useful ones for registration (not XYZ size and channels list)
    list_paths = get_paths()
    number_channels, number_timepoints, depth, size_X, size_Y = dimensions(
        list_paths
    )
    channels = get_channels_name(number_channels)
    channels_float, ch_ref = reference_channel(channels)

    sort_by_channels_and_timepoints(
        list_paths=list_paths,
        channels=channels,
        number_timepoints=number_timepoints,
    )

    voxel_size_input, voxel_size_output = get_voxel_sizes()
    trsf_type = get_trsf_type()

    return (
        list_paths,
        number_timepoints,
        channels_float,
        ch_ref,
        voxel_size_input,
        voxel_size_output,
        trsf_type,
    )


def run_registration(
    list_paths: list,
    channels_float: list,
    ch_ref: str,
    voxel_size_input: tuple,
    voxel_size_output: tuple,
    trsf_type: str,
):
    # Does the actual registration : Take the reference channel first to compute and apply the trsf, then loop on the float channels and only apply it
    for path in list_paths:
        json_string = []
        folder = os.path.dirname(path)
        name_movie = Path(path).stem
        movie = tifffile.imread(path)
        directory = name_movie + "_" + ch_ref
        data_ref = {
            "path_to_data": rf"{folder}/{directory}/stackseq/",
            "file_name": "movie_t{t:03d}.tif",
            "trsf_folder": rf"{folder}/{directory}/trsf/",
            "output_format": rf"{folder}/{directory}/output/",
            "projection_path": rf"{folder}/{directory}/proj_output/",
            "check_TP": 0,
            "voxel_size": voxel_size_input,
            "voxel_size_out": voxel_size_output,
            "first": 0,
            "last": movie.shape[0] - 1,
            "not_to_do": [],
            "compute_trsf": 1,
            "ref_TP": int(movie.shape[0] / 2),
            "trsf_type": trsf_type,
            "padding": 1,
            "recompute": 1,
            "apply_trsf": 1,
            "out_bdv": "",
            "plot_trsf": 0,
        }
        json_string.append(json.dumps(data_ref))
        tr = registrationtools.TimeRegistration(data_ref)
        tr.run_trsf()

        # registration rest of the channels
        for c in channels_float:
            directory = name_movie + "_" + c
            data_float = {
                "path_to_data": rf"{folder}/{directory}/stackseq/",
                "file_name": "movie_t{t:03d}.tif",
                "trsf_folder": rf"{folder}/{name_movie}_{ch_ref}/trsf/",
                "output_format": rf"{folder}/{directory}/output/",
                "projection_path": rf"{folder}/{directory}/proj_output/",
                "check_TP": 0,
                "voxel_size": voxel_size_input,
                "voxel_size_out": voxel_size_output,
                "first": 0,
                "last": movie.shape[0] - 1,
                "not_to_do": [],
                "compute_trsf": 0,
                "ref_TP": int(movie.shape[0] / 2),
                "trsf_type": trsf_type,
                "padding": 1,
                "recompute": 1,
                "apply_trsf": 1,
                "out_bdv": "",
                "plot_trsf": 0,
            }
            json_string.append(json.dumps(data_float))
            tr = registrationtools.TimeRegistration(data_float)
            tr.run_trsf()
    return json_string


def save_sequences_as_stacks(
    list_paths: list, channels: list, number_timepoints: int
):
    # save the timesequence as a hyperstack
    for path in list_paths:
        path_to_data = os.path.dirname(path)
        name_movie = Path(path).stem
        movie = tifffile.imread(path)
        stack0 = tifffile.imread(
            rf"{path_to_data}/{name_movie}_{channels[0]}/output/movie_t000.tif"
        )  # we use the first image (3D) to know the dimensions
        registered_movie = np.zeros(
            (
                number_timepoints,
                stack0.shape[0],
                len(channels),
                stack0.shape[1],
                stack0.shape[2],
            ),
            dtype=np.float32,
        )  # one movie per channel, of format (t,z,y,x).Datatype uint16 or float32 is necessary to export as hyperstack
        for ind_c, c in enumerate(channels):
            directory = name_movie + "_" + c
            for t in range(movie.shape[0]):
                stack = tifffile.imread(
                    rf"{path_to_data}/{directory}/output/movie_t{format(t,'03d')}.tif"
                )
                # we take each stack in a given timepoint
                registered_movie[t, :, ind_c, :, :] = (
                    stack  # and put it in a new hyperstack
                )
        tifffile.imwrite(
            path_to_data + rf"/{name_movie}_registered.tif",
            registered_movie.astype(np.float32),
            imagej=True,
        )  # write a hyperstack in the main folder
        print(
            "saved registered", name_movie, "of size", registered_movie.shape
        )


def save_jsonfile(list_paths, json_string):
    path_to_data = os.dirname(list_paths[0])
    keep_same_dir = int(
        input(
            str(
                "Do you want to save your json files in the same master directory, in "
                + path_to_data
                + "/jsonfiles ? (1 for yes, 0 for no)"
            )
        )
    )
    if keep_same_dir == 1:
        path_to_json = rf"{path_to_data}\jsonfiles"
        os.mkdir(os.path.join(path_to_data, "jsonfiles"))
    else:
        path_to_json = input(
            "In which folder do you want to write your jsonfile ?"
        ).replace(
            "\\", "/"
        )  # if single backslashes, fine. If double backslashes (when copy/paste the Windows path), compatibility problems, thats why we replace by single slashes.')

    print("saving", len(json_string), "json files :")
    for ind_json, json_file in enumerate(json_string):
        with open(
            path_to_json + "/param" + str(ind_json) + ".json", "w"
        ) as outfile:  # maybe not the best name
            outfile.write(json_file)
            print(path_to_json)
    print("Done saving")
