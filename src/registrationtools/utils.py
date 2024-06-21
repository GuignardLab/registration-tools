import tifffile
import os
import registrationtools
import json
import numpy as np
from ensure import check  ##need to pip install !
from pathlib import Path


def get_paths():
    """
    Ask the user for the path to the data, check if it is an image sequence or a movie, and return the list of paths to the movies and the type of data

    Returns
    -------
    paths_movies : list
        if input is a timesequence : List of the paths of the timeframes
        if input is a movie : List of the paths of the different movies to register independently
    data_type : int
        1 for image sequence, 2 for movie
    """
    path_correct = False
    while not path_correct:
        data_type = input(
            "Is your input data an image sequence or a tiff movie contained in one file ? (1 for image sequence, 2 for movie) : \n "
        )
        if data_type == "1":
            print(
                "The sequence has to be a serie of z stacks, one file per timeframe, one channel.\n The sequence of images should be in tiff format, with the same dimensions for each image.  \n For now, the timeframes have to be named 'movie_t000.tif', 'movie_t001.tif' and so on.\n"
            )
            path_folder = input("Path to the folder : \n ")
            paths_movies = list(Path(path_folder).glob("*.tif"))
            path_correct = True
            if len(paths_movies) > 0:
                path_correct = int(
                    input(
                        "You have",
                        len(paths_movies),
                        "timeframes; correct ? (1 for yes, 0 for no or to change folder) \n ",
                    )
                )
            else:
                print("There is no tiff file in the folder. \n")
        elif data_type == "2":
            path_folder = input(
                "Path to the folder of the movie(s), in tif format only (all the movies in that folder are going to be registered) : \n "
            )
            paths_movies = list(Path(path_folder).glob("*.tif"))
            if len(paths_movies) > 0:
                print(
                    "You have",
                    len(paths_movies),
                    "movie(s), which is (are) : ",
                )
                for path_movie in paths_movies:
                    print(
                        "", Path(path_movie).stem
                    )  # the empty str at the begining is to align the answers one space after the questions, for readability
                path_correct = int(
                    input(
                        "Correct ? (1 for yes, 0 for no or to change folder) \n "
                    )
                )
            else:
                print("There is no tiff file in the folder. \n")

    return (paths_movies, data_type)


def get_dimensions(list_paths: list, data_type: int):
    """
    Ask the user for the dimensions of the data, and return the dimensions in the format TZCYX
    Raise error if the other movies do not have the same C ot T dimensions.
    Parameters
    ----------
    list_paths : list
        if input is a timesequence : List of the paths of the timeframes
        if input is a movie : List of the paths of the different movies to register independently
    data_type : int
        1 for image sequence, 2 for movie
    """

    dim_correct = False
    if data_type == "1":
        number_timepoints = len(list_paths)
        number_channels = 1
        movie_0 = tifffile.imread(list_paths[0])
        print("The dimensions of the first movie are ", movie_0.shape, ". \n")
        for i in range(len(list_paths)):
            movie = tifffile.imread(list_paths[i])
            check(movie.shape).equals(movie_0.shape).or_raise(
                Exception,
                "These movies do not all have the same shape in (XYZ). \n",
            )

        while not dim_correct:
            print("\nThe dimensions of one stack is ", movie_0.shape, ". \n")
            dimensions = input(
                "What is the order of the dimensions (for example ZYX or XYZ) ? \n "
            )
            sorted_axes = sorted(dimensions)
            if "".join(sorted_axes) == "XYZ":
                number_channels = 1
                dim_correct = True
            elif "".join(sorted_axes) == "CXYZ":
                print(
                    "If you chose to give a timesequence, we can only register one channel at a time. \n"
                )
            elif len(sorted_axes) != len(movie.shape):
                print("Error : Number of dimensions is incorrect. \n")
                dim_correct = False
            else:
                print(
                    " \n The letters you choose has to be X, Y and Z and no other letters are allowed. \nEvery letter can be included only once. \n"
                )

            if dim_correct:
                size_X = movie.shape[dimensions.find("X")]
                size_Y = movie.shape[dimensions.find("Y")]
                if "Z" in dimensions:
                    depth = movie.shape[dimensions.find("Z")]
                else:
                    depth = 1

                print(
                    "\nSo the movie has",
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

    elif data_type == "2":
        movie = tifffile.imread(list_paths[0])
        name_movie = Path(list_paths[0]).stem
        dim_correct = False
        while not dim_correct:
            print(
                "\nThe dimensions of ", name_movie, "are ", movie.shape, ". \n"
            )
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
            elif "".join(sorted_axes) == "TXY":
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
                size_X = movie.shape[dimensions.find("X")]
                size_Y = movie.shape[dimensions.find("Y")]
                if "Z" in dimensions:
                    depth = movie.shape[dimensions.find("Z")]
                else:
                    depth = 1

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
    movie_dimensions = [
        number_timepoints,
        depth,
        number_channels,
        size_X,
        size_Y,
    ]  # TZCYX
    return movie_dimensions


# too complicated to ask filename -> for now the user has to name its files movie_t000.tif,movie_t001.tif
# def get_filename(list_paths):
#     name_correct = False
#     while not name_correct:
#         name = input(
#             "Give the name of your files (example : 'movie_t{t:03d}.tif' if your names are 'movies_t001.tif,movies_t002.tif'). Don't forget the quotes.  \n "
#         )
#         name_correct=True
#     return(name)
def get_channels_name(number_channels: int):
    """
    Ask for the name of the channels, return a list of str containing all the channels

    Parameters
    ----------
    number_channels : int
        Number of channels in the data, to know how many names to ask for

    Returns
    -------
    channels : list
        List of the names of the channels
    """

    channels = []
    if number_channels == 1:
        ch = "1"
        channels.append(ch)
    else:  # if multichannels
        for n in range(number_channels):
            ch = str(input("Name of channel n°" + str(n + 1) + " : \n "))
            channels.append(ch)
    return channels


def reference_channel(channels: list):
    """
    Ask for the reference channel among the channels given, return the list of the float channels and the reference channel

    Parameters
    ----------
    channels : list
        List of the names of the channels, given by the function get_channels_name

    Returns
    -------
    channels_float : list
        List of the names of the floating channels
    ch_ref : str
        Name of the reference channel, the one that will be used to compute the registration
    """

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
    list_paths: str, channels: str, dimensions: list
):
    """
    Take a list of movies and cut them into a timesequence of 3D stacks, in one folder per channel.
    Only used if the input is a movie.
    Calls the function cut_timesequence for each movie and each channel

    Parameters
    ----------
    list_paths : list
        List of the paths of the different movies to register independently
    channels : list
        List of the names of the channels
    dimensions : list
        List of the dimensions of the data, in the format TZCYX

    """

    for path in list_paths:
        for n in range(len(channels)):
            name_movie = Path(path).stem
            directory = name_movie + "_" + channels[n]
            cut_timesequence(
                path_to_movie=path,
                directory=directory,
                ind_current_channel=n,
                dimensions=dimensions,
            )


def cut_timesequence(
    path_to_movie: str,
    directory: str,
    ind_current_channel: int,
    dimensions: list,
):
    """
    Take a movie and cut it into a timesequence in the given directory. Creates the folder structure for the registration.

    Parameters
    ----------
    path_to_movie : str
        Path to the movie to cut
    directory : str
        Name of the directory (related the given channel) to create
    ind_current_channel : int
        Index of the current channel in the list of channels
    dimensions : list
        List of the dimensions of the data, in the format TZCYX

    """

    number_timepoints, depth, number_channels, size_X, size_Y = dimensions
    movie = tifffile.imread(path_to_movie)
    position_t = np.argwhere(np.array(movie.shape) == number_timepoints)[0][0]
    movie = np.moveaxis(
        movie, source=position_t, destination=0
    )  # i did not test this
    path_to_data = os.path.dirname(path_to_movie)
    path_dir = Path(path_to_data) / directory
    check(os.path.isdir(path_dir)).is_(False).or_raise(
        Exception, "Please delete the folder " + directory + " and run again."
    )
    os.mkdir(os.path.join(path_to_data, directory))
    os.mkdir(os.path.join(path_to_data, directory, "trsf"))
    os.mkdir(os.path.join(path_to_data, directory, "output"))
    os.mkdir(os.path.join(path_to_data, directory, "proj_output"))
    os.mkdir(os.path.join(path_to_data, directory, "stackseq"))

    if depth <= 1:  # 2D situation
        if number_channels > 1:
            position_c = np.argwhere(np.array(movie.shape) == number_channels)[
                0
            ][0]
            movie = np.moveaxis(
                movie, source=position_c, destination=1
            )  # we artificially change to the format TZCXY (reference in Fiji). Its just to cut into timesequence. does not modify the data
            for t in range(number_timepoints):
                stack = movie[t, ind_current_channel, :, :]
                path_output = os.path.join(
                    Path(path_to_data),
                    directory,
                    "stackseq",
                    "movie_t" + str(format(t, "03d")) + ".tif",
                )
                tifffile.imwrite(path_output, stack)

        else:
            for t in range(number_timepoints):
                stack = movie[t, :, :]
                path_output = os.path.join(
                    Path(path_to_data),
                    directory,
                    "stackseq",
                    "movie_t" + str(format(t, "03d")) + ".tif",
                )
                tifffile.imwrite(path_output, stack)

    elif depth > 1:  # 3D situation
        if number_channels > 1:
            position_c = np.argwhere(np.array(movie.shape) == number_channels)[
                0
            ][0]
            movie = np.moveaxis(
                movie, source=position_c, destination=2
            )  # we artificially change to the format TZCXY (reference in Fiji). Its just to cut into timesequence. does not modify the data
            for t in range(number_timepoints):
                stack = movie[t, :, ind_current_channel, :, :]
                path_output = os.path.join(
                    Path(path_to_data),
                    directory,
                    "stackseq",
                    "movie_t" + str(format(t, "03d")) + ".tif",
                )
                tifffile.imwrite(path_output, stack)
        else:
            for t in range(number_timepoints):
                stack = movie[t, :, :, :]
                path_output = os.path.join(
                    Path(path_to_data),
                    directory,
                    "stackseq",
                    "movie_t" + str(format(t, "03d")) + ".tif",
                )
                tifffile.imwrite(path_output, stack)

    elif depth <= 1:  # 2D situation
        if number_channels > 1:
            position_c = np.argwhere(np.array(movie.shape) == number_channels)[
                0
            ][0]
            movie = np.moveaxis(
                movie, source=position_c, destination=1
            )  # we artificially change to the format TZCXY (reference in Fiji). Its just to cut into timesequence. does not modify the data
            for t in range(number_timepoints):
                stack = movie[t, ind_current_channel, :, :]
                path_output = os.path.join(
                    Path(path_to_data),
                    directory,
                    "stackseq",
                    "movie_t" + str(format(t, "03d")) + ".tif",
                )
                tifffile.imwrite(path_output, stack)


def get_voxel_sizes():
    """
    Ask the user for the voxel size of his input and what voxel size does he want in output. Returns 2 tuples for this values.

    Returns
    -------
    voxel_size_input : tuple
        Tuple of the voxel size of the input image, in the format (ZYX)
    voxel_size_output : tuple
        Tuple of the voxel size of the output image, in the format (ZYX)
    """

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
    """
    Ask for what transformation the user want and return a string

    Returns
    -------
    trsf_type : str
        Name of the transformation type
    """

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
    """
    Function called in the notebook
    Gather all the functions asking the user for the parameters. Returns the useful ones for registration (not XYZ size and channels list)

    Returns
    -------
    data_type : int
        1 for image sequence, 2 for movie
    list_paths : list
        if input is a timesequence : List of the paths of the timeframes
        if input is a movie : List of the paths of the different movies to register independently
    dimensions : list
        List of the dimensions of the data, in the format TZCYX
    filename : str
    channels_float : list
        List of the names of the floating channels
    ch_ref : str
        Name of the reference channel, the one that will be used to compute the registration
    voxel_size_input : tuple
        Tuple of the voxel size of the input image, in the format (ZYX)
    voxel_size_output : tuple
        Tuple of the voxel size of the output image, in the format (ZYX)
    trsf_type : str
        Name of the transformation type (rigid2D, rigid3D, translation2D, translation3D)
    """

    list_paths, data_type = get_paths()
    dimensions = get_dimensions(list_paths, data_type)
    filename = "movie_t{t:03d}.tif"  # for now the user has no choice regarding the filename
    channels = get_channels_name(dimensions[2])
    channels_float, ch_ref = reference_channel(channels)
    if data_type == "2":  # if the datatype=1, it is already a timesequence
        sort_by_channels_and_timepoints(
            list_paths=list_paths, channels=channels, dimensions=dimensions
        )

    voxel_size_input, voxel_size_output = get_voxel_sizes()
    trsf_type = get_trsf_type()

    # print('Parameters : \ndata_type=',data_type,'\nfilename=',str(filename),'\nlist_paths=',list_paths,'\nnumber_timepoints=',dimensions[0],'\nchannels_float=',channels_float,'\nch_ref=',ch_ref,'\nvoxel_size_input=',voxel_size_input,'\nvoxel_size_output=',voxel_size_output,'\ntrsf_type=',trsf_type)
    return (
        data_type,
        filename,
        list_paths,
        dimensions[0],
        channels_float,
        ch_ref,
        voxel_size_input,
        voxel_size_output,
        trsf_type,
    )


def run_registration(
    data_type: int,
    filename: str,
    list_paths: list,
    channels_float: list,
    ch_ref: str,
    voxel_size_input: tuple,
    voxel_size_output: tuple,
    trsf_type: str,
    number_timepoints: int,
    first:int=0,
    last:int=None,
    ref_tp:int=None,
):
    """
    2nd function called in the notebook
    Organises the paths and channels to register, and run the function run_from_json for each of them. Returns the json strings to save them in a json file if asked.

    Parameters
    ----------
    data_type : int
        1 for image sequence, 2 for movie
    list_paths : list
        if input is a timesequence : List of the paths of the timeframes
        if input is a movie : List of the paths of the different movies to register independently
    channels_float : list
        List of the names of the floating channels
    ch_ref : str
        Name of the reference channel, the one that will be used to compute the registration
    voxel_size_input : tuple
        Tuple of the voxel size of the input image, in the format (ZYX)
    voxel_size_output : tuple
        Tuple of the voxel size of the output image, in the format (ZYX)
    trsf_type : str
        Name of the transformation type (rigid2D, rigid3D, translation2D, translation3D)
    number_timepoints : int
        Number of timeframes

    """

    json_string = []
    if last is None :
        last = number_timepoints - 1
    if ref_tp is None:
        ref_tp = int(number_timepoints / 2)

    if data_type == "1":  # simpler because its only 1 movie with 1 channel

        folder = os.path.dirname(list_paths[0])
        path_to_data = Path(
            folder
        )  # the timesequence is direclty the folder given by the user.
        path_trsf = Path(folder) / "trsf"
        path_output = Path(folder) / "output"
        path_proj = Path(folder) / "proj_output"
        json_str = run_from_json(
            path_to_data=path_to_data,
            path_trsf=path_trsf,
            filename=filename,
            path_output=path_output,
            path_proj=path_proj,
            voxel_size_input=voxel_size_input,
            voxel_size_output=voxel_size_output,
            compute_trsf=1,
            trsf_type=trsf_type,
            number_timepoints=number_timepoints,
            first=first,
            last=last,
            ref_tp=ref_tp
        )
        json_string.append(json_str)

    elif data_type == "2":

        for (
            path_movie
        ) in list_paths:  # loop on each movie from the original folder
            folder = os.path.dirname(path_movie)
            name_movie = Path(path_movie).stem
            directory = name_movie + "_" + ch_ref

            path_to_data = os.path.join(Path(folder), directory, "stackseq")
            path_trsf = Path(folder) / directory / "trsf"
            path_output = Path(folder) / directory / "output"
            path_proj = Path(folder) / directory / "proj_output"
            json_str = run_from_json(
                path_to_data=path_to_data,
                filename=filename,
                path_trsf=path_trsf,
                path_output=path_output,
                path_proj=path_proj,
                voxel_size_input=voxel_size_input,
                voxel_size_output=voxel_size_output,
                compute_trsf=1,
                trsf_type=trsf_type,
                number_timepoints=number_timepoints,
                first=first,
                last=last,
                ref_tp=ref_tp
            )
            json_string.append(json_str)
            # registration rest of the channels
            for c in channels_float:
                directory = name_movie + "_" + c
                path_to_data = os.path.join(
                    Path(folder), directory, "stackseq"
                )
                path_output = Path(folder) / directory / "output"
                path_proj = Path(folder) / directory / "proj_output"
                # we dont update path_trsf : the trsf directory is always the one of the reference channel
                json_str = run_from_json(
                    path_to_data=path_to_data,
                    filename=filename,
                    path_trsf=path_trsf,
                    path_output=path_output,
                    path_proj=path_proj,
                    voxel_size_input=voxel_size_input,
                    voxel_size_output=voxel_size_output,
                    compute_trsf=0,  # we do not compute the trsf again, we just apply it
                    trsf_type=trsf_type,
                    number_timepoints=number_timepoints,
                    first=first,
                    last=last,
                    ref_tp=ref_tp
                )
                json_string.append(json_str)
    return json_string


def run_from_json(
    path_to_data: Path,
    filename: str,
    path_trsf: Path,
    path_output: Path,
    path_proj: Path,
    voxel_size_input: tuple,
    voxel_size_output: tuple,
    compute_trsf: int,
    trsf_type: str,
    number_timepoints: int,
    first:int,
    last:int,
    ref_tp:int,
):
    """
    Does the actual registration of the data, and returns the json string to save it in a json file.

    Parameters
    ----------
    path_to_data : Path
        Path to the data to register
    filename : str
    path_trsf : Path
        Path to the folder where the transformation files will be saved
    path_output : Path
        Path to the folder where the registered images will be saved
    path_proj : Path
        Path to the folder where the projections will be saved
    voxel_size_input : tuple
        Tuple of the voxel size of the input image, in the format (ZYX)
    voxel_size_output : tuple
        Tuple of the voxel size of the output image, in the format (ZYX)
    compute_trsf : int
        1 if we want to compute the transformation, 0 if we just want to apply it
    trsf_type : str
        Name of the transformation type (rigid2D, rigid3D, translation2D, translation3D)
    number_timepoints : int
        Number of timeframes

    Returns
    -------
    json_string : list
        List of the json strings to save them in a json file if asked.
    """
    data_float = {
        "path_to_data": str(
            path_to_data
        ),  # paths have to be a string to be integrated in the json
        "file_name": filename,
        "trsf_folder": str(path_trsf),
        "output_format": str(path_output),
        "projection_path": str(path_proj),
        "check_TP": 0,
        "voxel_size": voxel_size_input,
        "voxel_size_out": voxel_size_output,
        "first": first,
        "last": last,
        "not_to_do": [],
        "compute_trsf": compute_trsf,
        "ref_TP": ref_tp,
        "trsf_type": trsf_type,
        "padding": 1,
        "recompute": 1,
        "apply_trsf": 1,
        "out_bdv": "",
        "plot_trsf": 0,
    }
    json_string = json.dumps(data_float, indent=2)
    tr = registrationtools.TimeRegistration(data_float)
    tr.run_trsf()
    return json_string


def save_sequences_as_stacks(
    list_paths: list, data_type: int, channels: list, number_timepoints: int
):
    """
    save the timesequence as a hyperstack in the main folder

    Parameters
    ----------
    list_paths : list
        List of the paths of the timeframes
    data_type : int
        1 for image sequence, 2 for movie
    channels : list
        List of the names of the channels
    number_timepoints : int
        Number of timeframes
    """
    if data_type == "1":
        path_main_directory = os.path.dirname(list_paths[0])
        stack0 = tifffile.imread(
            os.path.join(Path(path_main_directory), "output", "movie_t000.tif")
        )  # we use the first image (3D) to know the dimensions
        registered_movie = np.zeros(
            (
                number_timepoints,
                stack0.shape[0],
                stack0.shape[1],
                stack0.shape[2],
            )
        )
        for t in range(number_timepoints):
            stack = tifffile.imread(
                os.path.join(
                    Path(path_main_directory),
                    "output",
                    rf"movie_t{format(t,'03d')}.tif",
                )
            )
            registered_movie[t, :, :, :] = stack
        name_movie = "movie"
    elif data_type == "2":
        for path in list_paths:
            path_main_directory = os.path.dirname(path)
            name_movie = Path(path).stem

            stack0 = tifffile.imread(
                os.path.join(
                    Path(path_main_directory),
                    name_movie + "_" + channels[0],
                    "output",
                    "movie_t000.tif",
                )
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
                for t in range(number_timepoints):
                    stack = tifffile.imread(
                        os.path.join(
                            Path(path_main_directory),
                            name_movie + "_" + c,
                            "output",
                            rf"movie_t{format(t,'03d')}.tif",
                        )
                    )
                    # we take each stack in a given timepoint
                    registered_movie[t, :, ind_c, :, :] = (
                        stack  # and put it in a new hyperstack
                    )
    tifffile.imwrite(
        os.path.join(
            Path(path_main_directory), name_movie + "_registered.tif"
        ),
        registered_movie.astype(np.float32),
        imagej=True,
    )  # write a hyperstack in the main folder
    print("saved registered", name_movie, "of size", registered_movie.shape)


def save_jsonfile(list_paths, json_string):
    """
    Save the json strings in a json file in the folder of the data or in a folder chosen by the user

    Parameters
    ----------
    list_paths : list
        List of the paths of the timeframes
    json_string : list
        List of the json strings to save them in a json file if asked.
    """
    path_main_directory = os.path.dirname(list_paths[0])
    keep_same_dir = int(
        input(
            str(
                "Do you want to save your json files in the same master directory, in "
                + path_main_directory
                + "\jsonfiles ? (1 for yes, 0 for no)"
            )
        )
    )
    if keep_same_dir:
        path_to_json = Path(path_main_directory) / "jsonfiles"
        os.mkdir(os.path.join(path_main_directory, "jsonfiles"))
    else:
        path_to_json = input(
            "In which folder do you want to write your jsonfile ?"
        )
        # .replace(
        #    "\\", "/"
        # )  # if single backslashes, fine. If double backslashes (when copy/paste the Windows path), compatibility problems, thats why we replace by single slashes.')

    print("saving", len(json_string), "json files :")
    for ind_json, jsonfile in enumerate(json_string):
        with open(
            os.path.join(
                Path(path_to_json), "param" + str(ind_json) + ".json"
            ),
            "w",
        ) as outfile:  # maybe not the best name
            outfile.write(jsonfile)
