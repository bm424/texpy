# -*- coding: utf-8 -*-
# Copyright 2018-2021 the orix developers
#
# This file is part of orix.
#
# orix is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# orix is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with orix.  If not, see <http://www.gnu.org/licenses/>.

import gc
import os
from tempfile import TemporaryDirectory

from diffpy.structure import Atom, Lattice, Structure
from h5py import File
import numpy as np
import pytest

from orix.crystal_map import CrystalMap, create_coordinate_arrays
from orix.crystal_map.phase_list import PhaseList
from orix.quaternion.rotation import Rotation


@pytest.fixture
def rotations():
    return Rotation([(2, 4, 6, 8), (-1, -3, -5, -7)])


# TODO: Exchange for a multiphase header (change `phase_id` accordingly)
ANGFILE_TSL_HEADER = (
    "# TEM_PIXperUM          1.000000\n"
    "# x-star                0.413900\n"
    "# y-star                0.729100\n"
    "# z-star                0.514900\n"
    "# WorkingDistance       27.100000\n"
    "#\n"
    "# Phase 2\n"
    "# MaterialName      Aluminum\n"
    "# Formula       Al\n"
    "# Info      \n"
    "# Symmetry              43\n"
    "# LatticeConstants      4.040 4.040 4.040  90.000  90.000  90.000\n"
    "# NumberFamilies        69\n"
    "# hklFamilies        1 -1 -1 1 8.469246 1\n"
    "# ElasticConstants  -1.000000 -1.000000 -1.000000 -1.000000 -1.000000 -1.000000\n"
    "# Categories0 0 0 0 0 \n"
    "# Phase 3\n"
    "# MaterialName  	Iron Titanium Oxide\n"
    "# Formula     	FeTiO3\n"
    "# Info 		\n"
    "# Symmetry              32\n"
    "# LatticeConstants      5.123 5.123 13.760  90.000  90.000 120.000\n"
    "# NumberFamilies        60\n"
    "# hklFamilies   	 3  0  0 1 100.000000 1\n"
    "# ElasticConstants  -1.000000 -1.000000 -1.000000 -1.000000 -1.000000 -1.000000\n"
    "# Categories0 0 0 0 0\n"
    "#\n"
    "# GRID: SqrGrid\n"
    "# XSTEP: 0.100000\n"
    "# YSTEP: 0.100000\n"
    "# NCOLS_ODD: 42\n"
    "# NCOLS_EVEN: 42\n"
    "# NROWS: 13\n"
    "#\n"
    "# OPERATOR:     sem\n"
    "#\n"
    "# SAMPLEID:     \n"
    "#\n"
    "# SCANID:   \n"
    "#\n"
)

ANGFILE_ASTAR_HEADER = (
    "# File created from ACOM RES results\n"
    "# ni-dislocations.res\n"
    "#     \n"
    "#     \n"
    "# MaterialName      Nickel\n"
    "# Formula\n"
    "# Symmetry          43\n"
    "# LatticeConstants  3.520  3.520  3.520  90.000  90.000  90.000\n"
    "# NumberFamilies    4\n"
    "# hklFamilies       1  1  1 1 0.000000\n"
    "# hklFamilies       2  0  0 1 0.000000\n"
    "# hklFamilies       2  2  0 1 0.000000\n"
    "# hklFamilies       3  1  1 1 0.000000\n"
    "#\n"
    "# GRID: SqrGrid#\n"
)

ANGFILE_EMSOFT_HEADER = (
    "# TEM_PIXperUM          1.000000\n"
    "# x-star                 0.446667\n"
    "# y-star                 0.586875\n"
    "# z-star                 0.713450\n"
    "# WorkingDistance        0.000000\n"
    "#\n"
    "# Phase 1\n"
    "# MaterialName    austenite\n"
    "# Formula       austenite\n"
    "# Info          patterns indexed using EMsoft::EMEBSDDI\n"
    "# Symmetry              43\n"
    "# LatticeConstants      3.595 3.595 3.595   90.000 90.000 90.000\n"
    "# NumberFamilies        0\n"
    "# Phase 2\n"
    "# MaterialName    ferrite/ferrite\n"
    "# Formula       ferrite/ferrite\n"
    "# Info          patterns indexed using EMsoft::EMEBSDDI\n"
    "# Symmetry              43\n"
    "# LatticeConstants      2.867 2.867 2.867   90.000 90.000 90.000\n"
    "# NumberFamilies        0\n"
    "# GRID: SqrGrid\n"
    "# XSTEP:  1.500000\n"
    "# YSTEP:  1.500000\n"
    "# NCOLS_ODD:   13\n"
    "# NCOLS_EVEN:   13\n"
    "# NROWS:   42\n"
    "#\n"
    "# OPERATOR:   Håkon Wiik Ånes\n"
    "#\n"
    "# SAMPLEID:\n"
    "#\n"
    "# SCANID:\n"
    "#\n"
)


@pytest.fixture()
def temp_ang_file():
    with TemporaryDirectory() as tempdir:
        f = open(os.path.join(tempdir, "temp_ang_file.ang"), mode="w+")
        yield f
        gc.collect()  # Garbage collection so that file can be used by multiple tests


@pytest.fixture(params=["h5"])
def temp_file_path(request):
    """Temporary file in a temporary directory for use when tests need
    to write, and sometimes read again, data to, and from, a file.
    """
    ext = request.param
    with TemporaryDirectory() as tmp:
        file_path = os.path.join(tmp, "data_temp." + ext)
        yield file_path
        gc.collect()


@pytest.fixture(
    params=[
        # Tuple with default values for five parameters: map_shape, step_sizes,
        # phase_id, n_unknown_columns, and rotations (see docstring below)
        (
            (7, 13),  # map_shape
            (0.1, 0.1),  # step_sizes
            np.random.choice([2, 3], 7 * 13),  # phase_id
            5,  # Number of unknown columns (one between ci and fit, the rest after fit)
            np.array(
                [[4.48549, 0.95242, 0.79150], [1.34390, 0.27611, 0.82589],]
            ),  # rotations as rows of Euler angle triplets
        )
    ]
)
def angfile_tsl(tmpdir, request):
    """Create a dummy EDAX TSL .ang file from input.

    10% of map points are set to non-indexed (confidence index equal to
    -1).

    Parameters expected in `request`
    --------------------------------
    map_shape : tuple of ints
        Map shape to create.
    step_sizes : tuple of floats
        Step sizes in x and y coordinates in microns.
    phase_id : np.ndarray
        Array of map size with phase IDs in header.
    n_unknown_columns : int
        Number of columns with unknown values.
    rotations : np.ndarray
        A sample, smaller than the map size, of example rotations as
        rows of Euler angle triplets.
    """
    f = tmpdir.join("angfile_tsl.ang")

    # Unpack parameters
    (ny, nx), (dy, dx), phase_id, n_unknown_columns, example_rotations = request.param

    # File columns
    d, map_size = create_coordinate_arrays((ny, nx), (dy, dx))
    x = d["x"]
    y = d["y"]
    ci = np.random.random(map_size)  # [0, 1]
    iq = np.random.uniform(low=1e3, high=1e6, size=map_size)
    un = np.zeros(map_size, dtype=int)
    fit = np.random.uniform(low=0, high=3, size=map_size)
    # Rotations
    rot_idx = np.random.choice(np.arange(len(example_rotations)), map_size)
    rot = example_rotations[rot_idx]

    # Insert 10% non-indexed points
    non_indexed_points = np.random.choice(
        np.arange(map_size), replace=False, size=int(map_size * 0.1)
    )
    rot[non_indexed_points] = 4 * np.pi
    ci[non_indexed_points] = -1
    fit[non_indexed_points] = 180.0

    np.savetxt(
        fname=f,
        X=np.column_stack(
            (rot[:, 0], rot[:, 1], rot[:, 2], x, y, iq, ci, phase_id, un, fit)
            + (un,) * (n_unknown_columns - 1)
        ),
        fmt=(
            "%9.5f%10.5f%10.5f%13.5f%13.5f%9.1f%7.3f%3i%7i%8.3f"
            + "%10.5f" * (n_unknown_columns - 1)
            + " "
        ),
        header=ANGFILE_TSL_HEADER,
        comments="",
    )

    return f


@pytest.fixture(
    params=[
        # Tuple with default values for four parameters: map_shape, step_sizes,
        # phase_id, and rotations (see docstring below)
        (
            (7, 5),  # map_shape
            (2.86, 2.86),  # step_sizes
            np.ones(7 * 5, dtype=int),  # phase_id
            np.array(
                [[6.148271, 0.792205, 1.324879], [6.155951, 0.793078, 1.325229],]
            ),  # rotations as rows of Euler angle triplets
        )
    ]
)
def angfile_astar(tmpdir, request):
    """Create a dummy NanoMegas ASTAR Index .ang file from input.

    Parameters expected in `request`
    --------------------------------
    map_shape : tuple of ints
        Map shape to create.
    step_sizes : tuple of floats
        Step sizes in x and y coordinates in nanometres.
    phase_id : np.ndarray
        Array of map size with phase IDs in header.
    rotations : np.ndarray
        A sample, smaller than the map size, of example rotations as
        rows of Euler angle triplets.
    """
    f = tmpdir.join("angfile_astar.ang")

    # Unpack parameters
    (ny, nx), (dy, dx), phase_id, example_rotations = request.param

    # File columns
    d, map_size = create_coordinate_arrays((ny, nx), (dy, dx))
    x = d["x"]
    y = d["y"]
    ind = np.random.uniform(low=0, high=100, size=map_size)
    rel = np.round(np.random.random(map_size), decimals=2)  # [0, 1]
    relx100 = (rel * 100).astype(int)

    # Rotations
    n_rotations = len(example_rotations)
    if n_rotations == map_size:
        rot = example_rotations
    else:
        # Sample as many rotations from `rotations` as `map_size`
        rot_idx = np.random.choice(np.arange(len(example_rotations)), map_size)
        rot = example_rotations[rot_idx]

    np.savetxt(
        fname=f,
        X=np.column_stack(
            (rot[:, 0], rot[:, 1], rot[:, 2], x, y, ind, rel, phase_id, relx100)
        ),
        fmt="%8.6f%9.6f%9.6f%10.3f%10.3f%7.1f%7.3f%3i%8i",
        header=ANGFILE_ASTAR_HEADER,
        comments="",
    )

    return f


@pytest.fixture(
    params=[
        # Tuple with default values for four parameters: map_shape, step_sizes,
        # phase_id, and rotations (see docstring below)
        (
            (9, 7),  # map_shape
            (1.5, 1.5),  # step_sizes
            np.random.choice([1, 2], 9 * 7),  # phase_id
            np.array(
                [[6.148271, 0.792205, 1.324879], [6.155951, 0.793078, 1.325229],]
            ),  # rotations as rows of Euler angle triplets
        )
    ]
)
def angfile_emsoft(tmpdir, request):
    """Create a dummy EMsoft .ang file from input.

    Parameters expected in `request`
    --------------------------------
    map_shape : tuple of ints
        Map shape to create.
    step_sizes : tuple of floats
        Step sizes in x and y coordinates in nanometres.
    phase_id : np.ndarray
        Array of map size with phase IDs in header.
    rotations : np.ndarray
        A sample, smaller than the map size, of example rotations as
        rows of Euler angle triplets.
    """
    f = tmpdir.join("angfile_emsoft.ang")

    # Unpack parameters
    (ny, nx), (dy, dx), phase_id, example_rotations = request.param

    # File columns
    d, map_size = create_coordinate_arrays((ny, nx), (dy, dx))
    x = d["x"]
    y = d["y"]
    iq = np.round(np.random.uniform(low=0, high=100, size=map_size), decimals=1)
    dp = np.round(np.random.random(map_size), decimals=3)  # [0, 1]

    # Rotations
    rot_idx = np.random.choice(np.arange(len(example_rotations)), map_size)
    rot = example_rotations[rot_idx]

    np.savetxt(
        fname=f,
        X=np.column_stack((rot[:, 0], rot[:, 1], rot[:, 2], x, y, iq, dp, phase_id)),
        fmt="%.5f %.5f %.5f %.5f %.5f %.1f %.3f %i",
        header=ANGFILE_EMSOFT_HEADER,
        comments="",
    )

    return f


@pytest.fixture(
    params=[
        # Tuple with default values for parameters: map_shape, step_sizes,
        # rotations, n_top_matches, and refined (see docstring below)
        (
            (13, 3),  # map_shape
            (1.5, 1.5),  # step_sizes
            np.array(
                [[6.148271, 0.792205, 1.324879], [6.155951, 0.793078, 1.325229],]
            ),  # rotations as rows of Euler angle triplets
            50,  # n_top_matches
            True,  # refined
        )
    ]
)
def temp_emsoft_h5ebsd_file(tmpdir, request):
    """Create a dummy EMsoft h5ebsd .h5 file from input.

    Parameters expected in `request`
    --------------------------------
    map_shape : tuple of ints
        Map shape to create.
    step_sizes : tuple of floats
        Step sizes in x and y coordinates in nanometres.
    rotations : np.ndarray
        A sample, smaller than the map size, of example rotations as
        rows of Euler angle triplets.
    n_top_matches : int
        Number of top matching orientations per data point kept.
    refined : bool
        Whether refined Euler angles and dot products are read.
    """
    f = File(tmpdir.join("emsoft_h5ebsd_file.h5"), mode="w")

    # Unpack parameters
    map_shape, (dy, dx), example_rotations, n_top_matches, refined = request.param
    ny, nx = map_shape
    d, map_size = create_coordinate_arrays(map_shape, (dy, dx))
    x = d["x"]
    y = d["y"]

    # Create groups used in reader
    ebsd_group = f.create_group("Scan 1/EBSD")
    data_group = ebsd_group.create_group("Data")
    header_group = ebsd_group.create_group("Header")
    phase_group = header_group.create_group("Phase/1")  # Always single phase

    # Create `header_group` datasets used in reader
    for name, data, dtype in zip(
        ["nRows", "nColumns", "Step Y", "Step X"],
        [ny, nx, dy, dx],
        [np.int32, np.int32, np.float32, np.float32],
    ):
        header_group.create_dataset(name, data=np.array([data], dtype=dtype))

    # Create `data_group` datasets, mostly quality metrics
    data_group.create_dataset("X Position", data=x)
    # Note that "Y Position" is wrongly written to their h5ebsd file by EMsoft
    data_group.create_dataset(
        "Y Position",
        data=np.tile(np.arange(nx) * dx, ny),  # Wrong
        # data=y,  # Correct
    )
    for name, shape, dtype in [
        ("AvDotProductMap", map_shape, np.int32),
        ("CI", map_size, np.float32),
        ("CIMap", map_shape, np.int32),
        ("IQ", map_size, np.float32),
        ("IQMap", map_shape, np.int32),
        ("ISM", map_size, np.float32),
        ("ISMap", map_shape, np.int32),
        ("KAM", map_shape, np.float32),
        ("OSM", map_shape, np.float32),
        ("Phase", map_size, np.uint8),
    ]:
        data_group.create_dataset(name, data=np.zeros(shape, dtype=dtype))

    # `data_group` with rotations
    # Sample as many rotations from `rotations` as `map_size`
    rot_idx = np.random.choice(np.arange(len(example_rotations)), map_size)
    rot = example_rotations[rot_idx]
    n_sampled_oris = 333227  # Cubic space group with Ncubochoric = 100
    data_group.create_dataset("FZcnt", data=np.array([n_sampled_oris], dtype=np.int32))
    data_group.create_dataset(
        "TopMatchIndices",
        data=np.vstack(
            (np.random.choice(np.arange(n_sampled_oris), n_top_matches),) * map_size
        ),
        dtype=np.int32,
    )
    data_group.create_dataset(
        "TopDotProductList",
        data=np.vstack((np.random.random(size=n_top_matches),) * map_size),
        dtype=np.float32,
    )
    # In degrees
    data_group.create_dataset(
        "DictionaryEulerAngles",
        data=np.column_stack((np.linspace(150, 160, n_sampled_oris),) * 3),
        dtype=np.float32,
    )

    if refined:
        data_group.create_dataset("RefinedEulerAngles", data=rot.astype(np.float32))
        data_group.create_dataset(
            "RefinedDotProducts", data=np.zeros(map_size, dtype=np.float32)
        )

    # Number of top matches kept
    f.create_dataset(
        "NMLparameters/EBSDIndexingNameListType/nnk",
        data=np.array([n_top_matches], dtype=np.int32),
    )

    # `phase_group`
    for name, data in [
        ("Point Group", "Monoclinic b (C2h) [2/m]"),
        ("MaterialName", "fe4al13/fe4al13"),
        ("Lattice Constant a", "15.009001"),
        ("Lattice Constant b", "8.066"),
        ("Lattice Constant c", "12.469"),
        ("Lattice Constant alpha", "90.0"),
        ("Lattice Constant beta", "107.72"),
        ("Lattice Constant gamma", "90.0"),
    ]:
        phase_group.create_dataset(name, data=np.array([data], dtype=np.dtype("S")))

    yield f
    gc.collect()


@pytest.fixture(
    params=[
        (
            ["a", "b", "c"],
            [229, 207, 143],
            ["m-3m", "432", "3"],
            ["r", "g", "b"],
            [Lattice()] * 3,
            [[Atom()]] * 3,
        )
    ]
)
def phase_list(request):
    names, space_groups, point_group_names, colors, lattices, atoms = request.param
    # Apparently diffpy.structure don't allow iteration over a list of lattices
    structures = [Structure(lattice=lattices[i], atoms=a) for i, a in enumerate(atoms)]
    return PhaseList(
        names=names,
        space_groups=space_groups,
        point_groups=point_group_names,
        colors=colors,
        structures=structures,
    )


@pytest.fixture(
    params=[
        (
            # Tuple with default values for parameters: map_shape, step_sizes,
            # and n_rotations_per_point
            (1, 4, 3),  # map_shape
            (0, 1.5, 1.5),  # step_sizes
            1,  # rotations_per_point
            [0],  # unique phase IDs
        )
    ],
)
def crystal_map_input(request, rotations):
    # Unpack parameters
    (nz, ny, nx), (dz, dy, dx), rotations_per_point, unique_phase_ids = request.param
    d, map_size = create_coordinate_arrays((nz, ny, nx), (dz, dy, dx))
    rot_idx = np.random.choice(
        np.arange(rotations.size), map_size * rotations_per_point
    )
    data_shape = (map_size,)
    if rotations_per_point > 1:
        data_shape += (rotations_per_point,)
    d["rotations"] = rotations[rot_idx].reshape(*data_shape)
    d["phase_id"] = np.random.choice(unique_phase_ids, map_size)
    return d


@pytest.fixture
def crystal_map(crystal_map_input):
    return CrystalMap(**crystal_map_input)
