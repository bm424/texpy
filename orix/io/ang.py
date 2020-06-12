# -*- coding: utf-8 -*-
# Copyright 2018-2020 The pyXem developers
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

import re
from typing import List, Tuple
import warnings

from diffpy.structure import Lattice, Structure
import numpy as np

from orix.quaternion.rotation import Rotation
from orix.crystal_map import CrystalMap

# MTEX has this format sorted out, check out their readers when fixing issues and
# adapting to other versions of this file format in the future:
# https://github.com/mtex-toolbox/mtex/blob/develop/interfaces/loadEBSD_ang.m
# https://github.com/mtex-toolbox/mtex/blob/develop/interfaces/loadEBSD_ACOM.m


def load_ang(filename: str) -> CrystalMap:
    """Return a :class:`orix.crystal_map.CrystalMap` object from EDAX
    TSL's .ang file format. The map in the input file is assumed to be 2D.

    Many vendors produce an .ang file. Supported vendors are:
        * EDAX TSL
        * NanoMegas ASTAR Index
        * EMsoft (from program `EMdpmerge`)

    All points satisfying the following criteria are classified as not
    indexed:
        * EDAX TSL: confidence index == -1

    Parameters
    ----------
    filename : str
        Path and file name.

    Returns
    -------
    CrystalMap
    """
    # Get file header
    with open(filename) as f:
        header = _get_header(f)

    # Get phase names and crystal symmetries from header (potentially empty)
    phase_names, symmetries, lattice_constants = _get_phases_from_header(header)
    structures = []
    for name, abcABG in zip(phase_names, lattice_constants):
        structures.append(Structure(title=name, lattice=Lattice(*abcABG)))

    # Read all file data
    file_data = np.loadtxt(filename)

    # Get vendor and column names
    n_rows, n_cols = file_data.shape
    vendor, column_names = _get_vendor_columns(header, n_cols)

    # Data needed to create a CrystalMap object
    data = {
        "euler1": None,
        "euler2": None,
        "euler3": None,
        "x": None,
        "y": None,
        "phase_id": None,
        "prop": {},
    }
    for column, name in enumerate(column_names):
        if name in data.keys():
            data[name] = file_data[:, column]
        else:
            data["prop"][name] = file_data[:, column]

    # Set which data points are not indexed
    if vendor == "tsl":
        data["phase_id"][np.where(data["prop"]["ci"] == -1)] = -1
    # TODO: Add not-indexed convention for INDEX ASTAR

    # Create rotations
    rotations = Rotation.from_euler(
        np.column_stack((data["euler1"], data["euler2"], data["euler3"]))
    )

    return CrystalMap(
        rotations=rotations,
        phase_id=data["phase_id"],
        x=data["x"],
        y=data["y"],
        phase_name=phase_names,
        symmetry=symmetries,
        structure=structures,
        prop=data["prop"],
    )


def _get_header(file) -> List[str]:
    """Return the first lines starting with '#' in an .ang file.

    Parameters
    ----------
    file : _io.TextIO
        File object.

    Returns
    -------
    header : list
        List with header lines as individual elements.
    """
    header = []
    line = file.readline()
    while line.startswith("#"):
        header.append(line.rstrip())
        line = file.readline()
    return header


def _get_vendor_columns(header: list, n_cols_file: int) -> Tuple[str, List[str]]:
    """Return the .ang file column names and vendor, determined from the
    header.

    Parameters
    ----------
    header : list
        List with header lines as individual elements.
    n_cols_file : int
        Number of file columns.

    Returns
    -------
    vendor : str
        Determined vendor ("tsl", "astar", or "emsoft").
    column_names : list of str
        List of column names.
    """
    # Assume EDAX TSL by default
    vendor = "tsl"

    # Determine vendor by searching for the vendor footprint in the header
    vendor_footprint = {
        "emsoft": "EMsoft",
        "astar": "ACOM",
    }
    for name, footprint in vendor_footprint.items():
        for line in header:
            if footprint in line:
                vendor = name
                break

    # Vendor column names
    column_names = {
        "unknown": [
            "euler1",
            "euler2",
            "euler3",
            "x",
            "y",
            "unknown1",
            "unknown2",
            "phase_id",
        ],
        "tsl": [
            "euler1",
            "euler2",
            "euler3",
            "x",
            "y",
            "iq",  # Image quality from Hough transform
            "ci",  # Confidence index
            "phase_id",
            "unknown1",
            "fit",  # Pattern fit
            "unknown2",
            "unknown3",
            "unknown4",
            "unknown5",
        ],
        "emsoft": [
            "euler1",
            "euler2",
            "euler3",
            "x",
            "y",
            "iq",  # Image quality from Krieger Lassen's method
            "dp",  # Dot product
            "phase_id",
        ],
        "astar": [
            "euler1",
            "euler2",
            "euler3",
            "x",
            "y",
            "ind",  # Correlation index
            "rel",  # Reliability
            "phase_id",
            "relx100",  # Reliability x 100
        ],
    }

    n_cols_expected = len(column_names[vendor])
    if n_cols_file != n_cols_expected:
        warnings.warn(
            f"Number of columns, {n_cols_file}, in the file is not equal to "
            f"the expected number of columns, {n_cols_expected}, for the \n"
            f"assumed vendor '{vendor}'. Will therefore assume the following "
            "columns: euler1, euler2, euler3, x, y, unknown1, unknown2, "
            "phase_id, unknown3, unknown4, etc."
        )
        vendor = "unknown"
        n_cols_unknown = len(column_names["unknown"])
        if n_cols_file > n_cols_unknown:
            # Add potential extra columns to properties
            for i in range(n_cols_file - n_cols_unknown):
                column_names["unknown"].append("unknown" + str(i + 3))

    return vendor, column_names[vendor]


def _get_phases_from_header(
    header: list,
) -> Tuple[List[str], List[str], List[List[float]]]:
    """Return phase names and symmetries detected in an .ang file
    header.

    Parameters
    ----------
    header : list
        List with header lines as individual elements.

    Returns
    -------
    phase_names : list of str
        List of names of detected phases.
    phase_symmetries : list of str
        List of symmetries of detected phase.
    lattice_constants : list of list of floats
        List of list of lattice parameters of detected phases.

    Notes
    -----
    Regular expressions are used to collect phase name, formula and
    symmetry. This function have been tested with files from the following
    vendor's formats: EDAX TSL OIM Data Collection v7, ASTAR Index, and
    EMsoft v4/v5.
    """
    regexps = {
        "name": "# MaterialName([ \t]+)([A-z0-9 ]+)",
        "formula": "# Formula([ \t]+)([A-z0-9 ]+)",
        "symmetry": "# Symmetry([ \t]+)([A-z0-9 ]+)",
        "lattice_constants": r"# LatticeConstants([ \t+])(.*)",
    }
    phases = {"name": [], "formula": [], "symmetry": [], "lattice_constants": []}
    for line in header:
        for key, exp in regexps.items():
            match = re.search(exp, line)
            if match:
                group = re.split("[ \t]", match.group(2).lstrip(" ").rstrip(" "))
                group = list(filter(None, group))
                if key == "lattice_constants":
                    group = [float(i) for i in group]
                else:
                    group = group[0]
                phases[key].append(group)

    # Check if formula is empty (sometimes the case for ASTAR Index)
    names = phases["formula"]
    if len(names) == 0 or any([i != "" for i in names]):
        names = phases["name"]

    return names, phases["symmetry"], phases["lattice_constants"]