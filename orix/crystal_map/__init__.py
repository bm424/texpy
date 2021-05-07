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

"""Crystallographic map of rotations, crystal phases and key properties
associated with every spatial coordinate in a 1D, 2D or 3D space.

All map properties with a value in each data point are stored as 1D
arrays.
"""

from orix.crystal_map.crystal_map import CrystalMap, create_coordinate_arrays
from orix.crystal_map.crystal_map_properties import CrystalMapProperties
from orix.crystal_map.phase_list import Phase, PhaseList

# Lists what will be imported when calling "from orix.crystal_map import *"
__all__ = [
    "create_coordinate_arrays",
    "CrystalMap",
    "CrystalMapProperties",
    "Phase",
    "PhaseList",
]
