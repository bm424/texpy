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

"""Module for generating grids in orientation spaces."""

import numpy as np

from orix.gridding.gridding_utils import (
    create_equispaced_grid,
    _get_proper_point_group,
)
from orix.quaternion.orientation_region import OrientationRegion


def get_grid_fundamental(resolution=2, point_group=None, space_group=None):
    """
    Generates an equispaced grid of rotations within a fundamental zone.

    Parameters
    ----------
    resolution : float
        The smallest distance between a rotation and its neighbour (degrees)
    point_group : orix.quaternion.symmetry.Symmetry
        One of the 11 proper point groups
    space_group: int
        Between 1 and 231

    Returns
    -------
    q : orix.quaternion.rotation.Rotation
        Grid of rotations lying within the specified fundamental zone

    See Also
    --------
    orix.gridding.utils.create_equispaced_grid

    Examples
    --------
    >>> from orix.quaternion.symmetry import C2,C4
    >>> grid = get_grid_fundamental(1, point_group=C2)
    """
    if point_group is None:
        point_group = _get_proper_point_group(space_group)

    q = create_equispaced_grid(resolution)
    fundamental_region = OrientationRegion.from_symmetry(point_group)
    return q[q < fundamental_region]


def get_grid_local(resolution=2, center=None, grid_width=10):
    """
    Generates a grid of rotations about a given rotation

    Parameters
    ----------
    resolution : float
        The smallest distance between a rotation and its neighbour (degrees)
    center : orix.quaternion.rotation.Rotation
        The rotation at which the grid is centered. If None uses the identity
    grid_width : float
        The largest angle of rotation away from center that is acceptable (degrees)

    Returns
    -------
    q : orix.quaternion.rotation.Rotation
        Grid of rotations lying within grid_width of center

    See Also
    --------
    orix.gridding_utils.create_equispaced_grid
    """

    q = create_equispaced_grid(resolution)
    grid_cosine = np.arccos(np.deg2rad(grid_width / 2))
    q = q[q.a > grid_cosine]
    if center is not None:
        q = center * q
    return q