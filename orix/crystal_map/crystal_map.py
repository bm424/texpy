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

import copy

import matplotlib.pyplot as plt
import numpy as np

from orix.crystal_map.crystal_map_properties import CrystalMapProperties
from orix.crystal_map.phase_list import Phase, PhaseList
from orix.quaternion.orientation import Orientation
from orix.quaternion.rotation import Rotation


class CrystalMap:
    """Crystallographic map of rotations, crystal phases and key
    properties associated with every spatial coordinate in a 1D, 2D or 3D
    space.

    All properties are stored as 1D arrays, and reshaped when necessary.
    """

    def __init__(
        self,
        rotations,
        phase_id=None,
        x=None,
        y=None,
        z=None,
        phase_list=None,
        prop=None,
        scan_unit=None,
        is_in_data=None,
    ):
        """
        Parameters
        ----------
        rotations : orix.quaternion.rotation.Rotation
            Rotation of each data point. Must be passed with all spatial
            dimensions in the first array axis (flattened). May contain
            multiple rotations per point, included in the second array
            axes. Crystal map data size is set equal to the first array
            axis' size.
        phase_id : numpy.ndarray, optional
            Phase ID of each pixel. IDs equal to -1 are considered not
            indexed. If None is passed (default), all points are
            considered to belong to one phase with ID 0.
        x : numpy.ndarray, optional
            Map x coordinate of each data point. If None is passed,
            the map is assumed to be 1D, and it is set to an array of
            increasing integers from 0 to the length of the `phase_id`
            array.
        y : numpy.ndarray, optional
            Map y coordinate of each data point. If None is passed,
            the map is assumed to be 1D, and it is set to None.
        z : numpy.ndarray, optional
            Map z coordinate of each data point. If None is passed, the
            map is assumed to be 2D or 1D, and it is set to None.
        phase_list : PhaseList, optional
            A list of phases in the data with their with names,
            space groups, point groups, and structures. The order in which
            the phases appear in the list is important, as it is this, and
            not the phases' IDs, that is used to link the phases to the
            input `phase_id` if the IDs aren't exactly the same as in
            `phase_id`. If None (default), a phase list with as many
            phases as there are unique phase IDs in `phase_id` is created.
        prop : dict of numpy.ndarray, optional
            Dictionary of properties of each data point.
        scan_unit : str, optional
            Length unit of the data. If None (default), "px" is used.
        is_in_data : np.ndarray, optional
            Array of booleans signifying whether a point is in the data.

        Examples
        --------
        >>> from diffpy.structure import Atom, Lattice, Structure
        >>> import numpy as np
        >>> from orix.crystal_map import CrystalMap
        >>> from orix.quaternion.rotation import Rotation
        >>> euler1, euler2, euler3, x, y, iq, dp, phase_id = np.loadtxt(
        ...     "/some/file.ang", unpack=True)
        >>> euler_angles = np.column_stack((euler1, euler2, euler3))
        >>> rotations = Rotation.from_euler(euler_angles)
        >>> properties = {"iq": iq, "dp": dp}
        >>> structures = [
        ...     Structure(
        ...         title="austenite",
        ...         atoms=[Atom("fe", [0] * 3)],
        ...         lattice=Lattice(0.360, 0.360, 0.360, 90, 90, 90)
        ...     ),
        ...     Structure(
        ...         title="ferrite",
        ...         atoms=[Atom("fe", [0] * 3)],
        ...         lattice=Lattice(0.287, 0.287, 0.287, 90, 90, 90)
        ...     )
        ... ]
        >>> pl = PhaseList(space_groups=[225, 229], structures=structures)
        >>> cm = CrystalMap(
        ...     rotations=rotations,
        ...     phase_id=phase_id,
        ...     x=x,
        ...     y=y,
        ...     phase_list=pl,
        ...     prop=properties,
        ... )
        """
        # Set rotations
        if not isinstance(rotations, Rotation):
            raise ValueError(
                f"rotations must be of type {Rotation}, not {type(rotations)}."
            )
        self._rotations = rotations

        # Set data size
        data_size = rotations.shape[0]

        # Set phase IDs
        if phase_id is None:  # Assume single phase data
            phase_id = np.zeros(data_size)
        phase_id = phase_id.astype(int)
        self._phase_id = phase_id

        # Set data point IDs
        point_id = np.arange(data_size)
        self._id = point_id

        # Set spatial coordinates
        if x is None and y is None and z is None:
            x = np.arange(data_size)
        self._x = x
        self._y = y
        self._z = z

        # Create phase list
        # Sorted in ascending order
        unique_phase_ids = np.unique(phase_id)
        include_not_indexed = False
        if unique_phase_ids[0] == -1:
            include_not_indexed = True
            unique_phase_ids = unique_phase_ids[1:]
        # Also sorted in ascending order
        if phase_list is None:
            self.phases = PhaseList(ids=unique_phase_ids)
        else:
            phase_list = copy.deepcopy(phase_list)
            phase_ids = phase_list.ids
            n_different = len(phase_ids) - len(unique_phase_ids)
            if n_different > 0:
                # Remove superfluous phases by removing the phases whose
                # ID is not in the ID array, in descending list order
                for i in phase_ids[::-1]:
                    if i not in unique_phase_ids:
                        del phase_list[i]
                        n_different -= 1
                    if n_different == 0:
                        break
            elif n_different < 0:
                # Create new phase list adding the missing phases with
                # default initial values
                phase_list = PhaseList(
                    names=phase_list.names,
                    space_groups=phase_list.space_groups,
                    point_groups=phase_list.point_groups,
                    colors=phase_list.colors,
                    structures=phase_list.structures,
                    ids=unique_phase_ids,
                )
            # Ensure phase list IDs correspond to IDs in phase_id array
            new_ids = list(unique_phase_ids.astype(int))
            phase_list._dict = dict(zip(new_ids, phase_list._dict.values()))
            self.phases = phase_list

        # Set whether measurements are indexed
        is_indexed = np.ones(data_size, dtype=bool)
        is_indexed[np.where(phase_id == -1)] = False

        # Add "not_indexed" to phase list and ensure not indexed points
        # have correct phase ID
        if include_not_indexed:
            self.phases.add_not_indexed()
            self._phase_id[~is_indexed] = -1

        # Set array with True for points in data
        if is_in_data is None:
            is_in_data = np.ones(data_size, dtype=bool)
        self.is_in_data = is_in_data

        # Set scan unit
        if scan_unit is None:
            scan_unit = "px"
        self.scan_unit = scan_unit

        # Set properties
        if prop is None:
            prop = {}
        self._prop = CrystalMapProperties(prop, id=point_id)

        # Set original data shape (needed if data shape changes in
        # __getitem__())
        self._original_shape = self._data_shape_from_coordinates()

    @property
    def id(self):
        """ID of points in data."""
        return self._id[self.is_in_data]

    @property
    def size(self):
        """Total number of points in data."""
        return np.count_nonzero(self.is_in_data)

    @property
    def shape(self):
        """Shape of points in data."""
        return self._data_shape_from_coordinates()

    @property
    def ndim(self):
        """Number of data dimensions of points in data."""
        return len(self.shape)

    @property
    def x(self):
        """X coordinates of points in data."""
        if self._x is None or len(np.unique(self._x)) == 1:
            return None
        else:
            return self._x[self.is_in_data]

    @property
    def y(self):
        """Y coordinates of points in data."""
        if self._y is None or len(np.unique(self._y)) == 1:
            return None
        else:
            return self._y[self.is_in_data]

    @property
    def z(self):
        """Z coordinates of points in data."""
        if self._z is None or len(np.unique(self._z)) == 1:
            return None
        else:
            return self._z[self.is_in_data]

    @property
    def dx(self):
        return self._step_size_from_coordinates(self._x)

    @property
    def dy(self):
        return self._step_size_from_coordinates(self._y)

    @property
    def dz(self):
        return self._step_size_from_coordinates(self._z)

    @property
    def phase_id(self):
        """Phase IDs of points in data."""
        return self._phase_id[self.is_in_data]

    @phase_id.setter
    def phase_id(self, value):
        """Set phase ID of points in data by passing an int to `value`."""
        self._phase_id[self.is_in_data] = value
        if value == -1 and "not_indexed" not in self.phases.names:
            self.phases.add_not_indexed()

    @property
    def phases_in_data(self):
        """List of phases in data.

        Needed because it can be useful to have phases not in data but in
        `self.phases`.
        """
        unique_ids = np.unique(self.phase_id)
        phase_list = self.phases[np.intersect1d(unique_ids, self.phases.ids)]
        if isinstance(phase_list, Phase):  # One phase in data
            # Get phase ID so it carries over to the new `PhaseList` object
            phase = phase_list  # Since it's actually a single phase
            phase_id = self.phases.id_from_name(phase.name)
            return PhaseList(phases=phase, ids=phase_id)
        else:  # Multiple phases in data
            return phase_list

    @property
    def rotations(self):
        """Rotations in data."""
        return self._rotations[self.is_in_data]

    @property
    def rotations_per_point(self):
        """Number of rotations per data point in data."""
        return self.rotations.size // self.is_indexed.size

    @property
    def rotations_shape(self):
        """Shape of rotation object.

        Map shape and possible multiple rotations per point are accounted
        for. 1-dimensions are squeezed out.
        """
        return tuple(i for i in self.shape + (self.rotations_per_point,) if i != 1)

    @property
    def orientations(self):
        """Rotations, respecting symmetry, in data."""
        # TODO: Consider whether orientations should be calculated upon
        #  loading since computing orientations are slow (should benefit
        #  from dask!)
        phases = self.phases_in_data
        if phases.size == 1:
            # Extract top matching rotations per point, if more than one
            if self.rotations_per_point > 1:
                rotations = self.rotations[:, 0]
            else:
                rotations = self.rotations
            return Orientation(rotations).set_symmetry(phases[:].point_group)
        else:
            raise ValueError(
                f"Data has the phases {phases.names}, however, you are executing a "
                "command that only permits one phase."
            )

    @property
    def is_indexed(self):
        """Whether points in data are indexed."""
        return self.phase_id != -1

    @property
    def all_indexed(self):
        """Whether all points in data are indexed."""
        return np.count_nonzero(self.is_indexed) == self.is_indexed.size

    @property
    def prop(self):
        """:class:`~orix.crystal_map.CrystalMapProperties` dictionary with
        data properties in each data point.
        """
        self._prop.is_in_data = self.is_in_data
        self._prop.id = self.id
        return self._prop

    @property
    def _coordinates(self):
        """Dictionary of coordinates of points in data."""
        # TODO: Make this "dynamic"/dependable when enabling specimen
        #  reference frame
        return {"z": self.z, "y": self.y, "x": self.x}

    @property
    def _step_sizes(self):
        """Dictionary of step sizes of dimensions in data."""
        # TODO: Make this "dynamic"/dependable when enabling specimen
        #  reference frame
        return {"z": self.dz, "y": self.dy, "x": self.dx}

    @property
    def _coordinate_axes(self):
        """Dictionary of which data axis corresponds to which cartesian
        coordinate.
        """
        present_coordinates = [k for k, v in self._coordinates.items() if v is not None]
        return {i: coord for i, coord in zip(range(self.ndim), present_coordinates)}

    def __getattr__(self, item):
        """Get an attribute in the `prop` dictionary directly from the
        CrystalMap object.

        Called when the default attribute access fails with an
        AttributeError.
        """
        if item in self.__getattribute__("_prop"):
            # Calls CrystalMapProperties.__getitem__()
            return self.prop[item]
        else:
            return object.__getattribute__(self, item)

    def __setattr__(self, name, value):
        """Set a class instance attribute."""
        if hasattr(self, "_prop") and name in self._prop:
            # Calls CrystalMapProperties.__setitem__()
            self.prop[name] = value
        else:
            return object.__setattr__(self, name, value)

    def __getitem__(self, key):
        """Get a masked copy of the CrystalMap object.

        Parameters
        ----------
        key : str, slice, tuple, int or boolean numpy.ndarray
            If ``str``, it must be a valid phase or "not_indexed" or
            "indexed". If ``slice`` or ``tuple``, it must be within the
            map shape. If ``int``, it must be a valid ``self.id``. If
            boolean array, it must be of map shape.

        Examples
        --------
        A CrystalMap object can be indexed in multiple ways...

        >>> cm
        Phase  Orientations       Name  Space group  Point group  Proper point group       Color
            1  5657 (48.4%)  austenite         None          432                 432    tab:blue
            2  6043 (51.6%)    ferrite         None          432                 432  tab:orange
        Properties: iq, dp
        Scan unit: um
        >>> cm.shape
        (100, 117)

        ... by slicing with slices, integers, or both

        >>> cm2 = cm[20:40, 50:60]
        >>> cm2
        Phase  Orientations       Name  Space group  Point group  Proper point group       Color
            1   148 (74.0%)  austenite         None          432                 432    tab:blue
            2    52 (26.0%)    ferrite         None          432                 432  tab:orange
        Properties: iq, dp
        Scan unit: um
        >>> cm2.shape
        (20, 10)
        >>> cm2 = cm[20:40, 3]
        >>> cm2
        Phase  Orientations       Name  Space group  Point group  Proper point group       Color
            1    16 (80.0%)  austenite         None          432                 432    tab:blue
            2     4 (20.0%)    ferrite         None          432                 432  tab:orange
        Properties: iq, dp
        Scan unit: um
        >>> cm2.shape
        (20, 3)

        Note that 1-dimensions are NOT removed

        >>> cm2 = cm[10, 10]
        >>> cm2
        Phase  Orientations     Name  Space group  Point group  Proper point group       Color
            2    1 (100.0%)  ferrite         None          432                 432  tab:orange
        Properties: iq, dp
        Scan unit: um
        >>> cm.shape
        (1, 1)

        ... by phase name(s)

        >>> cm2 = cm["austenite"]
        Phase  Orientations       Name  Space group  Point group  Proper point group     Color
            1  5657 (100.0%)  austenite         None          432                 432  tab:blue
        Properties: iq, dp
        Scan unit: um
        >>> cm2.shape
        (100, 117)
        >>> cm["austenite", "ferrite"]
        Phase  Orientations       Name  Space group  Point group  Proper point group       Color
            1  5657 (48.4%)  austenite         None          432                 432    tab:blue
            2  6043 (51.6%)    ferrite         None          432                 432  tab:orange
        Properties: iq, dp
        Scan unit: um

        ... by "indexed" and "not_indexed"

        >>> cm["indexed"]
        Phase  Orientations       Name  Space group  Point group  Proper point group       Color
            1  5657 (48.4%)  austenite         None          432                 432    tab:blue
            2  6043 (51.6%)    ferrite         None          432                 432  tab:orange
        Properties: iq, dp
        Scan unit: um
        >>> cm["not_indexed"]
        No data.

        ... or by boolean arrays ((chained) conditional(s))

        >>> cm[cm.dp > 0.81]
        Phase  Orientations       Name  Space group  Point group  Proper point group       Color
            1  4092 (44.8%)  austenite         None          432                 432    tab:blue
            2  5035 (55.2%)    ferrite         None          432                 432  tab:orange
        Properties: iq, dp
        Scan unit: um
        >>> cm[(cm.iq > np.mean(cm.iq)) & (cm.phase_id == 1)]
        Phase  Orientations       Name  Space group  Point group  Proper point group     Color
            1  1890 (100.0%)  austenite         None          432                 432  tab:blue
        Properties: iq, dp
        Scan unit: um
        """
        # TODO: Crop new map to the extremal spatial values (e.g. if all values in first
        #  or last row/column are masked out by the key), i.e. not just mask the values

        # Initiate a mask to be added to the returned copy of the
        # CrystalMap object, to ensure that only the unmasked values are
        # in the data of the copy (True in `is_in_data`). First, no points
        # are in the data, but are added if they satisfy the condition in
        # the input key.
        is_in_data = np.zeros(self.size, dtype=bool)

        # The original object might already have set some points to not be
        # in the data. If so, `is_in_data` is used to update the original
        # `is_in_data`. Since `new_is_in_data` is not initiated for all
        # key types, we declare it here and check for it later.
        new_is_in_data = None

        # Override mask values
        if isinstance(key, str) or (isinstance(key, tuple) and isinstance(key[0], str)):
            # From phase string(s)
            if not isinstance(key, tuple):  # Make single string iterable
                key = (key,)
            for k in key:
                for phase_id, phase in self.phases:
                    if k == phase.name:
                        is_in_data[self.phase_id == phase_id] = True
                    elif k.lower() == "indexed":
                        # Add all indexed phases to data
                        is_in_data[self.phase_id != -1] = True
        elif isinstance(key, np.ndarray) and key.dtype == np.bool_:
            # From boolean numpy array
            is_in_data = key
        elif isinstance(key, (slice, int)) or (
            isinstance(key, tuple)
            and any([(isinstance(i, slice) or isinstance(i, int)) for i in key])
        ):
            # From slice(s) or int
            if isinstance(key, (slice, int)):
                key = (key,)

            slices = [slice(None, None, None)] * self.ndim
            for i, k in enumerate(key):
                slices[i] = k

            new_is_in_data = np.zeros(self._original_shape, dtype=bool)  # > 1D
            new_is_in_data[tuple(slices)] = True
            # Note that although all points within slice(s) was sought,
            # points within the slice(s) which are already removed from
            # the data are still kept out by this boolean multiplication
            new_is_in_data = new_is_in_data.flatten() * self.is_in_data

        # Insert the mask into a mask with the full map shape, if not done
        # already
        if new_is_in_data is None:
            new_is_in_data = np.zeros_like(self.is_in_data, dtype=bool)  # 1D
            new_is_in_data[self.id] = is_in_data

        # Return a copy with all attributes shallow except for the mask
        new_map = copy.copy(self)
        new_map.is_in_data = new_is_in_data

        return new_map

    def __repr__(self):
        """Print a nice representation of the data."""
        if self.size == 0:
            return "No data."

        phases = self.phases_in_data
        phase_ids = self.phase_id

        # Ensure attributes set to None are treated OK
        names = ["None" if not name else name for name in phases.names]
        sg_names = ["None" if not i else i.short_name for i in phases.space_groups]
        pg_names = ["None" if not i else i.name for i in phases.point_groups]
        ppg_names = [
            "None" if not i else i.proper_subgroup.name for i in phases.point_groups
        ]

        # Determine column widths
        unique_phases = np.unique(phase_ids)
        p_sizes = [np.where(phase_ids == i)[0].size for i in unique_phases]
        id_len = 5
        ori_len = max(max([len(str(p_size)) for p_size in p_sizes]) + 9, 12)
        name_len = max(max([len(n) for n in names]), 4)
        sg_len = max(max([len(i) for i in sg_names]), 11)
        pg_len = max(max([len(i) for i in pg_names]), 11)
        ppg_len = max(max([len(i) for i in ppg_names]), 18)
        col_len = max(max([len(i) for i in phases.colors]), 5)

        # Column alignment
        align = ">"  # right ">" or left "<"

        # Header (note the two-space spacing)
        representation = (
            "{:{align}{width}}  ".format("Phase", width=id_len, align=align)
            + "{:{align}{width}}  ".format("Orientations", width=ori_len, align=align)
            + "{:{align}{width}}  ".format("Name", width=name_len, align=align)
            + "{:{align}{width}}  ".format("Space group", width=sg_len, align=align)
            + "{:{align}{width}}  ".format("Point group", width=pg_len, align=align)
            + "{:{align}{width}}  ".format(
                "Proper point group", width=ppg_len, align=align
            )
            + "{:{align}{width}}\n".format("Color", width=col_len, align=align)
        )

        # Overview of data for each phase
        for i, phase_id in enumerate(unique_phases.astype(int)):
            p_size = np.where(phase_ids == phase_id)[0].size
            p_fraction = 100 * p_size / self.size
            ori_str = f"{p_size} ({p_fraction:.1f}%)"
            representation += (
                f"{phase_id:{align}{id_len}}  "
                + f"{ori_str:{align}{ori_len}}  "
                + f"{names[i]:{align}{name_len}}  "
                + f"{sg_names[i]:{align}{sg_len}}  "
                + f"{pg_names[i]:{align}{pg_len}}  "
                + f"{ppg_names[i]:{align}{ppg_len}}  "
                + f"{phases.colors[i]:{align}{col_len}}\n"
            )

        # Properties and spatial coordinates
        props = []
        for k in self.prop.keys():
            props.append(k)
        representation += "Properties: " + ", ".join(props) + "\n"

        # Scan unit
        representation += f"Scan unit: {self.scan_unit}"

        return representation

    def deepcopy(self):
        """Return a deep copy using :func:`copy.deepcopy` function."""
        return copy.deepcopy(self)

    @classmethod
    def empty(cls, shape=None, step_sizes=None):
        """Create a crystal map of a given shape and step sizes with
        identity rotations.

        Parameters
        ----------
        shape : tuple of int, optional
            Map shape. Default is a 2D map of shape (5, 10), i.e. with
            five rows and ten columns.
        step_sizes : tuple, optional
            Map step sizes. If None (default), it is set to 1 px in each
            map direction given by `shape`.

        Returns
        -------
        CrystalMap


        Examples
        --------
        >>> from orix.crystal_map import CrystalMap
        >>> xmap = CrystalMap.empty((5, 10))
        >>> xmap
        Phase  Orientations  Name  Space group  Point group  Proper point group     Color
            0   50 (100.0%)  None         None         None                None  tab:blue
        Properties:
        Scan unit: px
        >>> xmap.plot("x")  # Increasing towards the right
        """
        d, n = create_coordinate_arrays(shape, step_sizes)
        d["rotations"] = Rotation.identity((n,))
        return cls(**d)

    def get_map_data(self, item, decimals=3, fill_value=None):
        """Return an array of a class instance attribute, with values
        equal to ``False`` in ``self.is_in_data`` set to `fill_value`, of
        map data shape.

        If `item` is "orientations"/"rotations" and there are multiple
        rotations per point, only the first rotation is used. Rotations
        are returned as Euler angles.

        Parameters
        ----------
        item : str or numpy.ndarray
            Name of the class instance attribute or a numpy.ndarray.
        decimals : int, optional
            How many decimals to round data point values to (default is
            3).
        fill_value : None, optional
            Value to fill points not in the data with. If None
            (default), np.nan is used.

        Returns
        -------
        output_array : numpy.ndarray
            Array of the class instance attribute with points not in data
            set to `fill_value`, of float data type.
        """
        # TODO: Consider an `axes` argument along which to get map data
        #  if > 2D

        # Get full map shape
        map_shape = self._original_shape

        # Declare array of correct shape, accounting for RGB
        # TODO: Better account for `item.shape`, e.g. quaternions
        #  (item.shape[-1] == 4) in a more general way than here (not more
        #  if/else)!
        map_size = np.prod(map_shape)
        if isinstance(item, np.ndarray):
            array = np.empty(map_size, dtype=item.dtype)
            if item.shape[-1] == 3:  # Assume RGB
                map_shape += (3,)
                array = np.column_stack((array,) * 3)
        elif item in ["orientations", "rotations"]:  # Definitely RGB
            array = np.empty(map_size, dtype=np.float64)
            map_shape += (3,)
            array = np.column_stack((array,) * 3)
        else:
            array = np.empty(map_size, dtype=np.float64)

        # Enter non-masked values into array
        if isinstance(item, np.ndarray):
            # TODO: Account for 2D map with more than one value per point
            array[self.is_in_data] = item
        elif item in ["orientations", "rotations"]:
            if item == "rotations":
                # Use only the top matching rotation per point
                if self.rotations_per_point > 1:
                    rotations = self.rotations[:, 0]
                else:
                    rotations = self.rotations
                array[self.is_in_data] = rotations.to_euler()
            else:  # item == "orientations"
                # Fill in orientations per phase
                # TODO: Consider whether orientations should be calculated
                #  upon loading
                for i, phase in self.phases_in_data:
                    phase_mask = (self._phase_id == i) * self.is_in_data
                    phase_mask_in_data = self.phase_id == i
                    array[phase_mask] = self[phase_mask_in_data].orientations.to_euler()
        else:  # String
            data = self.__getattr__(item)
            if data is None:
                raise ValueError(f"{item} is {data}.")
            else:
                # TODO: Account for 2D map with more than one value per point
                array[self.is_in_data] = data
                array = array.astype(data.dtype)

        # Slice and reshape array
        slices = self._data_slices_from_coordinates()
        reshaped_array = array.reshape(map_shape)
        sliced_array = reshaped_array[slices]

        # Reshape and slice mask with points not in data
        if array.shape[-1] == 3:  # RGB
            not_in_data = np.dstack((~self.is_in_data,) * 3)
        else:  # Scalar
            not_in_data = ~self.is_in_data
        not_in_data = not_in_data.reshape(map_shape)[slices]

        # Fill points not in data with the fill value
        if not_in_data.any():
            if fill_value is None or fill_value is np.nan:
                sliced_array = sliced_array.astype(np.float64)
            sliced_array[not_in_data] = fill_value

        # Round values
        if np.issubdtype(array.dtype, np.bool_):
            output_array = sliced_array
        else:
            output_array = np.round(sliced_array, decimals=decimals)

        return output_array

    def plot(
        self,
        value=None,
        overlay=None,
        scalebar=True,
        scalebar_properties=None,
        legend=True,
        legend_properties=None,
        colorbar=False,
        colorbar_label=None,
        colorbar_properties=dict(),
        remove_padding=False,
        return_figure=False,
        figure_kwargs=dict(),
        **kwargs,
    ):
        r"""Plot a 2D map with any crystallographic map property as map
        values.

        Wraps :meth:`matplotlib.axes.Axes.imshow`, see that method for
        relevant keyword arguments.

        Parameters
        ----------
        value : numpy.ndarray, optional
            Attribute array to plot. If None (default), a phase map is
            plotted.
        overlay : str or numpy.ndarray, optional
            Name of map property or a property array to use in the
            alpha (RGBA) channel. The property range is adjusted for
            maximum contrast. Default is None.
        scalebar : bool, optional
            Whether to add a scalebar (default is True) along the
            horizontal map dimension.
        scalebar_properties : dict, optional
            Keyword arguments passed to
            :class:`matplotlib_scalebar.scalebar.ScaleBar`.
        legend : bool, optional
            Whether to add a legend to the plot. This is only
            implemented for a phase plot (in which case default is
            True).
        legend_properties : dict, optional
            Keyword arguments passed to :meth:`matplotlib.axes.legend`.
        colorbar : bool, optional
            Whether to add an opinionated colorbar (default is False).
        colorbar_label : str, optional
            Label/title of colorbar.
        colorbar_properties : dict, optional
            Keyword arguments passed to
            :meth:`orix.plot.CrystalMapPlot.add_colorbar`.
        axes : tuple of ints, optional
            Which data axes to plot if data has more than two
            dimensions. The index of data to plot in the final dimension
            is determined by `depth`. If None (default), data along the
            two last axes is plotted.
        depth : int, optional
            Which layer along the third axis to plot if data has more
            than two dimensions. If None (default), data in the first
            index (layer) is plotted.
        return_figure: bool, optional
            Whether to return the figure (default is False).
        figure_kwargs : dict, optional
            Keyword arguments passed to
            :func:`matplotlib.pyplot.subplots`.
        kwargs
            Keyword arguments passed to
            :meth:`matplotlib.axes.Axes.imshow`.

        Returns
        -------
        fig : matplotlib.figure.Figure
            The created figure, returned if `return_figure` is True.

        See Also
        --------
        matplotlib.axes.Axes.imshow
        orix.plot.CrystalMapPlot.plot_map
        orix.plot.CrystalMapPlot.add_scalebar
        orix.plot.CrystalMapPlot.add_overlay
        orix.plot.CrystalMapPlot.add_colorbar
        """
        # Register "plot_map" projection with Matplotlib
        import orix.plot.crystal_map_plot

        fig, ax = plt.subplots(subplot_kw=dict(projection="plot_map"), **figure_kwargs)
        ax.plot_map(
            self,
            value=value,
            scalebar=scalebar,
            scalebar_properties=scalebar_properties,
            legend=legend,
            legend_properties=legend_properties,
            **kwargs,
        )
        if overlay is not None:
            ax.add_overlay(self, overlay)
        if remove_padding:
            ax.remove_padding()
        if colorbar:
            ax.add_colorbar(label=colorbar_label, **colorbar_properties)
        if return_figure:
            return fig

    @staticmethod
    def _step_size_from_coordinates(coordinates):
        """Return step size in input `coordinates` array.

        Parameters
        ----------
        coordinates : numpy.ndarray
            Linear coordinate array.

        Returns
        -------
        step_size : float
            Step size in `coordinates` array.
        """
        unique_sorted = np.sort(np.unique(coordinates))
        step_size = 0
        if unique_sorted.size != 1:
            step_size = unique_sorted[1] - unique_sorted[0]
        return step_size

    def _data_slices_from_coordinates(self):
        """Return a tuple of slices defining the current data extent in
        all directions.

        Returns
        -------
        slices : tuple of slices
            Data slice in each existing dimension, in (z, y, x) order.
        """
        slices = []

        # Loop over dimension coordinates and step sizes
        for coordinates, step in zip(
            self._coordinates.values(), self._step_sizes.values()
        ):
            if coordinates is not None:
                c_min, c_max = np.min(coordinates), np.max(coordinates)
                i_min = int(np.around(c_min / step))
                i_max = int(np.around((c_max / step) + 1))
                slices.append(slice(i_min, i_max))

        return tuple(slices)

    def _data_shape_from_coordinates(self):
        """Return data shape based upon coordinate arrays.

        Returns
        -------
        data_shape : tuple of ints
            Shape of data in all existing dimensions, in (z, y, x) order.
        """
        data_shape = []
        for dim_slice in self._data_slices_from_coordinates():
            data_shape.append(dim_slice.stop - dim_slice.start)
        return tuple(data_shape)


def create_coordinate_arrays(shape=None, step_sizes=None):
    """Create flattened coordinate arrays from a given map shape and
    step sizes, suitable for initializing a
    :class:`orix.crystal_map.CrystalMap`. Arrays for 1D, 2D, or 3D maps
    can be returned.

    Parameters
    ----------
    shape : tuple of int, optional
        Map shape. Default is a 2D map of shape (5, 10) with five rows
        and ten columns. Can be up to 3D.
    step_sizes : tuple, optional
        Map step sizes. If None (default), it is set to 1 px in each
        map direction given by `shape`.

    Returns
    -------
    d : dict of numpy.ndarray
        Dictionary with keys "z", "y", and "x", depending on the length
        of `shape`, with coordinate arrays.
    map_size : int
        Number of map points.
    """
    if shape is None:
        shape = (5, 10)
    ndim = len(shape)
    if step_sizes is None:
        step_sizes = (1,) * ndim

    # Set up as if a 3D map is to be returned
    dz, dy, dx = (1,) * (3 - ndim) + step_sizes
    nz, ny, nx = (1,) * (3 - ndim) + shape
    d = dict()

    # Add coordinate arrays depending on the number of map dimensions
    d["x"] = np.tile(np.arange(nx) * dx, ny * nz).flatten()
    map_size = nx
    if ndim > 1:
        d["y"] = np.tile(np.sort(np.tile(np.arange(ny) * dy, nx)), nz).flatten()
        map_size *= ny
    if ndim > 2:
        d["z"] = np.array([np.ones(ny * nx) * i * dz for i in range(nz)]).flatten()
        map_size *= nz

    return d, map_size
