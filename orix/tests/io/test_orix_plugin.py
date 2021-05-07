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

from diffpy.structure.spacegroups import GetSpaceGroup
from h5py import File
import pytest
import numpy as np

from orix import __version__ as orix_version
from orix.crystal_map import CrystalMap, Phase
from orix.io import load, save
from orix.io.plugins.orix_hdf5 import (
    hdf5group2dict,
    dict2crystalmap,
    dict2phaselist,
    dict2phase,
    dict2structure,
    dict2lattice,
    dict2atom,
    dict2hdf5group,
    crystalmap2dict,
    phaselist2dict,
    phase2dict,
    structure2dict,
    lattice2dict,
    atom2dict,
)
from orix.io.plugins import ang, emsoft_h5ebsd, orix_hdf5
from orix.tests.io.test_io import assert_dictionaries_are_equal

plugin_list = [ang, emsoft_h5ebsd, orix_hdf5]


class TestOrixHDF5Plugin:
    def test_file_writer(self, crystal_map, temp_file_path):
        save(filename=temp_file_path, object2write=crystal_map)

        with File(temp_file_path, mode="r") as f:
            assert f["manufacturer"][()][0].decode() == "orix"
            assert f["version"][()][0].decode() == orix_version

    @pytest.mark.parametrize(
        "crystal_map_input",
        [
            ((4, 4, 3), (1, 1.5, 1.5), 1, [0, 1]),
            ((2, 4, 3), (1, 1.5, 1.5), 2, [0, 1, 2]),
        ],
        indirect=["crystal_map_input"],
    )
    def test_write_read_masked(self, crystal_map_input, temp_file_path):
        cm = CrystalMap(**crystal_map_input)
        save(filename=temp_file_path, object2write=cm[cm.x > 2])
        cm2 = load(temp_file_path)

        assert cm2.size != cm.size
        with pytest.raises(ValueError, match="operands could not be broadcast"):
            _ = np.allclose(cm2.x, cm.x)

        cm2.is_in_data = cm.is_in_data
        assert cm2.size == cm.size
        assert np.allclose(cm2.x, cm.x)

    def test_file_writer_raises(self, temp_file_path, crystal_map):
        with pytest.raises(OSError, match="Cannot write to the already open file "):
            with File(temp_file_path, mode="w") as _:
                save(temp_file_path, crystal_map, overwrite=True)

    def test_dict2hdf5group(self, temp_file_path):
        with File(temp_file_path, mode="w") as f:
            group = f.create_group(name="a_group")
            with pytest.warns(UserWarning, match="The orix HDF5 writer could not"):
                dict2hdf5group(
                    dictionary={"a": [np.array(24.5)], "c": set()}, group=group
                )

    def test_crystalmap2dict(self, crystal_map_input):
        cm = CrystalMap(**crystal_map_input)
        cm_dict = crystalmap2dict(cm)

        this_dict = {"hello": "there"}
        cm_dict2 = crystalmap2dict(cm, dictionary=this_dict)

        cm_dict2.pop("hello")
        assert_dictionaries_are_equal(cm_dict, cm_dict2)

        assert np.allclose(cm_dict["data"]["x"], crystal_map_input["x"])
        assert cm_dict["header"]["z_step"] == cm.dz

    def test_phaselist2dict(self, phase_list):
        pl_dict = phaselist2dict(phase_list)
        this_dict = {"hello": "there"}
        this_dict = phaselist2dict(phase_list, dictionary=this_dict)
        this_dict.pop("hello")

        assert_dictionaries_are_equal(pl_dict, this_dict)

    def test_phase2dict(self, phase_list):
        phase_dict = phase2dict(phase_list[0])
        this_dict = {"hello": "there"}
        this_dict = phase2dict(phase_list[0], dictionary=this_dict)
        this_dict.pop("hello")

        assert_dictionaries_are_equal(phase_dict, this_dict)

    def test_phase2dict_spacegroup(self):
        """Space group is written to dict as an int or "None"."""
        sg100 = 100
        phase = Phase(space_group=sg100)
        phase_dict1 = phase2dict(phase)
        assert phase_dict1["space_group"] == sg100

        sg200 = GetSpaceGroup(200)
        phase.space_group = sg200
        phase_dict2 = phase2dict(phase)
        assert phase_dict2["space_group"] == sg200.number

        phase.space_group = None
        phase_dict3 = phase2dict(phase)
        assert phase_dict3["space_group"] == "None"

    def test_structure2dict(self, phase_list):
        structure = phase_list[0].structure
        structure_dict = structure2dict(structure)
        this_dict = {"hello": "there"}
        this_dict = structure2dict(structure, this_dict)
        this_dict.pop("hello")

        lattice1 = structure_dict["lattice"]
        lattice2 = this_dict["lattice"]
        assert np.allclose(lattice1["abcABG"], lattice2["abcABG"])
        assert np.allclose(lattice1["baserot"], lattice2["baserot"])
        assert_dictionaries_are_equal(structure_dict["atoms"], this_dict["atoms"])

    def test_hdf5group2dict_update_dict(self, temp_file_path, crystal_map):
        save(temp_file_path, crystal_map)
        with File(temp_file_path, mode="r") as f:
            this_dict = {"hello": "there"}
            this_dict = hdf5group2dict(f["crystal_map"], dictionary=this_dict)

            assert this_dict["hello"] == "there"
            assert this_dict["data"] == f["crystal_map/data"]
            assert this_dict["header"] == f["crystal_map/header"]

    def test_file_reader(self, crystal_map, temp_file_path):
        save(filename=temp_file_path, object2write=crystal_map)
        cm2 = load(filename=temp_file_path)
        assert_dictionaries_are_equal(crystal_map.__dict__, cm2.__dict__)

    def test_dict2crystalmap(self, crystal_map):
        cm2 = dict2crystalmap(crystalmap2dict(crystal_map))
        assert_dictionaries_are_equal(crystal_map.__dict__, cm2.__dict__)

    def test_dict2phaselist(self, phase_list):
        phase_list2 = dict2phaselist(phaselist2dict(phase_list))

        assert phase_list.size == phase_list2.size
        assert phase_list.ids == phase_list2.ids
        assert phase_list.names == phase_list2.names
        assert phase_list.colors == phase_list2.colors
        assert [
            s1.name == s2.name
            for s1, s2 in zip(phase_list.point_groups, phase_list2.point_groups)
        ]

    def test_dict2phase(self, phase_list):
        phase1 = phase_list[0]
        phase2 = dict2phase(phase2dict(phase1))

        assert phase1.name == phase2.name
        assert phase1.color == phase2.color
        assert phase1.space_group.number == phase2.space_group.number
        assert phase1.point_group.name == phase2.point_group.name
        assert phase1.structure.lattice.abcABG() == phase2.structure.lattice.abcABG()

    def test_dict2phase_spacegroup(self):
        """Space group number int or None is properly parsed from a dict.
        """
        phase1 = Phase(space_group=200)
        phase_dict = phase2dict(phase1)
        phase2 = dict2phase(phase_dict)
        assert phase1.space_group.number == phase2.space_group.number

        phase_dict.pop("space_group")
        phase3 = dict2phase(phase_dict)
        assert phase3.space_group is None

    def test_dict2structure(self, phase_list):
        structure1 = phase_list[0].structure
        structure2 = dict2structure(structure2dict(structure1))

        lattice1 = structure1.lattice
        lattice2 = structure2.lattice
        assert lattice1.abcABG() == lattice2.abcABG()
        assert np.allclose(lattice1.baserot, lattice2.baserot)

        assert str(structure1.element) == str(structure2.element)
        assert np.allclose(structure1.xyz, structure2.xyz)

    def test_dict2lattice(self, phase_list):
        lattice = phase_list[0].structure.lattice
        lattice2 = dict2lattice(lattice2dict(lattice))

        assert lattice.abcABG() == lattice2.abcABG()
        assert np.allclose(lattice.baserot, lattice2.baserot)

    def test_dict2atom(self, phase_list):
        atom = phase_list[0].structure[0]
        atom2 = dict2atom(atom2dict(atom))

        assert str(atom.element) == str(atom2.element)
        assert np.allclose(atom.xyz, atom2.xyz)

    def test_read_point_group_from_v0_3_x(self, temp_file_path, crystal_map):
        crystal_map.phases[0].point_group = "1"
        save(filename=temp_file_path, object2write=crystal_map)

        # First, ensure point group data set name is named "symmetry", as in v0.3.0
        with File(temp_file_path, mode="r+") as f:
            for phase in f["crystal_map/header/phases"].values():
                phase["symmetry"] = phase["point_group"]
                del phase["point_group"]

        # Then, make sure it can still be read
        cm2 = load(filename=temp_file_path)
        # And that the symmetry operations are the same, for good measure
        print(crystal_map)
        print(cm2)
        assert np.allclose(
            crystal_map.phases[0].point_group.data, cm2.phases[0].point_group.data
        )

    def test_write_read_nd_crystalmap_properties(self, temp_file_path, crystal_map):
        """Crystal map properties with more than one value in each point
        (e.g. top matching scores from dictionary indexing) can be written
        and read from file correctly.
        """
        xmap = crystal_map
        map_size = xmap.size

        prop2d_name = "prop2d"
        prop2d_shape = (map_size, 2)
        prop2d = np.arange(map_size * 2).reshape(prop2d_shape)
        xmap.prop[prop2d_name] = prop2d

        prop3d_name = "prop3d"
        prop3d_shape = (map_size, 2, 2)
        prop3d = np.arange(map_size * 4).reshape(prop3d_shape)
        xmap.prop[prop3d_name] = prop3d

        save(filename=temp_file_path, object2write=xmap)
        xmap2 = load(temp_file_path)

        assert np.allclose(xmap2.prop[prop2d_name], xmap.prop[prop2d_name])
        assert np.allclose(xmap2.prop[prop3d_name], xmap.prop[prop3d_name])
        assert np.allclose(xmap2.prop[prop2d_name].reshape(prop2d_shape), prop2d)
        assert np.allclose(xmap2.prop[prop3d_name].reshape(prop3d_shape), prop3d)

    def test_write_read_2d(self, temp_file_path, crystal_map_input):
        crystal_map_input["z"] = None
        xmap = CrystalMap(**crystal_map_input)
        assert xmap._coordinates["z"] == None
        save(filename=temp_file_path, object2write=xmap)
        xmap2 = load(temp_file_path)
        assert xmap2._coordinates["z"] == None
