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

import pytest
import numpy as np

from orix.crystal_map import CrystalMap, Phase
from orix.io import load, loadang, save
from orix.io.plugins.ang import (
    _get_header,
    _get_phases_from_header,
    _get_vendor_columns,
    _get_nrows_ncols_step_sizes,
    _get_column_width,
)
from orix.tests.conftest import (
    ANGFILE_TSL_HEADER,
    ANGFILE_ASTAR_HEADER,
    ANGFILE_EMSOFT_HEADER,
)


@pytest.mark.parametrize(
    "angfile_astar, expected_data",
    [
        (
            (
                (2, 5),
                (1, 1),
                np.ones(2 * 5, dtype=int),
                np.array(
                    [
                        [4.485496, 0.952426, 0.791507],
                        [1.343904, 0.276111, 0.825890],
                        [1.343904, 0.276111, 0.825890],
                        [1.343904, 0.276111, 0.825890],
                        [4.555309, 2.895152, 3.972020],
                        [1.361357, 0.276111, 0.825890],
                        [4.485496, 0.220784, 0.810182],
                        [0.959931, 2.369110, 4.058938],
                        [0.959931, 2.369110, 4.058938],
                        [4.485496, 0.220784, 0.810182],
                    ],
                ),
            ),
            np.array(
                [
                    [0.77861956, -0.12501022, 0.44104243, 0.42849224],
                    [0.46256046, -0.13302712, -0.03524667, -0.87584204],
                    [0.46256046, -0.13302712, -0.03524667, -0.87584204],
                    [0.46256046, -0.13302712, -0.03524667, -0.87584204],
                    [0.05331986, 0.95051048, 0.28534763, -0.11074093],
                    [0.45489991, -0.13271448, -0.03640618, -0.87984517],
                    [0.8752001, -0.02905178, 0.10626836, 0.47104969],
                    [0.3039118, 0.01972273, -0.92612154, 0.22259272],
                    [0.3039118, 0.01972273, -0.92612154, 0.22259272],
                    [0.8752001, -0.02905178, 0.10626836, 0.47104969],
                ]
            ),
        ),
    ],
    indirect=["angfile_astar"],
)
def test_loadang(angfile_astar, expected_data):
    loaded_data = loadang(angfile_astar)
    assert np.allclose(loaded_data.data, expected_data)


class TestAngReader:
    @pytest.mark.parametrize(
        "angfile_tsl, map_shape, step_sizes, phase_id, example_rot",
        [
            (
                # Read by angfile_tsl() via request.param (passed via `indirect` below)
                (
                    (5, 3),  # map_shape
                    (0.1, 0.1),  # step_sizes
                    np.zeros(5 * 3, dtype=int),  # phase_id
                    5,  # n_unknown_columns
                    np.array(
                        [[1.59942, 2.37748, 4.53419], [1.59331, 2.37417, 4.53628]]
                    ),  # rotations as rows of Euler angle triplets
                ),
                (5, 3),
                (0.1, 0.1),
                np.zeros(5 * 3, dtype=int),
                np.array(
                    [[1.59942, 2.37748, -1.74690], [1.59331, 2.37417, -1.74899]]
                ),  # rotations as rows of Euler angle triplets
            ),
            (
                (
                    (8, 4),  # map_shape
                    (1.5, 1.5),  # step_sizes
                    np.zeros(8 * 4, dtype=int),  # phase_id
                    5,  # n_unknown_columns
                    np.array(
                        [[5.81107, 2.34188, 4.47345], [6.16205, 0.79936, 1.31702]]
                    ),  # rotations as rows of Euler angle triplets
                ),
                (8, 4),
                (1.5, 1.5),
                np.zeros(8 * 4, dtype=int),
                np.array(
                    [[-0.12113, 2.34188, 1.31702], [-0.47211, 0.79936, -1.80973]]
                ),  # rotations as rows of Euler angle triplets
            ),
        ],
        indirect=["angfile_tsl"],
    )
    def test_load_ang_tsl(
        self, angfile_tsl, map_shape, step_sizes, phase_id, example_rot,
    ):
        xmap = load(angfile_tsl)

        # Fraction of non-indexed points
        non_indexed_fraction = int(np.prod(map_shape) * 0.1)
        assert non_indexed_fraction == np.sum(~xmap.is_indexed)

        # Properties
        assert list(xmap.prop.keys()) == [
            "iq",
            "ci",
            "unknown1",
            "fit",
            "unknown2",
            "unknown3",
            "unknown4",
            "unknown5",
        ]

        # Coordinates
        ny, nx = map_shape
        dy, dx = step_sizes
        assert np.allclose(xmap.x, np.tile(np.arange(nx) * dx, ny))
        assert np.allclose(xmap.y, np.sort(np.tile(np.arange(ny) * dy, nx)))

        # Map shape and size
        assert xmap.shape == map_shape
        assert xmap.size == np.prod(map_shape)

        # Attributes are within expected ranges or have a certain value
        assert xmap.prop["ci"].max() <= 1
        assert xmap["indexed"].fit.max() <= 3
        assert all(xmap["not_indexed"].fit == 180)
        assert all(xmap["not_indexed"].ci == -1)

        # Phase IDs (accounting for non-indexed points)
        phase_id[xmap["not_indexed"].id] = -1
        assert np.allclose(xmap.phase_id, phase_id)

        # Rotations
        rot_unique = np.unique(xmap["indexed"].rotations.to_euler(), axis=0)
        assert np.allclose(
            np.sort(rot_unique, axis=0), np.sort(example_rot, axis=0), atol=1e-5
        )
        assert np.allclose(
            xmap["not_indexed"].rotations.to_euler()[0],
            np.array([np.pi, 0, np.pi]),
            atol=1e-5,
        )

        # Phases
        assert xmap.phases.size == 2  # Including non-indexed
        assert xmap.phases.ids == [-1, 0]
        phase = xmap.phases[0]
        assert phase.name == "Aluminum"
        assert phase.point_group.name == "432"

    @pytest.mark.parametrize(
        "angfile_astar, map_shape, step_sizes, phase_id, example_rot",
        [
            (
                # Read by angfile_astar() via request.param (passed via `indirect`
                # below)
                (
                    (9, 3),  # map_shape
                    (4.5, 4.5),  # step_sizes
                    np.ones(9 * 3, dtype=int),  # phase_id
                    np.array(
                        [
                            [1.895079, 0.739496, 1.413542],
                            [1.897871, 0.742638, 1.413717],
                        ]
                    ),
                ),
                (9, 3),
                (4.5, 4.5),
                np.ones(9 * 3, dtype=int),
                np.array(
                    [[1.895079, 0.739496, 1.413542], [1.897871, 0.742638, 1.413717]]
                ),
            ),
            (
                (
                    (11, 13),  # map_shape
                    (10, 10),  # step_sizes
                    np.ones(11 * 13, dtype=int),  # phase_id
                    np.array(
                        [
                            [1.621760, 2.368935, 4.559324],
                            [1.604481, 2.367539, 4.541870],
                        ]
                    ),
                ),
                (11, 13),
                (10, 10),
                np.ones(11 * 13, dtype=int),
                np.array(
                    [[1.621760, 2.368935, -1.723861], [1.604481, 2.367539, -1.741315]]
                ),
            ),
        ],
        indirect=["angfile_astar"],
    )
    def test_load_ang_astar(
        self, angfile_astar, map_shape, step_sizes, phase_id, example_rot,
    ):
        cm = load(angfile_astar)

        # Properties
        assert list(cm.prop.keys()) == ["ind", "rel", "relx100"]

        # Coordinates
        ny, nx = map_shape
        dy, dx = step_sizes
        assert np.allclose(cm.x, np.tile(np.arange(nx) * dx, ny))
        assert np.allclose(cm.y, np.sort(np.tile(np.arange(ny) * dy, nx)))

        # Map shape and size
        assert cm.shape == map_shape
        assert cm.size == np.prod(map_shape)

        # Attributes are within expected ranges or have a certain value
        assert cm.prop["ind"].max() <= 100
        assert cm.prop["rel"].max() <= 1
        assert cm.prop["relx100"].max() <= 100
        relx100 = (cm.prop["rel"] * 100).astype(int)
        assert np.allclose(cm.prop["relx100"], relx100)

        # Phase IDs
        assert np.allclose(cm.phase_id, phase_id)

        # Rotations
        rot_unique = np.unique(cm.rotations.to_euler(), axis=0)
        assert np.allclose(
            np.sort(rot_unique, axis=0), np.sort(example_rot, axis=0), atol=1e-6
        )

        # Phases
        assert cm.phases.size == 1
        assert cm.phases.ids == [1]
        phase = cm.phases[1]
        assert phase.name == "Nickel"
        assert phase.point_group.name == "432"

    @pytest.mark.parametrize(
        "angfile_emsoft, map_shape, step_sizes, phase_id, example_rot",
        [
            (
                # Read by angfile_emsoft() via request.param (passed via `indirect`
                # below)
                (
                    (10, 11),  # map_shape
                    (4.5, 4.5),  # step_sizes
                    np.concatenate(
                        (
                            np.ones(int(np.ceil((10 * 11) / 2))),
                            np.ones(int(np.floor((10 * 11) / 2))) * 2,
                        )
                    ),  # phase_id
                    np.array(
                        [
                            [1.895079, 0.739496, 1.413542],
                            [1.897871, 0.742638, 1.413717],
                        ]
                    ),
                ),
                (10, 11),
                (4.5, 4.5),
                np.concatenate(
                    (
                        np.ones(int(np.ceil((10 * 11) / 2))),
                        np.ones(int(np.floor((10 * 11) / 2))) * 2,
                    )
                ),
                np.array(
                    [[1.895079, 0.739496, 1.413542], [1.897871, 0.742638, 1.413717]]
                ),
            ),
            (
                (
                    (3, 6),  # map_shape
                    (10, 10),  # step_sizes
                    np.concatenate(
                        (
                            np.ones(int(np.ceil((3 * 6) / 2))),
                            np.ones(int(np.floor((3 * 6) / 2))) * 2,
                        )
                    ),  # phase_id
                    np.array(
                        [[1.62176, 2.36894, -1.72386], [1.60448, 2.36754, -1.72386]]
                    ),
                ),
                (3, 6),
                (10, 10),
                np.concatenate(
                    (
                        np.ones(int(np.ceil((3 * 6) / 2))),
                        np.ones(int(np.floor((3 * 6) / 2))) * 2,
                    )
                ),
                np.array([[1.62176, 2.36894, -1.72386], [1.60448, 2.36754, -1.72386]]),
            ),
        ],
        indirect=["angfile_emsoft"],
    )
    def test_load_ang_emsoft(
        self, angfile_emsoft, map_shape, step_sizes, phase_id, example_rot,
    ):
        cm = load(angfile_emsoft)

        # Properties
        assert list(cm.prop.keys()) == ["iq", "dp"]

        # Coordinates
        ny, nx = map_shape
        dy, dx = step_sizes
        assert np.allclose(cm.x, np.tile(np.arange(nx) * dx, ny))
        assert np.allclose(cm.y, np.sort(np.tile(np.arange(ny) * dy, nx)))

        # Map shape and size
        assert cm.shape == map_shape
        assert cm.size == np.prod(map_shape)

        # Attributes are within expected ranges or have a certain value
        assert cm.prop["iq"].max() <= 100
        assert cm.prop["dp"].max() <= 1

        # Phase IDs
        assert np.allclose(cm.phase_id, phase_id)

        # Rotations
        rot_unique = np.unique(cm.rotations.to_euler(), axis=0)
        assert np.allclose(
            np.sort(rot_unique, axis=0), np.sort(example_rot, axis=0), atol=1e-5
        )

        # Phases (change if file header is changed!)
        phases_in_data = cm["indexed"].phases_in_data
        assert phases_in_data.size == 2
        assert phases_in_data.ids == [1, 2]
        assert phases_in_data.names == ["austenite", "ferrite/ferrite"]
        assert [i.name for i in phases_in_data.point_groups] == ["432"] * 2

    def test_get_header(self, temp_ang_file):
        temp_ang_file.write(ANGFILE_ASTAR_HEADER)
        temp_ang_file.close()
        assert _get_header(open(temp_ang_file.name)) == [
            "# File created from ACOM RES results",
            "# ni-dislocations.res",
            "#     ".rstrip(),
            "#     ".rstrip(),
            "# MaterialName      Nickel",
            "# Formula",
            "# Symmetry          43",
            "# LatticeConstants  3.520  3.520  3.520  90.000  90.000  90.000",
            "# NumberFamilies    4",
            "# hklFamilies       1  1  1 1 0.000000",
            "# hklFamilies       2  0  0 1 0.000000",
            "# hklFamilies       2  2  0 1 0.000000",
            "# hklFamilies       3  1  1 1 0.000000",
            "#",
            "# GRID: SqrGrid#",
        ]

    @pytest.mark.parametrize(
        "expected_vendor, expected_columns, vendor_header",
        [
            (
                "tsl",
                [
                    "iq",
                    "ci",
                    "phase_id",
                    "unknown1",
                    "fit",
                    "unknown2",
                    "unknown3",
                    "unknown4",
                    "unknown5",
                ],
                ANGFILE_TSL_HEADER,
            ),
            ("astar", ["ind", "rel", "phase_id", "relx100"], ANGFILE_ASTAR_HEADER),
            ("emsoft", ["iq", "dp", "phase_id"], ANGFILE_EMSOFT_HEADER),
        ],
    )
    def test_get_vendor_columns(
        self, expected_vendor, expected_columns, vendor_header, temp_ang_file
    ):
        expected_columns = ["euler1", "euler2", "euler3", "x", "y"] + expected_columns
        n_cols_file = len(expected_columns)

        temp_ang_file.write(vendor_header)
        temp_ang_file.close()
        header = _get_header(open(temp_ang_file.name))
        vendor, column_names = _get_vendor_columns(header, n_cols_file)

        assert vendor == expected_vendor
        assert column_names == expected_columns

    @pytest.mark.parametrize("n_cols_file", [15, 20])
    def test_get_vendor_columns_unknown(self, temp_ang_file, n_cols_file):
        temp_ang_file.write("Look at me!\nI'm Mr. .ang file!\n")
        temp_ang_file.close()
        header = _get_header(open(temp_ang_file.name))
        with pytest.warns(UserWarning, match=f"Number of columns, {n_cols_file}, "):
            vendor, column_names = _get_vendor_columns(header, n_cols_file)
            assert vendor == "unknown"
            expected_columns = [
                "euler1",
                "euler2",
                "euler3",
                "x",
                "y",
                "unknown1",
                "unknown2",
                "phase_id",
            ] + ["unknown" + str(i + 3) for i in range(n_cols_file - 8)]
            assert column_names == expected_columns

    @pytest.mark.parametrize(
        "header_phase_part, expected_names, expected_point_groups, "
        "expected_lattice_constants, expected_phase_id",
        [
            (
                [
                    [
                        "# Phase 42",
                        "# MaterialName      Nickel",
                        "# Formula",
                        "# Symmetry          43",
                        "# LatticeConstants  3.520  3.520  3.520  90.000  90.000  "
                        "90.000",
                    ],
                    [
                        "# MaterialName      Aluminium",
                        "# Formula  Al",
                        "# Symmetry          m3m",
                        "# LatticeConstants  3.520  3.520  3.520  90.000  90.000  "
                        "90.000",
                    ],
                ],
                ["Nickel", "Aluminium"],
                ["43", "m3m"],
                [[3.52, 3.52, 3.52, 90, 90, 90], [3.52, 3.52, 3.52, 90, 90, 90]],
                [42, 43],
            ),
        ],
    )
    def test_get_phases_from_header(
        self,
        header_phase_part,
        expected_names,
        expected_point_groups,
        expected_lattice_constants,
        expected_phase_id,
    ):
        # Create header from parts
        header = [
            "# File created from ACOM RES results",
            "# ni-dislocations.res",
            "#     ",
            "#     ",
        ]
        hkl_families = [
            "# NumberFamilies    4",
            "# hklFamilies       1  1  1 1 0.000000",
            "# hklFamilies       2  0  0 1 0.000000",
            "# hklFamilies       2  2  0 1 0.000000",
            "# hklFamilies       3  1  1 1 0.000000",
        ]
        for phase in header_phase_part:
            header += phase + hkl_families
        header += [
            "#",
            "# GRID: SqrGrid#",
        ]
        ids, names, point_groups, lattice_constants = _get_phases_from_header(header)

        assert names == expected_names
        assert point_groups == expected_point_groups
        assert np.allclose(lattice_constants, expected_lattice_constants)
        assert np.allclose(ids, expected_phase_id)


class TestAngWriter:
    def test_write_read_loop(self, crystal_map, tmp_path):
        fname = tmp_path / "test_write_read_loop.ang"
        save(filename=fname, object2write=crystal_map)
        xmap_reload = load(filename=fname)

        assert np.allclose(
            xmap_reload.rotations.to_euler(), crystal_map.rotations.to_euler(),
        )
        assert np.allclose(xmap_reload.phase_id - 1, crystal_map.phase_id)

    @pytest.mark.parametrize(
        "crystal_map_input, desired_shape, desired_step_sizes",
        [
            (((1, 4, 3), (1, 2, 3), 1, [0]), (4, 3), (2, 3)),
            (((1, 1, 3), (1, 1, 1), 1, [0]), (1, 3), (1, 1)),
            (((1, 1, 6), (1, 1, 3.14), 1, [0]), (1, 6), (1, 3.14)),
        ],
        indirect=["crystal_map_input"],
    )
    def test_get_nrows_ncols_step_sizes(
        self, crystal_map_input, desired_shape, desired_step_sizes
    ):
        xmap = CrystalMap(**crystal_map_input)
        nrows, ncols, dy, dx = _get_nrows_ncols_step_sizes(xmap)

        assert (nrows, ncols) == desired_shape
        assert (dy, dx) == desired_step_sizes

    @pytest.mark.parametrize(
        "max_value, decimals, expected_width",
        [(3.14, 2, 5), (3.1415, 2, 5), (3.141592, 6, 9)],
    )
    def test_get_column_width(self, max_value, decimals, expected_width):
        assert _get_column_width(max_value, decimals) == expected_width

    @pytest.mark.parametrize(
        "extra_prop", ["a", ["abc", "iq"], ["scores", "simulation_indices"]]
    )
    def test_extra_prop(self, crystal_map, tmp_path, extra_prop):
        fname = tmp_path / "test_extra_prop.ang"
        desired_arrays = []
        n_points = crystal_map.size
        for i, name in enumerate(extra_prop):
            new_array = np.arange(n_points) * i
            crystal_map.prop[name] = new_array
            desired_arrays.append(new_array)

        save(fname, crystal_map, extra_prop=extra_prop)
        xmap_reload = load(fname)

        for name in extra_prop:
            assert np.allclose(xmap_reload.prop[name], crystal_map.prop[name])

    def test_non_indexed_points(self, crystal_map, tmp_path):
        crystal_map[2].phase_id = -1
        fname = tmp_path / "test_non_indexed_points.ang"
        save(fname, crystal_map)

        xmap_reload = load(fname)

        crystal_map.phases[0].name = "phase1"
        assert xmap_reload.phases.names == crystal_map.phases.names

        new_phase_ids = xmap_reload.phase_id
        new_phase_ids[xmap_reload.is_indexed] -= 1
        assert np.allclose(new_phase_ids, crystal_map.phase_id)

    def test_points_not_in_data(self, crystal_map, tmp_path):
        crystal_map.prop["iq"] = np.ones(crystal_map.size)
        crystal_map[2].iq = 0
        xmap2 = crystal_map[crystal_map.iq == 1]
        fname = tmp_path / "test_points_not_in_data.ang"
        save(fname, xmap2)

        xmap_reload = load(fname)
        assert not xmap2.is_in_data.all()
        assert np.allclose(xmap2.is_in_data, xmap_reload.is_indexed)

    @pytest.mark.parametrize(
        "extra_phase_names", ["ni", ["ferrite", "austenite"], ["si", "al", "cu"]],
    )
    def test_extra_phases(self, crystal_map, tmp_path, extra_phase_names):
        crystal_map.phases.add_not_indexed()
        for i, name in enumerate(extra_phase_names):
            crystal_map.phases.add(Phase(name=name))
            crystal_map[i].phase_id = crystal_map.phases.id_from_name(name)
        fname = tmp_path / "test_extra_phases.ang"
        save(fname, crystal_map)
        xmap_reload = load(fname)

        crystal_map.phases[0].name = "phase1"
        assert np.allclose(xmap_reload.phase_id - 1, crystal_map.phase_id)

        pl = crystal_map.phases
        del pl[-1]
        assert xmap_reload.phases.names == pl.names

    @pytest.mark.parametrize("point_group", ["432", "121", "222"])
    def test_point_group_aliases(self, crystal_map, tmp_path, point_group):
        crystal_map.phases[0].point_group = point_group
        fname = tmp_path / "test_point_group_aliases.ang"
        save(fname, crystal_map)
        xmap_reload = load(fname)

        phase_ids = xmap_reload.phases.ids
        assert xmap_reload.phases[phase_ids[0]].point_group.name == point_group

    @pytest.mark.parametrize(
        "crystal_map_input",
        [((1, 1, 5), (1, 1, 2), 1, [0])],
        indirect=["crystal_map_input"],
    )
    def test_1d_map(self, crystal_map_input, tmp_path):
        xmap = CrystalMap(**crystal_map_input)
        assert xmap.ndim == 1
        fname = tmp_path / "test_1d_map.ang"
        save(fname, xmap)

        xmap_reload = load(fname)

        assert xmap_reload.ndim == 1
        assert np.allclose(xmap.rotations.to_euler(), xmap_reload.rotations.to_euler())

    @pytest.mark.parametrize(
        "crystal_map_input",
        [((3, 3, 3), (1, 2, 3), 1, [0])],
        indirect=["crystal_map_input"],
    )
    def test_3d_map_raises(self, crystal_map_input, tmp_path):
        xmap = CrystalMap(**crystal_map_input)
        fname = tmp_path / "test_3d_raises.ang"
        with pytest.raises(ValueError, match="Writing a 3D dataset to an .ang file"):
            save(fname, xmap)

    @pytest.mark.parametrize(
        "crystal_map_input, index",
        [
            (((1, 4, 3), (1, 2, 3), 5, [0]), 0),
            (((1, 4, 3), (1, 2, 3), 5, [0]), 1),
            (((1, 4, 3), (1, 2, 3), 5, [0]), 4),
        ],
        indirect=["crystal_map_input"],
    )
    def test_write_data_layer_i(self, crystal_map_input, tmp_path, index):
        xmap = CrystalMap(**crystal_map_input)
        xmap.prop["ci"] = np.arange(xmap.size * xmap.rotations_per_point).reshape(
            (xmap.size, xmap.rotations_per_point)
        )
        xmap.prop["iq"] = np.arange(xmap.size)
        extra_prop = "iq_times_ci"
        xmap.prop[extra_prop] = xmap.ci * xmap.iq[:, np.newaxis]
        fname = tmp_path / "test_write_data_layer_i.ang"
        save(fname, xmap, index=index, extra_prop=[extra_prop, "iq"])

        xmap_reload = load(fname)

        assert np.allclose(
            xmap.rotations[:, index].to_euler(), xmap_reload.rotations.to_euler()
        )
        assert np.allclose(xmap_reload.iq, np.zeros(xmap.size))
        assert np.allclose(xmap_reload.ci, xmap.ci[:, index])
        assert np.allclose(xmap_reload.iq_times_ci, xmap.iq_times_ci[:, index])
