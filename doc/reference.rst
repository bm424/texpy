=============
API reference
=============

This reference manual details the public modules, classes, and functions in orix, as
generated from their docstrings. Many of the docstrings contain examples, however, see
the :ref:`user guide <user-guide>` and the
`demos <https://github.com/pyxem/orix-demos>`_ for how to use orix.

.. caution::

    orix is in an alpha stage, so there will be breaking changes with each release.

.. module:: orix

The list of top modules:

.. autosummary::
    base
    crystal_map
    io
    plot
    projections
    quaternion
    sampling
    scalar
    vector

....

base
====
.. automodule:: orix.base
    :members:
    :show-inheritance:

....

crystal_map
===========
.. automodule:: orix.crystal_map
.. currentmodule:: orix.crystal_map
.. autosummary::
    CrystalMap
    CrystalMapProperties
    Phase
    PhaseList
    create_coordinate_arrays

Phase
-----
.. currentmodule:: orix.crystal_map.Phase
.. autosummary::
    deepcopy
.. autoclass:: orix.crystal_map.Phase
    :members:
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__

PhaseList
---------
.. currentmodule:: orix.crystal_map.PhaseList
.. autosummary::
    add
    deepcopy
    id_from_name
    sort_by_id
.. autoclass:: orix.crystal_map.PhaseList
    :members:
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__

CrystalMap
----------
.. currentmodule:: orix.crystal_map.CrystalMap
.. autosummary::
    deepcopy
    empty
    get_map_data
    plot
.. autoclass:: orix.crystal_map.CrystalMap
    :members:
    :undoc-members:
    :show-inheritance:

    .. automethod:: __init__

CrystalMapProperties
--------------------
.. currentmodule:: orix.crystal_map.CrystalMapProperties
.. autoclass:: orix.crystal_map.CrystalMapProperties
    :members:
    :undoc-members:
    :show-inheritance:

Other functions
---------------

.. currentmodule:: orix.crystal_map
.. autofunction:: create_coordinate_arrays

....

io
===
.. currentmodule:: orix.io
.. autosummary::
    load
    loadang
    loadctf
    plugins
    save

.. automodule:: orix.io
    :members:

plugins
-------
.. automodule:: orix.io.plugins
.. currentmodule:: orix.io.plugins
.. autosummary::
    ang
    emsoft_h5ebsd
    orix_hdf5

ang
~~~
.. automodule:: orix.io.plugins.ang
    :members:
    :undoc-members:
    :show-inheritance:

emsoft_h5ebsd
~~~~~~~~~~~~~
.. automodule:: orix.io.plugins.emsoft_h5ebsd
    :members:
    :undoc-members:
    :show-inheritance:

orix_hdf5
~~~~~~~~~
.. automodule:: orix.io.plugins.orix_hdf5
    :members:
    :undoc-members:
    :show-inheritance:

....

plot
====
.. automodule:: orix.plot
.. currentmodule:: orix.plot
.. autosummary::
    AxAnglePlot
    CrystalMapPlot
    RodriguesPlot
    RotationPlot
    StereographicPlot

AxAnglePlot
-----------
.. currentmodule:: orix.plot.AxAnglePlot
.. autoclass:: orix.plot.AxAnglePlot
    :show-inheritance:

CrystalMapPlot
--------------
.. currentmodule:: orix.plot.CrystalMapPlot
.. autosummary::
    add_colorbar
    add_overlay
    add_scalebar
    plot_map
    remove_padding
.. autoclass:: orix.plot.CrystalMapPlot
    :members:
    :undoc-members:
    :show-inheritance:

RodriguesPlot
-------------
.. currentmodule:: orix.plot.RodriguesPlot
.. autoclass:: orix.plot.RodriguesPlot
    :show-inheritance:

RotationPlot
------------
.. currentmodule:: orix.plot.RotationPlot
.. autoclass:: orix.plot.RotationPlot
    :show-inheritance:

StereographicPlot
-----------------
.. currentmodule:: orix.plot.StereographicPlot
.. autosummary::
    azimuth_grid
    draw_circle
    polar_grid
    scatter
    show_hemisphere_label
    set_labels
    symmetry_marker
    text
.. autoclass:: orix.plot.StereographicPlot
    :members: azimuth_grid, draw_circle, hemisphere, name, polar_grid, pole, scatter, set_labels, show_hemisphere_label, symmetry_marker, text
    :undoc-members:
    :show-inheritance:

....

projections
===========
.. automodule:: orix.projections
.. currentmodule:: orix.projections
.. autosummary::
    StereographicProjection
    InverseStereographicProjection

StereographicProjection
-----------------------
.. currentmodule:: orix.projections.StereographicProjection
.. autosummary::
    inverse
    spherical2xy
    spherical2xy_split
    vector2xy
    vector2xy_split
.. autoclass:: orix.projections.StereographicProjection
    :members:
    :undoc-members:

    .. automethod:: __init__

InverseStereographicProjection
------------------------------
.. currentmodule:: orix.projections.InverseStereographicProjection
.. autosummary::
    xy2spherical
    xy2vector
.. autoclass:: orix.projections.InverseStereographicProjection
    :members:
    :undoc-members:

    .. automethod:: __init__

....

quaternion
==========
.. automodule:: orix.quaternion
.. currentmodule:: orix.quaternion
.. autosummary::
    Orientation
    OrientationRegion
    Misorientation
    Quaternion
    Rotation
    Symmetry

Orientation and Misorientation
------------------------------
.. currentmodule:: orix.quaternion.orientation
.. automodule:: orix.quaternion.orientation

Orientation
~~~~~~~~~~~
.. currentmodule:: orix.quaternion.Orientation
.. autosummary::
    set_symmetry
.. autoclass:: orix.quaternion.Orientation
    :show-inheritance:
    :members:
    :undoc-members:

Misorientation
~~~~~~~~~~~~~~
.. currentmodule:: orix.quaternion.Misorientation
.. autosummary::
    distance
    equivalent
    set_symmetry
.. autoclass:: orix.quaternion.Misorientation
    :show-inheritance:
    :members:
    :undoc-members:

OrientationRegion
-----------------
.. automodule:: orix.quaternion.OrientationRegion
    :noindex:
.. currentmodule:: orix.quaternion.OrientationRegion
.. autosummary::
    from_symmetry
    get_plot_data
    vertices
.. autoclass:: orix.quaternion.OrientationRegion
    :show-inheritance:
    :members:
    :undoc-members:

Quaternion
----------
.. autoclass:: orix.quaternion.Quaternion
    :show-inheritance:
    :members:

Rotation
--------
.. automodule:: orix.quaternion.rotation
.. currentmodule:: orix.quaternion.Rotation
.. autosummary::
    angle_with
    dot_outer
    flatten
    from_euler
    from_matrix
    from_neo_euler
    identity
    outer
    random
    random_vonmises
    to_euler
    to_matrix
    unique
.. autoclass:: orix.quaternion.Rotation
    :show-inheritance:
    :members:
    :undoc-members:

Symmetry
--------
.. automodule:: orix.quaternion.symmetry
.. currentmodule:: orix.quaternion.Symmetry
.. autosummary::
    from_generators
.. autoclass:: orix.quaternion.Symmetry
    :show-inheritance:
    :members:
    :undoc-members:

Other functions

.. currentmodule:: orix.quaternion.symmetry
.. autofunction:: get_distinguished_points
.. autofunction:: get_point_group

....

sampling
========
.. currentmodule:: orix.sampling
.. autosummary::
    get_sample_fundamental
    get_sample_local
    uniform_SO3_sample
.. automodule:: orix.sampling
    :members:
    :show-inheritance:

....

scalar
======
.. automodule:: orix.scalar
    :members:
    :show-inheritance:

....

vector
======
.. automodule:: orix.vector
.. currentmodule:: orix.vector
.. autosummary::
    AxAngle
    Homochoric
    Miller
    NeoEuler
    Rodrigues
    SphericalRegion
    Vector3d

AxAngle
-------
.. currentmodule:: orix.vector.AxAngle
.. autosummary::
    from_axes_angles
.. autoclass:: orix.vector.AxAngle
    :members:
    :undoc-members:
    :show-inheritance:

Homochoric
----------
.. autoclass:: orix.vector.Homochoric
    :members:
    :undoc-members:
    :show-inheritance:

Miller
------
.. currentmodule:: orix.vector.Miller
.. autosummary::
    angle_with
    cross
    dot
    dot_outer
    deepcopy
    draw_circle
    flatten
    from_highest_indices
    from_min_dspacing
    get_circle
    mean
    reshape
    round
    rotate
    scatter
    symmetrise
    unique
.. autoclass:: orix.vector.Miller
    :members:
    :undoc-members:
    :show-inheritance:

NeoEuler
--------
.. currentmodule:: orix.vector.NeoEuler
.. autosummary::
    from_rotation
.. autoclass:: orix.vector.NeoEuler
    :members:
    :undoc-members:
    :show-inheritance:

Rodrigues
---------
.. autoclass:: orix.vector.Rodrigues
    :members:
    :undoc-members:
    :show-inheritance:

SphericalRegion
---------------
.. automodule:: orix.vector.spherical_region
.. autoclass:: orix.vector.SphericalRegion
    :members:
    :undoc-members:
    :show-inheritance:

Vector3d
--------
.. currentmodule:: orix.vector.Vector3d
.. autosummary::
    angle_with
    cross
    dot
    dot_outer
    draw_circle
    from_polar
    get_circle
    get_nearest
    mean
    rotate
    scatter
    to_polar
    xvector
    yvector
    zvector
    zero
.. autoclass:: orix.vector.Vector3d
    :members:
    :undoc-members:
    :show-inheritance:
