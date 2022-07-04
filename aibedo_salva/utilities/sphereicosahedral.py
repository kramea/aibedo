# -*- coding: utf-8 -*-

import numpy as np
from pygsp.graphs import NNGraph  # prevent circular import in Python < 3.5


def _import_trimesh():
    try:
        import trimesh
    except Exception as e:
        raise ImportError('Cannot import trimesh. Choose another graph '
                          'or try to install it with '
                          'conda or pip install trimesh. '
                          'Original exception: {}'.format(e))
    return trimesh


def xyz2latlon(x, y, z):
    r"""
    Taken from PyGSP:

    Convert 3D spherical coordinates to latitude and longitude.

    Parameters
    ----------
    x, y, z : array_like
        3D coordinates.

    Returns
    -------
    lat : :class:`numpy.ndarray`
        Latitude in [-π/2, π/2].
    lon : :class:`numpy.ndarray`
        Longitude in [0, 2π[.

    See Also
    --------
    latlon2xyz : inverse transformation

    Examples
    --------
    >>> utils.xyz2latlon(1, 0, 0)
    (0.0, 0.0)
    >>> utils.xyz2latlon(0, 1, 0)
    (0.0, 1.5707963267948966)
    >>> utils.xyz2latlon(0, 0, 1)
    (1.5707963267948966, 0.0)

    """
    lon = np.arctan2(y, x)
    lon += (lon < 0) * 2 * np.pi  # signed [-π,π] to unsigned [0,2π[
    lon[lon == 2 * np.pi] = 0  # 2*np.pi-x == 2*np.pi if x < np.spacing(2*np.pi)
    lat = np.arctan2(z, np.sqrt(x ** 2 + y ** 2))
    return lat, lon


class SphereIcosahedral(NNGraph):
    r"""Sphere sampled as a subdivided icosahedron.

    Background information is found at :doc:`/background/spherical_samplings`.

    Parameters
    ----------
    subdivisions : int
        Number of edges the icosahedron's edges are divided into, resulting in
        ``10*subdivisions**2+2`` vertices (or ``20*subdivisions**2`` if
        ``dual=True``).
        It must be a power of 2 in the current implementation.
    dual : bool
        Whether the graph vertices correspond to the vertices (``dual=False``)
        or the triangular faces (``dual=True``) of the subdivided icosahedron.
    kwargs : dict
        Additional keyword parameters are passed to :class:`NNGraph`.

    Attributes
    ----------
    signals : dict
        Vertex position as latitude ``'lat'`` in [-π/2,π/2] and longitude
        ``'lon'`` in [0,2π[.

    See Also
    --------
    SphereEquiangular, SphereGaussLegendre : based on quadrature theorems
    SphereCubed, SphereHealpix : based on subdivided polyhedra
    SphereRandom : random uniform sampling

    Notes
    -----
    Edge weights are computed by :class:`NNGraph`. Gaussian kernel widths have
    however not been optimized for convolutions on the resulting graph to be
    maximally equivariant to rotation [4]_.

    References
    ----------
    .. [1] J. R. Baumgardner, P. O. Frederickson, Icosahedral discretization of
       the two-sphere, 1985.
    .. [2] M. Tegmark, An icosahedron-based method for pixelizing the celestial
       sphere, 1996.
    .. [3] https://sinestesia.co/blog/tutorials/python-icospheres/
    .. [4] M. Defferrard et al., DeepSphere: a graph-based spherical CNN, 2019.

    Examples
    --------

    """

    def __init__(self, subdivisions=2, dual=False, **kwargs):
        self.subdivisions = subdivisions
        self.dual = dual

        # Vertices as the corners of three orthogonal golden planes.
        φ = (1 + 5 ** 0.5) / 2  # scipy.constants.golden_ratio
        vertices = np.array([
            [-1, φ, 0], [1, φ, 0], [-1, -φ, 0], [1, -φ, 0],
            [0, -1, φ], [0, 1, φ], [0, -1, -φ], [0, 1, -φ],
            [φ, 0, -1], [φ, 0, 1], [-φ, 0, -1], [-φ, 0, 1],
        ]) / np.sqrt(φ ** 2 + 1)
        faces = np.array([
            # Faces around vertex 0.
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            # Adjacent faces.
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            # Faces around vertex 3.
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            # Adjacent faces.
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ])

        trimesh = _import_trimesh()
        mesh = trimesh.Trimesh(vertices, faces)

        def normalize(vertices):
            """Project the vertices on the sphere."""
            vertices /= np.linalg.norm(vertices, axis=1)[:, None]

        if np.log2(subdivisions) % 1 != 0:
            raise NotImplementedError('Only recursive subdivisions by two are '
                                      'implemented. Choose a power of two.')

        for _ in range(int(np.log2(subdivisions))):
            mesh = mesh.subdivide()
            # TODO: shall we project between subdivisions? Some do, some don't.
            # Projecting pushes points away from the 12 base vertices, which
            # may make the point density more uniform.
            # See "A Comparison of Popular Point Configurations on S^2".
            # As the equiangular vs equidistant spacing on the subdivided cube.
            normalize(mesh.vertices)

        if not dual:
            vertices = mesh.vertices
        else:
            vertices = mesh.vertices[mesh.faces].mean(axis=1)
            normalize(vertices)

        super(SphereIcosahedral, self).__init__(vertices, **kwargs)
        # lat, lon = xyz2latlon(*vertices.T)
        # self.signals['lat'] = lat
        # self.signals['lon'] = lon

    def _get_extra_repr(self):
        attrs = {
            'subdivisions': self.subdivisions,
            'dual': self.dual,
        }
        attrs.update(super(SphereIcosahedral, self)._get_extra_repr())
        return attrs
