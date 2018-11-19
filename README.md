BBWADG is based on the approach outlined in https://arxiv.org/abs/1808.08645.

BBWADG is built with Eigen 3.2.8 (most recent stable version as of Feb 16, 2016), which is included in this repository.
To update Eigen, replace header files in include/Eigen with new header files. (http://eigen.tuxfamily.org)

BBWADG depends on the version 0.1 OCCA library (http://libocca.org) for portability between different GPU vendors.

High order visualization can be achieved using GMSH (http://gmsh.info).

These codes have been tested on OSX 10.11.16.

- To compile with a degree 3 nodal basis

> `make N=3 B=0 -j`

- To compile with a degree 3 Bernstein-Bezier basis:

> `make N=3 B=1 -j`

- To run:

> `./main meshes/cube1.msh`
                                                                