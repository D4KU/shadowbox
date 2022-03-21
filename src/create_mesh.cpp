#include <iostream>
#include <igl/marching_cubes.h>
#include <igl/voxel_grid.h>
#include <npe.h>

npe_function(create_mesh)
npe_arg(xtex, dense_float)
npe_arg(ytex, npe_matches(xtex))
npe_arg(ztex, npe_matches(xtex))
npe_arg(grid, dense_float)
npe_arg(gridres, dense_int)

npe_begin_code()
{
    int xres = gridres(0, 0);
    int yres = gridres(1, 0);
    int zres = gridres(2, 0);

    int xsize = ytex.cols();
    int ysize = xtex.cols();
    int zsize = xtex.rows();

    // set voxel values
    Eigen::VectorXf S(gridres.prod());
    for (int x = 0; x < xsize; ++x)
    {
        for (int y = 0; y < ysize; ++y)
        {
            for (int z = 0; z < zsize; ++z)
            {
                // Imagine 2x2x2 cube with these entries:
                // front: [c d] back: [g h]
                //        [a b]       [e f]
                // Then S is laid out like this:
                // [a b][c d][e f][g h]
                S(x + y * xres + z * xres * yres) =
                    xtex(z, y) *
                    ytex(z, x) *
                    ztex(y, x);
            }
        }
    }

    // construct mesh
    Eigen::MatrixXf verts;
    Eigen::MatrixXi faces;
    igl::marching_cubes(S, grid, xres, yres, zres, 0, verts, faces);

    return std::make_tuple(npe::move(verts), npe::move(faces));
}
npe_end_code()
