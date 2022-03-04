#include <iostream>
#include <igl/marching_cubes.h>
#include <igl/voxel_grid.h>
#include <npe.h>

npe_function(shadowbox)
npe_arg(xtex, dense_float)
npe_arg(ytex, npe_matches(xtex))
npe_arg(ztex, npe_matches(xtex))

npe_begin_code()
{
    int xsize = ytex.cols();
    int ysize = xtex.cols();
    int zsize = xtex.rows();

    Eigen::Vector3f size(xsize, ysize, zsize);
    int maxsize = size.maxCoeff();

    // construct grid bounding box
    Eigen::Vector3f min(0, 0, 0);
    Eigen::AlignedBox3f box(min, size / maxsize);
    const int pad = 0;

    // construct voxel grid
    Eigen::MatrixXf grid;
    Eigen::RowVector3i gridres;
    igl::voxel_grid(box, maxsize, pad, grid, gridres);

    // get voxel values
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
                S(x + y * gridres(0) + z * gridres(0) * gridres(1)) =
                    xtex(z, y) *
                    ytex(z, x) *
                    ztex(y, x);
            }
        }
    }

    const float iso = 0;
    Eigen::MatrixXf V;
    Eigen::MatrixXi F;
    igl::marching_cubes(S, grid, gridres(0), gridres(1), gridres(2), iso, V, F);

    return std::make_tuple(V, F);
}
npe_end_code()
