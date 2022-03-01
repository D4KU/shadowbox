#include <iostream>
#include <igl/marching_cubes.h>
#include <igl/voxel_grid.h>
#include <npe.h>

npe_function(shadowbox)
npe_arg(xtex, dense_double)
npe_arg(ytex, npe_matches(xtex))
npe_arg(ztex, npe_matches(xtex))

npe_begin_code()
{
    int xsize = ytex.cols();
    int ysize = xtex.cols();
    int zsize = xtex.rows();

    if (xsize != ztex.cols() || ysize != ztex.rows() || zsize != ytex.rows())
    {
        std::stringstream ss;
        ss << "Shape error";
        throw pybind11::value_error(ss.str());
    }

    // construct grid bounding box
    Eigen::Vector3d min(-.5, -.5, -.5);
    Eigen::Vector3d max(.5, .5, .5);
    Eigen::AlignedBox<double, 3> box(min, max);
    const int pad = 0;

    // construct voxel grid
    Eigen::MatrixXd grid;
    Eigen::RowVector3i gridres;
    igl::voxel_grid(box, xsize + (pad * 2), pad, grid, gridres);

    // get voxel values
    Eigen::VectorXd S(gridres.prod());
    for (int x = 0; x < xsize; ++x)
    {
        for (int y = 0; y < ysize; ++y)
        {
            for (int z = 0; z < zsize; ++z)
            {
                const double val = xtex(z, y) * ytex(z, x) * ztex(y, x);
                S(x + y * gridres(0) + z * gridres(0) * gridres(1)) = val;
            }
        }
    }

    const double iso = 0;
    Eigen::MatrixXd V;
    Eigen::MatrixXi F;
    igl::marching_cubes(S, grid, gridres(0), gridres(1), gridres(2), iso, V, F);

    return std::make_tuple(V, F);
}
npe_end_code()
