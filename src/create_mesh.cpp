#include <iostream>
#include <igl/marching_cubes.h>
#include <igl/voxel_grid.h>
#include <npe.h>

npe_function(create_mesh)
npe_arg(xtex, dense_float)
npe_arg(ytex, npe_matches(xtex))
npe_arg(ztex, npe_matches(xtex))
npe_arg(iso, float)

npe_begin_code()
{
    const int xres = ytex.cols();
    const int yres = xtex.cols();
    const int zres = xtex.rows();

    // construct grid bounding box
    Eigen::Vector3f size(xres, yres, zres);
    int maxsize = size.maxCoeff();
    Eigen::AlignedBox3f box(Eigen::Vector3f(0), size / maxsize);

    // construct voxel grid
    Eigen::MatrixXf grid;
    Eigen::RowVector3i gridres;
    igl::voxel_grid(box, maxsize, 0, grid, gridres);

    // set voxel values
    Eigen::VectorXf vol(xres * yres * zres);
    for (int x = 0; x < xres; ++x)
    {
        for (int y = 0; y < yres; ++y)
        {
            const float zpx = ztex(y, x);
            const float xyres = xres * yres;

            for (int z = 0; z < zres; ++z)
            {
                // Imagine 2x2x2 cube with these entries:
                // front: [c d] back: [g h]
                //        [a b]       [e f]
                // Then S is laid out like this:
                // [a b][c d][e f][g h]
                vol(x + y * xres + z * xyres) = xtex(z, y) * ytex(z, x) * zpx;
            }
        }
    }

    // construct mesh
    Eigen::MatrixXf verts;
    Eigen::MatrixXi faces;
    igl::marching_cubes(vol, grid, xres, yres, zres, iso, verts, faces);

    return std::make_tuple(npe::move(verts), npe::move(faces));
}
npe_end_code()
