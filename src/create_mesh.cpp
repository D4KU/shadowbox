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
    const int gridresx = gridres(0, 0);
    const int gridresy = gridres(1, 0);
    const int gridresz = gridres(2, 0);

    const int texresx = ytex.cols();
    const int texresy = xtex.cols();
    const int texresz = xtex.rows();

    const int kernx = texresx / gridresx;
    const int kerny = texresy / gridresy;
    const int kernz = texresz / gridresz;

    // set voxel values
    Eigen::VectorXf S(gridres.prod());

    for (int gx = 0; gx < gridresx; ++gx)
    {
        for (int gy = 0; gy < gridresy; ++gy)
        {
            for (int gz = 0; gz < gridresz; ++gz)
            {
                float sum = 0.f;
                for (int kx = 0; kx < kernx; ++kx)
                {
                    const int x = gx * kernx + kx;
                    for (int ky = 0; ky < kerny; ++ky)
                    {
                        const int y = gy * kerny + ky;
                        for (int kz = 0; kz < kernz; ++kz)
                        {
                            const int z = gz * kernz + kz;
                            sum = xtex(z, y) * ytex(z, x) * ztex(y, x);
                        }
                    }
                }
                sum /= kernx * kerny * kernz;

                // Imagine 2x2x2 cube with these entries:
                // front: [c d] back: [g h]
                //        [a b]       [e f]
                // Then S is laid out like this:
                // [a b][c d][e f][g h]
                S(gx + gy * gridresx + gz * gridresx * gridresy) = sum;
            }
        }
    }

    // construct mesh
    Eigen::MatrixXf verts;
    Eigen::MatrixXi faces;
    igl::marching_cubes(S, grid, gridresx, gridresy, gridresz, 0, verts, faces);

    return std::make_tuple(npe::move(verts), npe::move(faces));
}
npe_end_code()
