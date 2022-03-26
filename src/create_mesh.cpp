#include <cmath>
#include <iostream>
#include <igl/marching_cubes.h>
#include <igl/voxel_grid.h>
#include <npe.h>
#include <math.h>

float sample(
    Eigen::MatrixXf xtex,
    Eigen::MatrixXf ytex,
    Eigen::MatrixXf ztex,
    int x, int y, int z)
{
    return xtex(z, y) * ytex(z, x) * ztex(y, x);
}

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

    const float kernx = (texresx - 1) / (float)(gridresx - 1);
    const float kerny = (texresy - 1) / (float)(gridresy - 1);
    const float kernz = (texresz - 1) / (float)(gridresz - 1);

    // set voxel values
    Eigen::VectorXf S(gridres.prod());

    for (int gx = 0; gx < gridresx; ++gx)
    {
        const int lowx = std::floor(kernx * gx);
        const int highx = std::ceil(kernx * gx);
        const float wx = kernx * gx - lowx;

        for (int gy = 0; gy < gridresy; ++gy)
        {
            const int lowy = std::floor(kerny * gy);
            const int highy = std::ceil(kerny * gy);
            const float wy = kerny * gy - lowy;

            for (int gz = 0; gz < gridresz; ++gz)
            {
                const int lowz = std::floor(kernz * gz);
                const int highz = std::ceil(kernz * gz);
                const float wz = kernz * gz - lowz;

                const float lll = sample(xtex, ytex, ztex, lowx,  lowy,  lowz);
                const float llh = sample(xtex, ytex, ztex, lowx,  lowy,  highz);
                const float lhl = sample(xtex, ytex, ztex, lowx,  highy, lowz);
                const float lhh = sample(xtex, ytex, ztex, lowx,  highy, highz);
                const float hll = sample(xtex, ytex, ztex, highx, lowy,  lowz);
                const float hlh = sample(xtex, ytex, ztex, highx, lowy,  highz);
                const float hhl = sample(xtex, ytex, ztex, highx, highy, lowz);
                const float hhh = sample(xtex, ytex, ztex, highx, highy, highz);

                const float Wx = 1 - wx;
                const float Wy = 1 - wy;
                const float Wz = 1 - wz;

                // Imagine 2x2x2 cube with these entries:
                // front: [c d] back: [g h]
                //        [a b]       [e f]
                // Then S is laid out like this:
                // [a b][c d][e f][g h]
                S(gx + gy * gridresx + gz * gridresx * gridresy) =
                    lll * Wx * Wy * Wz +
                    llh * Wx * Wy * wz +
                    lhl * Wx * wy * Wz +
                    lhh * Wx * wy * wz +
                    hll * wx * Wy * Wz +
                    hlh * wx * Wy * wz +
                    hhl * wx * wy * Wz +
                    hhh * wx * wy * wz;
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
