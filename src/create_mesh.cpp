#include <cmath>
#include <iostream>
#include <igl/marching_cubes.h>
#include <igl/voxel_grid.h>
#include <npe.h>

float fracf(float a)
{
    float intpart;
    return std::modf(a, &intpart);
}

auto create_mesh_impl(
    const Eigen::MatrixXf& xtex,
    const Eigen::MatrixXf& ytex,
    const Eigen::MatrixXf& ztex,
    const int xstart,
    const int xend,
    const int i,
    const int cores,
    const float isolvl,
    Eigen::MatrixXf& verts,
    Eigen::MatrixXi& faces)
{
    const int xres_tex = xend - xstart;
    const int yres_tex = xtex.cols();
    const int zres_tex = xtex.rows();

    const bool first = i == 0;
    const bool last = i == cores - 1;
    Eigen::Vector3f minpt(xstart - (first * .5f) + .25f, -.5f, -.5f);
    Eigen::Vector3f maxpt(xend   + (last  * .5f) - .25f, yres_tex + .5f, zres_tex + .5f);
    const Eigen::Vector3f diff = maxpt - minpt;
    const int maxsize = std::floor(diff.maxCoeff()) + 1;

    if (fracf(diff.x()) < .001f)
    {
        const bool left = (i + 1) <= cores * .5f;
        const bool right = i >= cores * .5f;
        const Eigen::Vector3f offset((right - left) * .25f, 0, 0);
        minpt += offset;
        maxpt += offset;
    }

    // construct voxel grid
    Eigen::MatrixXf grid;
    Eigen::RowVector3i gridres;
    const Eigen::AlignedBox3f box(minpt, maxpt);
    igl::voxel_grid(box, maxsize, 0, grid, gridres);

    const int xres_grid = gridres(0);
    const int yres_grid = gridres(1);
    const int zres_grid = gridres(2);
    const int xyres_grid = xres_grid * yres_grid;

    Eigen::IOFormat f(4, 0, " ", " ", " ", " ", " ", " ");
    std::cout << "row0 " << grid.row(0).format(f) << std::endl;
    std::cout << "xstart/end " << xstart << " " << xend << std::endl;
    std::cout << "minpt " << minpt.format(f) << std::endl;
    std::cout << "maxpt " << maxpt.format(f) << std::endl;
    std::cout << "diff " << diff.format(f) << std::endl;
    std::cout << "gridres   " << xres_grid << "   " << yres_grid << "   " << zres_grid << std::endl;
    std::cout << std::endl;

    // set voxel values
    Eigen::VectorXf vol(gridres.prod());
    vol.setZero();
    for (int x = 0; x < xres_tex; ++x)
    {
        const int x2 = x + xstart;
        for (int y = 0; y < yres_tex; ++y)
        {
            const float zpx = ztex(y, x2);
            for (int z = 0; z < zres_tex; ++z)
            {
                // Imagine 2x2x2 cube with these entries:
                // front: [c d] back: [g h]
                //        [a b]       [e f]
                // Then S is laid out like this:
                // [a b][c d][e f][g h]
                vol(x + first + (y + 1) * xres_grid + (z + 1) * xyres_grid)
                    = xtex(z, y) * ytex(z, x2) * zpx;
            }
        }
    }

    // construct mesh
    igl::marching_cubes(vol, grid, xres_grid, yres_grid, zres_grid, isolvl, verts, faces);
}

npe_function(create_mesh)
npe_arg(xtex, dense_float)
npe_arg(ytex, npe_matches(xtex))
npe_arg(ztex, npe_matches(xtex))
npe_arg(iso, float)
npe_arg(cores, int)

npe_begin_code()
{
    const int xres = ytex.cols();
    const int yres = xtex.cols();
    const int zres = xtex.rows();

    const float slice = xres / (float)cores;
    int vertoff = 0;

    Eigen::MatrixXf allverts;
    Eigen::MatrixXi allfaces;
    for (int i = 0; i < cores; ++i)
    {
        Eigen::MatrixXf verts;
        Eigen::MatrixXi faces;
        create_mesh_impl(
            xtex, ytex, ztex,
            std::floor(i * slice),
            std::min((int)std::floor((i + 1) * slice) + 1, xres),
            i,
            cores,
            iso,
            verts, faces);

        // i i/2    i==0    i==1    i==2    i==3    i==4
        // ---------------------------------------------
        // 2 1      1,0     0X1
        // 3 1.5    1,0     0,0  X  0,1
        // 4 2      1,0     1,0     0X1     0,1
        // 5 2.5    1,0     1,0     0,0  X  0,1     0,1

        // verts and faces are laid out like this:
        // [x, y, z],
        // [x, y, z],
        // ...

        if (vertoff > 0)
            faces.array() += vertoff;
        vertoff += verts.rows();

        allverts.conservativeResize(allverts.rows() + verts.rows(), 3);
        allfaces.conservativeResize(allfaces.rows() + faces.rows(), 3);
        allverts.bottomRows(verts.rows()) = std::move(verts);
        allfaces.bottomRows(faces.rows()) = std::move(faces);

        // return std::make_tuple(npe::move(verts), npe::move(faces));
    }

    std::cout << "----------------------------------" << std::endl;
    return std::make_tuple(npe::move(allverts), npe::move(allfaces));
}
npe_end_code()
