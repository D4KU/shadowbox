#include <thread>
#include <cmath>
#include <future>
#include <iostream>
#include <igl/marching_cubes.h>
#include <igl/voxel_grid.h>
#include <mutex>
#include <npe.h>

std::mutex mutex;

float fracf(float a)
{
    float intpart;
    return std::modf(a, &intpart);
}

void create_mesh_slice(
    const Eigen::MatrixXf& xtex,
    const Eigen::MatrixXf& ytex,
    const Eigen::MatrixXf& ztex,
    const int i,
    const int cores,
    const float isolvl,
    Eigen::MatrixXf* allverts,
    Eigen::MatrixXi* allfaces)
{
    const int xres = ytex.cols();
    const float slice = xres / (float)cores;
    const int xstart = std::floor(i * slice);
    const int xend = std::min((int)std::floor((i + 1) * slice) + 1, xres);

    const int xres_tex = xend - xstart;
    const int yres_tex = xtex.cols();
    const int zres_tex = xtex.rows();

    const bool first = i == 0;
    const bool last = i == cores - 1;
    Eigen::Vector3f minpt(xstart - (first * .5f) + .25f, -.5f, -.5f);
    Eigen::Vector3f maxpt(xend   + (last  * .5f) - .25f, yres_tex + .5f, zres_tex + .5f);
    const Eigen::Vector3f diff = maxpt - minpt;

    if (fracf(diff.x()) < .001f)
    {
        // When left or right is true:
        // 1,0 ... left is true, right is false
        //   X ... marks position of i/2
        //
        // i  i/2    i==0    i==1    i==2    i==3    i==4
        // ----------------------------------------------
        // 2  1      1,0     0X1
        // 3  1.5    1,0     0,0  X  0,1
        // 4  2      1,0     1,0     0X1     0,1
        // 5  2.5    1,0     1,0     0,0  X  0,1     0,1

        const bool left = (i + 1) <= cores * .5f;
        const bool right = i >= cores * .5f;
        const Eigen::Vector3f offset((right - left) * .25f, 0, 0);
        minpt += offset;
        maxpt += offset;
    }

    // construct voxel grid
    Eigen::MatrixXf grid;
    Eigen::RowVector3i gridres;
    const int maxsize = diff.maxCoeff();
    const int div = maxsize - 1;
    const Eigen::AlignedBox3f box(minpt / div, maxpt / div);
    igl::voxel_grid(box, maxsize + 1, 0, grid, gridres);

    const int xres_grid = gridres(0);
    const int yres_grid = gridres(1);
    const int zres_grid = gridres(2);
    const int xyres_grid = xres_grid * yres_grid;

    // Eigen::IOFormat f(4, 0, " ", " ", " ", " ", " ", " ");
    // std::cout << "row0 " << grid.row(0).format(f) << std::endl;
    // std::cout << "xstart/end " << xstart << " " << xend << std::endl;
    // std::cout << "minpt " << minpt.format(f) << std::endl;
    // std::cout << "maxpt " << maxpt.format(f) << std::endl;
    // std::cout << "diff " << diff.format(f) << std::endl;
    // std::cout << "gridres   " << xres_grid << "   " << yres_grid << "   " << zres_grid << std::endl;
    // std::cout << std::endl;

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
    // verts and faces are laid out like this:
    // [x, y, z],
    // [x, y, z],
    // ...
    Eigen::MatrixXf verts;
    Eigen::MatrixXi faces;
    igl::marching_cubes(vol, grid, xres_grid, yres_grid, zres_grid, isolvl, verts, faces);

    {
        std::lock_guard<std::mutex> lock(mutex);

        const int vertoff = allverts->rows();
        if (vertoff > 0)
            faces.array() += vertoff;

        allverts->conservativeResize(allverts->rows() + verts.rows(), 3);
        allfaces->conservativeResize(allfaces->rows() + faces.rows(), 3);
        allverts->bottomRows(verts.rows()) = std::move(verts);
        allfaces->bottomRows(faces.rows()) = std::move(faces);
    }
}

npe_function(create_mesh)
npe_arg(xtex, dense_float)
npe_arg(ytex, npe_matches(xtex))
npe_arg(ztex, npe_matches(xtex))
npe_arg(iso, float)

npe_begin_code()
{
    Eigen::MatrixXf verts;
    Eigen::MatrixXi faces;
    auto futures = std::vector<std::future<void>>();
    const int cores = std::max(std::thread::hardware_concurrency(), 1u);

    for (int i = 0; i < cores; ++i)
        futures.push_back(std::async(
            &create_mesh_slice,
            xtex,
            ytex,
            ztex,
            i,
            cores,
            iso,
            &verts,
            &faces));

    for (size_t i = 0; i < futures.size(); ++i)
        futures[i].get();

    // std::cout << "----------------------------------" << std::endl;
    return std::make_tuple(npe::move(verts), npe::move(faces));
}
npe_end_code()
