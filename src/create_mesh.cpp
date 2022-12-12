#include <cmath>
#include <future>
#include <iostream>
#include <thread>
#include <tuple>
#include <vector>

#include <Eigen/Dense>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <Python.h>

#include <openvdb/openvdb.h>
#include <openvdb/Grid.h>
#include <openvdb/Types.h>
#include <openvdb/io/io.h>
#include <openvdb/tools/VolumeToMesh.h>
#include <openvdb/tools/Prune.h>

namespace vdb = openvdb;

vdb::BoolGrid::Ptr get_vol_slice(
    const Eigen::MatrixXf& xtex,
    const Eigen::MatrixXf& ytex,
    const Eigen::MatrixXf& ztex,
    const int sliceIdx,
    const int sliceCount,
    const float iso)
{
    const int xres = (int)ytex.cols();
    const int yres = (int)xtex.cols();
    const int zres = (int)xtex.rows();

    const float sliceRes = xres / (float)sliceCount;
    const int xstart = (int)(sliceIdx * sliceRes);
    const int xend = std::min((int)((sliceIdx + 1) * sliceRes), xres);

    auto grid = vdb::BoolGrid::create();
    grid->setGridClass(vdb::GRID_FOG_VOLUME);
    auto accessor = grid->getAccessor();

    for (int x = xstart; x < xend; ++x)
        for (int y = 0; y < yres; ++y)
            if (ztex(y, x) >= iso)
                for (int z = 0; z < zres; ++z)
                    if (xtex(z, y) >= iso && ytex(z, x) >= iso)
                        accessor.setValue(vdb::Coord(x, y, z), true);

    grid->tree().prune();
    return grid;
}

std::pair<Eigen::MatrixX3f, Eigen::MatrixX4i> create_mesh(
    const Eigen::Ref<const Eigen::MatrixXf>& xtex,
    const Eigen::Ref<const Eigen::MatrixXf>& ytex,
    const Eigen::Ref<const Eigen::MatrixXf>& ztex,
    const float iso,
    const float adaptivity)
{
    vdb::initialize();
    std::vector<vdb::Vec3s> points;
    std::vector<vdb::Vec3I> tris;
    std::vector<vdb::Vec4I> quads;

    const int cores = std::max(std::thread::hardware_concurrency(), 1u);
    auto futures = new std::future<vdb::BoolGrid::Ptr>[cores];

    for (int i = 0; i < cores; ++i)
        futures[i] = std::async(&get_vol_slice, xtex, ytex, ztex, i, cores, iso);

    auto grid = futures[0].get();
    for (int i = 1; i < cores; ++i)
        grid->merge(*futures[i].get());

    grid->tree().prune();
    vdb::tools::volumeToMesh(*grid, points, tris, quads, .5, adaptivity);
    Eigen::MatrixX3f verts(points.size(), 3);
    Eigen::MatrixX4i faces(tris.size() + quads.size(), 4);

    const size_t xres = ytex.cols();
    const size_t yres = xtex.cols();
    const size_t zres = xtex.rows();
    const float res = (float)std::max(std::max(xres, yres), zres);

    for (size_t i = 0; i < points.size(); ++i)
    {
        verts(i, 0) = points[i][0] / res;
        verts(i, 1) = points[i][1] / res;
        verts(i, 2) = points[i][2] / res;
    }

    for (size_t i = 0; i < quads.size(); ++i)
    {
        faces(i, 0) = quads[i][0];
        faces(i, 1) = quads[i][1];
        faces(i, 2) = quads[i][2];
        faces(i, 3) = quads[i][3];
    }

    for (size_t i = 0; i < tris.size(); ++i)
    {
        const size_t j = i + quads.size();
        faces(j, 0) = tris[i][0];
        faces(j, 1) = tris[i][1];
        faces(j, 2) = tris[i][2];
        faces(j, 3) = tris[i][2];
    }

    return std::make_pair(std::move(verts), std::move(faces));
}

PYBIND11_MODULE(core, m)
{
    m.def(
        "create_mesh",
        &create_mesh,
        pybind11::arg("xtex").noconvert(),
        pybind11::arg("ytex").noconvert(),
        pybind11::arg("ztex").noconvert(),
        pybind11::arg("iso"),
        pybind11::arg("adaptivity"));
}
