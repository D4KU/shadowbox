#include "openvdb/tools/Prune.h"
#include <cmath>
#include <iostream>
#include <openvdb/Types.h>
#include <openvdb/io/io.h>
#include <vector>

#include <npe.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/VolumeToMesh.h>

npe_function(create_mesh)
npe_arg(xtex, dense_float)
npe_arg(ytex, dense_float)
npe_arg(ztex, dense_float)
npe_arg(iso, double)
npe_arg(adaptivity, double)

npe_begin_code()
{
    const int xres = ytex.cols();
    const int yres = xtex.cols();
    const int zres = xtex.rows();
    const float maxres = std::max(std::max(xres, yres), zres);

    openvdb::initialize();
    std::vector<openvdb::Vec3s> points;
    std::vector<openvdb::Vec3I> tris;
    std::vector<openvdb::Vec4I> quads;

    {
        auto grid = *openvdb::BoolGrid::create();
        grid.setGridClass(openvdb::GRID_FOG_VOLUME);
        auto accessor = grid.getAccessor();

        for (int x = 0; x < xres; ++x)
            for (int y = 0; y < yres; ++y)
                if (ztex(y, x) >= iso)
                    for (int z = 0; z < zres; ++z)
                        if (xtex(z, y) >= iso && ytex(z, x) >= iso)
                            accessor.setValue(openvdb::Coord(x, y, z), true);

        grid.tree().prune();
        openvdb::tools::volumeToMesh(grid, points, tris, quads, iso, adaptivity);
    }

    Eigen::MatrixX3f verts(points.size(), 3);
    Eigen::MatrixX4i faces(tris.size() + quads.size(), 4);

    for (int i = 0; i < points.size(); ++i)
    {
        verts(i, 0) = points[i][0] / maxres;
        verts(i, 1) = points[i][1] / maxres;
        verts(i, 2) = points[i][2] / maxres;
    }

    for (int i = 0; i < quads.size(); ++i)
    {
        faces(i, 0) = quads[i][0];
        faces(i, 1) = quads[i][1];
        faces(i, 2) = quads[i][2];
        faces(i, 3) = quads[i][3];
    }

    for (int i = 0; i < tris.size(); ++i)
    {
        const int j = i + quads.size();
        faces(j, 0) = tris[i][0];
        faces(j, 1) = tris[i][1];
        faces(j, 2) = tris[i][2];
        faces(j, 3) = tris[i][2];
    }

    return std::make_tuple(npe::move(verts), npe::move(faces));
}
npe_end_code()
