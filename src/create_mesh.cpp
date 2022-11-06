#include <cmath>
#include <iostream>
#include <vector>

#include <npe.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/VolumeToMesh.h>

npe_function(create_mesh)
npe_arg(xtex, dense_float)
npe_arg(ytex, npe_matches(xtex))
npe_arg(ztex, npe_matches(xtex))
npe_arg(iso, double)

npe_begin_code()
{
    const int xres = ytex.cols();
    const int yres = xtex.cols();
    const int zres = xtex.rows();
    const float maxres = std::max(std::max(xres, yres), zres);

    openvdb::initialize();
    openvdb::FloatGrid::Ptr grid = openvdb::FloatGrid::create();
    openvdb::FloatGrid::Accessor accessor = grid->getAccessor();

    for (int x = 0; x < xres; ++x)
    {
        for (int y = 0; y < yres; ++y)
        {
            const float zpx = ztex(y, x);
            for (int z = 0; z < zres; ++z)
            {
                float val = xtex(z, y) * ytex(z, x) * zpx;
                if (val > 0)
                    accessor.setValue(openvdb::Coord(x, y, z), val);
            }
        }
    }

    std::vector<openvdb::Vec3s> points;
    std::vector<openvdb::Vec4I> quads;
    openvdb::tools::volumeToMesh(*grid, points, quads, iso);

    Eigen::MatrixX3f verts(points.size(), 3);
    Eigen::MatrixX4i faces(quads.size(), 4);

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

    return std::make_tuple(npe::move(verts), npe::move(faces));
}
npe_end_code()
