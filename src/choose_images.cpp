#include <tuple>
#include <igl/voxel_grid.h>
#include <npe.h>
#include <npe_utils.h>

npe_function(choose_images)
npe_arg(xtex, dense_float)
npe_arg(ytex, npe_matches(xtex))
npe_arg(ztex, npe_matches(xtex))

npe_begin_code()
{
    // construct grid bounding box
    Eigen::Vector3f size(ytex.cols(), xtex.cols(), xtex.rows());
    int maxsize = size.maxCoeff();
    Eigen::AlignedBox3f box(Eigen::Vector3f(0), size / maxsize);

    // construct voxel grid
    Eigen::MatrixXf grid;
    Eigen::RowVector3i gridres;
    igl::voxel_grid(box, maxsize, 0, grid, gridres);
    return std::make_tuple(npe::move(grid), npe::move(gridres));
}
npe_end_code()
