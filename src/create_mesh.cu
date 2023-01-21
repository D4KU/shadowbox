#include <cmath>
#include <future>
#include <iostream>
#include <thread>
#include <tuple>
#include <vector>

#include <Eigen/Dense>

#include <cuda_runtime.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include <Python.h>

#include <nanovdb/util/CudaDeviceBuffer.h>
#include <nanovdb/util/GridBuilder.h>
#include <nanovdb/util/NanoToOpenVDB.h>
#include <openvdb/openvdb.h>
#include <openvdb/Grid.h>
#include <openvdb/Types.h>
#include <openvdb/io/io.h>
#include <openvdb/tools/VolumeToMesh.h>
#include <openvdb/tools/Prune.h>

namespace vdb = openvdb;

__global__ void add(
    const float* xtex,
    const float* ytex,
    const float* ztex,
    const size_t xres,
    const size_t yres,
    const size_t zres,
    const float iso,
    nanovdb::NanoGrid<bool>* grid)
{
    int xstart = blockIdx.x * blockDim.x + threadIdx.x;
    int ystart = blockIdx.y * blockDim.y + threadIdx.y;
    int zstart = blockIdx.z * blockDim.z + threadIdx.z;

    int xstep = blockDim.x * gridDim.x;
    int ystep = blockDim.y * gridDim.y;
    int zstep = blockDim.z * gridDim.z;

    auto acc = grid->getAccessor();

    for (int x = xstart; x < xres; x += xstep)
        for (int y = ystart; y < yres; y += ystep)
            if (ztex[x * yres + y] >= iso)
                for (int z = zstart; z < zres; z += zstep)
                    if (xtex[y * zres + z] >= iso && ytex[x * zres + z] >= iso)
                        acc.setValue(nanovdb::Coord(x, y, z), true);
}

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

auto create_mesh(
    const Eigen::Ref<const Eigen::MatrixXf>& xtex,
    const Eigen::Ref<const Eigen::MatrixXf>& ytex,
    const Eigen::Ref<const Eigen::MatrixXf>& ztex,
    const float iso,
    const float adaptivity)
{
    const size_t xres = ytex.cols();
    const size_t yres = xtex.cols();
    const size_t zres = xtex.rows();

    const uint64_t xsize = xtex.size() * sizeof(float);
    const uint64_t ysize = ytex.size() * sizeof(float);
    const uint64_t zsize = ztex.size() * sizeof(float);

    const float* xraw = xtex.data();
    const float* yraw = ytex.data();
    const float* zraw = ztex.data();

    float* xcuda;
    float* ycuda;
    float* zcuda;

    auto t1 = std::chrono::high_resolution_clock::now();
    cudaMalloc((void**)&xcuda, xsize);
    cudaMalloc((void**)&ycuda, ysize);
    cudaMalloc((void**)&zcuda, zsize);

    cudaMemcpy(xcuda, xraw, xsize, cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(ycuda, yraw, ysize, cudaMemcpyKind::cudaMemcpyHostToDevice);
    cudaMemcpy(zcuda, zraw, zsize, cudaMemcpyKind::cudaMemcpyHostToDevice);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    nanovdb::GridBuilder<bool> builder(false);
    auto handle = builder.getHandle<>();
    handle.deviceUpload(stream, true);
    auto* grid = handle.deviceGrid<bool>();

    dim3 numThreads(256, 256, 256);
    dim3 numBlocks(
        (int)ceil(xres / numThreads.x),
        (int)ceil(yres / numThreads.y),
        (int)ceil(zres / numThreads.z));

    auto t2 = std::chrono::high_resolution_clock::now();
    auto t3 = std::chrono::high_resolution_clock::now();

    add<<<numBlocks, numThreads, 0, stream>>>(xcuda, ycuda, zcuda, xres, yres, zres, iso, grid);
    cudaDeviceSynchronize();
    handle.deviceDownload(stream, true);

    auto t4 = std::chrono::high_resolution_clock::now();
    auto t5 = std::chrono::high_resolution_clock::now();

    cudaStreamDestroy(stream);
    cudaFree(xcuda);
    cudaFree(ycuda);
    cudaFree(zcuda);

    auto t6 = std::chrono::high_resolution_clock::now();

    auto d1 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1);
    std::cout << d1.count() << " ms" << std::endl;

    auto d3 = std::chrono::duration_cast<std::chrono::milliseconds>(t4 - t3);
    std::cout << d3.count() << " ms" << std::endl;

    auto d5 = std::chrono::duration_cast<std::chrono::milliseconds>(t6 - t5);
    std::cout << d5.count() << " ms" << std::endl;

    std::vector<vdb::Vec3s> points;
    std::vector<vdb::Vec4I> quads;
    std::vector<vdb::Vec3I> tris;

    auto vdbgrid = nanovdb::nanoToOpenVDB(*grid, 1);
    vdb::tools::volumeToMesh(*vdbgrid, points, tris, quads, .5, adaptivity);

    const size_t psize = points.size();
    const size_t qsize = quads.size();
    const size_t tsize = tris.size();

    Eigen::VectorXf verts(psize * 3);
    Eigen::VectorXi faces(qsize * 4 + tsize * 3);
    Eigen::VectorXi steps(qsize + tsize);
    Eigen::VectorXi corners(qsize + tsize);

    const float res = 1 / (float)std::max(std::max(xres, yres), zres);

    for (size_t i = 0; i < psize; ++i)
    {
        verts[i * 3    ] = points[i][0] * res;
        verts[i * 3 + 1] = points[i][1] * res;
        verts[i * 3 + 2] = points[i][2] * res;
    }

    for (size_t i = 0; i < qsize; ++i)
    {
        faces[i * 4    ] = quads[i][0];
        faces[i * 4 + 1] = quads[i][1];
        faces[i * 4 + 2] = quads[i][2];
        faces[i * 4 + 3] = quads[i][3];
    }

    for (size_t i = 0; i < tsize; ++i)
    {
        const size_t j = qsize * 4 + i * 3;
        faces[j    ] = tris[i][0];
        faces[j + 1] = tris[i][1];
        faces[j + 2] = tris[i][2];
    }

    for (size_t i = 0; i < qsize; ++i)
        steps[i] = i * 4;
    for (size_t i = 0; i < tsize; ++i)
        steps[qsize + i] = qsize * 4 + i * 3;

    for (size_t i = 0; i < qsize; ++i)
        corners[i] = 4;
    for (size_t i = 0; i < tsize; ++i)
        corners[qsize + i] = 3;

    return std::make_tuple(
        std::move(verts),
        std::move(faces),
        std::move(steps),
        std::move(corners));
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
