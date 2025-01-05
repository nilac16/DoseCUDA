#include "TextureClasses.cuh"
#include "CudaClasses.cuh"


void Texture3D::makeArray(const PointIJK &size)
{
    auto chdesc = cudaCreateChannelDesc<float>();
    auto extent = make_cudaExtent(size.i, size.j, size.k);

    CUDA_CHECK(cudaMalloc3DArray(&m_arr, &chdesc, extent, cudaArrayDefault));
}


void Texture3D::makeTexture(cudaTextureFilterMode filterMode)
{
    cudaResourceDesc rsdesc{ };
    cudaTextureDesc txdesc{ };

    rsdesc.resType = cudaResourceTypeArray;
    rsdesc.res.array.array = m_arr;

    txdesc.addressMode[0]               = cudaAddressModeClamp;
    txdesc.addressMode[1]               = cudaAddressModeClamp;
    txdesc.addressMode[2]               = cudaAddressModeClamp;
    txdesc.filterMode                   = filterMode;
    txdesc.readMode                     = cudaReadModeElementType;
    txdesc.sRGB                         = false;
    txdesc.borderColor[0]               = 0.0f;
    txdesc.borderColor[1]               = 0.0f;
    txdesc.borderColor[2]               = 0.0f;
    txdesc.borderColor[3]               = 0.0f;
    txdesc.normalizedCoords             = false;
    txdesc.maxAnisotropy                = 0;
    txdesc.mipmapFilterMode             = cudaFilterModePoint;
    txdesc.mipmapLevelBias              = 0.0f;
    txdesc.minMipmapLevelClamp          = 0.0f;
    txdesc.maxMipmapLevelClamp          = 0.0f;
    txdesc.disableTrilinearOptimization = false;

    CUDA_CHECK(cudaCreateTextureObject(&m_tex, &rsdesc, &txdesc, nullptr));
}


Texture3D::Texture3D(const float data[], const PointIJK &size, cudaTextureFilterMode filterMode, cudaMemcpyKind direction):
    m_arr(NULL),
    m_tex(0),
    m_isCopy(false)
{
    this->makeArray(size);

    cudaMemcpy3DParms params{ };
    params.srcPtr   = make_cudaPitchedPtr((void *)data, size.i * sizeof *data, size.i, size.j);
    params.dstArray = m_arr;
    params.extent   = make_cudaExtent(size.i, size.j, size.k);
    params.kind     = direction;
    CUDA_CHECK(cudaMemcpy3D(&params));

    this->makeTexture(filterMode);
}


Texture3D::~Texture3D()
{
    if (!m_isCopy) {
        cudaDestroyTextureObject(m_tex);
        cudaFreeArray(m_arr);
    }
}
