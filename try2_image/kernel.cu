#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <iostream>
#include <string>
#include <cassert>
#include <chrono>

#include "stb_image.h"
#include "stb_image_write.h"

struct Pixel
{
    unsigned char r, g, b, a;
};

void ConvertImageToGrayCpu(unsigned char* imageRGBA, int width, int height)
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            Pixel* ptrPixel = (Pixel*)&imageRGBA[y * width * 4 + 4 * x];
            unsigned char pixelValue = (unsigned char)(ptrPixel->r * 0.2126f + ptrPixel->g * 0.7152f + ptrPixel->b * 0.0722f);
            ptrPixel->r = pixelValue;
            ptrPixel->g = pixelValue;
            ptrPixel->b = pixelValue;
            ptrPixel->a = 255;
        }
    }
}

__global__ void ConvertImageToGrayGpu(unsigned char* imageRGBA)
{
    uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t idx = y * blockDim.x * gridDim.x + x;

    Pixel* ptrPixel = (Pixel*)&imageRGBA[idx * 4];
    unsigned char pixelValue = (unsigned char)
        (ptrPixel->r * 0.2126f + ptrPixel->g * 0.7152f + ptrPixel->b * 0.0722f);
    ptrPixel->r = pixelValue;
    ptrPixel->g = pixelValue;
    ptrPixel->b = pixelValue;
    ptrPixel->a = 255;
}

int main(int argc, char** argv)
{
    // Check argument count
    if (argc < 2)
    {
        std::cout << "Usage: 02_ImageToGray <filename>";
        return -1;
    }

    // Open image
    int width, height, componentCount;
    std::cout << "Loading png file...";
    unsigned char* imageData = stbi_load(argv[1], &width, &height, &componentCount, 4);
    if (!imageData)
    {
        std::cout << std::endl << "Failed to open \"" << argv[1] << "\"";
        return -1;
    }
    std::cout << " DONE" << std::endl;

    // Validate image sizes
    if (width % 32 || height % 32)
    {
        std::cout << "Width and/or Height is not divisible by 32!";
        return -1;
    }

    // CPU processing
    std::cout << "Running on CPU...";
    auto startCpu = std::chrono::high_resolution_clock::now();
    ConvertImageToGrayCpu(imageData, width, height);
    auto endCpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpuDuration = endCpu - startCpu;
    std::cout << " DONE" << std::endl;

    // Save the CPU-processed image
    std::string cpuFileName = "ship_4k_rgba_gray_cpu.png";
    std::cout << "Writing png to disk FOR CPU...";
    stbi_write_png(cpuFileName.c_str(), width, height, 4, imageData, 4 * width);
    std::cout << " DONE" << std::endl;
    std::cout << "CPU TIME: " << cpuDuration.count() << " seconds" << std::endl;

    // Reload the original image to avoid processing already processed data
    stbi_image_free(imageData);
    imageData = stbi_load(argv[1], &width, &height, &componentCount, 4);
    if (!imageData)
    {
        std::cout << std::endl << "Failed to reopen \"" << argv[1] << "\"";
        return -1;
    }

    // GPU processing
    std::cout << "Copy data to GPU...";
    unsigned char* ptrImageDataGpu = nullptr;
    assert(cudaMalloc(&ptrImageDataGpu, width * height * 4) == cudaSuccess);
    assert(cudaMemcpy(ptrImageDataGpu, imageData, width * height * 4, cudaMemcpyHostToDevice) == cudaSuccess);
    std::cout << " DONE" << std::endl;

    // Run the kernel
    std::cout << "Running CUDA Kernel...";
    dim3 blockSize(32, 32);
    dim3 gridSize(width / blockSize.x, height / blockSize.y);
    auto startGpu = std::chrono::high_resolution_clock::now();
    ConvertImageToGrayGpu << <gridSize, blockSize >> > (ptrImageDataGpu);
    cudaDeviceSynchronize();
    auto endGpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpuDuration = endGpu - startGpu;
    std::cout << " DONE" << std::endl;

    // Copy the processed data back from the GPU
    std::cout << "Copy data from GPU...";
    assert(cudaMemcpy(imageData, ptrImageDataGpu, width * height * 4, cudaMemcpyDeviceToHost) == cudaSuccess);
    std::cout << " DONE" << std::endl;

    // Save the GPU-processed image
    std::string gpuFileName = "ship_4k_rgba_gray_gpu.png";
    std::cout << "Writing png to disk FOR GPU...";
    stbi_write_png(gpuFileName.c_str(), width, height, 4, imageData, 4 * width);
    std::cout << " DONE" << std::endl;
    std::cout << "GPU TIME: " << gpuDuration.count() << " seconds" << std::endl;

    // Free memory
    cudaFree(ptrImageDataGpu);
    stbi_image_free(imageData);

    return 0;
}
