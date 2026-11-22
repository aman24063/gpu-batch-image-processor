// Copyright 2026 Aman Pal
//
// GPU batch image processor.
//
// Loads every PGM image found (recursively) in an input folder and runs a
// three-stage pipeline on the GPU for each one:
//   1. 5x5 Gaussian blur        (integer weights held in __constant__ memory)
//   2. Sobel edge detection     (applied to the blurred image)
//   3. Histogram equalization   (shared-memory atomic histogram + LUT)
// Images are distributed round-robin over several CUDA streams so host<->device
// copies of one image overlap with kernels of another. Every GPU result is
// verified against an identical CPU implementation, and per-image timings,
// sample outputs and a PNG montage are written as proof of execution.
//
// Usage:
//   image_batch_processor --input <dir> [--output <dir>] [--streams <n>]
//                         [--block <n>] [--samples <n>] [--save_all]

#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "image_io.h"

#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t status = (call);                                           \
    if (status != cudaSuccess) {                                           \
      std::fprintf(stderr, "CUDA error '%s' at %s:%d: %s\n", #call,        \
                   __FILE__, __LINE__, cudaGetErrorString(status));        \
      std::exit(EXIT_FAILURE);                                             \
    }                                                                      \
  } while (0)

namespace {

constexpr int kBlurRadius = 2;
constexpr int kBlurWidth = 2 * kBlurRadius + 1;  // 5x5 filter.
constexpr int kBlurShift = 8;                     // Weights sum to 256 = 2^8.
constexpr int kHistogramBins = 256;
constexpr int kMaxHistogramBlocks = 64;

// 5x5 Gaussian = outer product of the binomial row [1 4 6 4 1].
// Integer weights make the GPU and CPU results bit-for-bit identical.
const int kHostBlurWeights[kBlurWidth * kBlurWidth] = {
    1, 4,  6,  4,  1,
    4, 16, 24, 16, 4,
    6, 24, 36, 24, 6,
    4, 16, 24, 16, 4,
    1, 4,  6,  4,  1};

__constant__ int c_blur_weights[kBlurWidth * kBlurWidth];

// ---------------------------------------------------------------------------
// Device code.
// ---------------------------------------------------------------------------

__device__ __host__ inline int Clamp(int value, int low, int high) {
  return value < low ? low : (value > high ? high : value);
}

// One thread per output pixel. Borders are handled by clamping coordinates.
__global__ void GaussianBlurKernel(const unsigned char* input,
                                   unsigned char* output, int width,
                                   int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  int sum = 0;
  for (int dy = -kBlurRadius; dy <= kBlurRadius; ++dy) {
    const int yy = Clamp(y + dy, 0, height - 1);
    for (int dx = -kBlurRadius; dx <= kBlurRadius; ++dx) {
      const int xx = Clamp(x + dx, 0, width - 1);
      const int weight =
          c_blur_weights[(dy + kBlurRadius) * kBlurWidth + dx + kBlurRadius];
      sum += weight * input[yy * width + xx];
    }
  }
  output[y * width + x] =
      static_cast<unsigned char>((sum + (1 << (kBlurShift - 1))) >> kBlurShift);
}

// Sobel gradient magnitude, clamped to 0..255.
__global__ void SobelKernel(const unsigned char* input, unsigned char* output,
                            int width, int height) {
  const int x = blockIdx.x * blockDim.x + threadIdx.x;
  const int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= width || y >= height) return;

  const int xl = Clamp(x - 1, 0, width - 1);
  const int xr = Clamp(x + 1, 0, width - 1);
  const int yu = Clamp(y - 1, 0, height - 1);
  const int yd = Clamp(y + 1, 0, height - 1);

  const int p00 = input[yu * width + xl];
  const int p01 = input[yu * width + x];
  const int p02 = input[yu * width + xr];
  const int p10 = input[y * width + xl];
  const int p12 = input[y * width + xr];
  const int p20 = input[yd * width + xl];
  const int p21 = input[yd * width + x];
  const int p22 = input[yd * width + xr];

  const int gx = (p02 + 2 * p12 + p22) - (p00 + 2 * p10 + p20);
  const int gy = (p20 + 2 * p21 + p22) - (p00 + 2 * p01 + p02);
  const float magnitude = sqrtf(static_cast<float>(gx * gx + gy * gy));
  output[y * width + x] =
      static_cast<unsigned char>(Clamp(static_cast<int>(magnitude + 0.5f),
                                       0, 255));
}

// Each block builds a private histogram in shared memory (fast atomics), then
// merges it into the global histogram with one atomic per bin.
__global__ void HistogramKernel(const unsigned char* input,
                                unsigned int* histogram, int num_pixels) {
  __shared__ unsigned int local_histogram[kHistogramBins];
  for (int i = threadIdx.x; i < kHistogramBins; i += blockDim.x) {
    local_histogram[i] = 0;
  }
  __syncthreads();

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_pixels;
       i += blockDim.x * gridDim.x) {
    atomicAdd(&local_histogram[input[i]], 1u);
  }
  __syncthreads();

  for (int i = threadIdx.x; i < kHistogramBins; i += blockDim.x) {
    atomicAdd(&histogram[i], local_histogram[i]);
  }
}

// Builds the equalization lookup table from the histogram (one block of 256
// threads). The 256-entry prefix sum is tiny, so thread 0 computes it.
__global__ void EqualizationLutKernel(const unsigned int* histogram,
                                      unsigned char* lut, int num_pixels) {
  __shared__ unsigned int cdf[kHistogramBins];
  __shared__ unsigned int cdf_min;
  if (threadIdx.x == 0) {
    unsigned int running = 0;
    cdf_min = 0;
    for (int i = 0; i < kHistogramBins; ++i) {
      running += histogram[i];
      cdf[i] = running;
      if (cdf_min == 0 && running > 0) cdf_min = running;
    }
  }
  __syncthreads();

  const int i = threadIdx.x;
  if (i >= kHistogramBins) return;
  const unsigned int range = static_cast<unsigned int>(num_pixels) - cdf_min;
  if (range == 0 || cdf[i] < cdf_min) {
    lut[i] = (range == 0) ? static_cast<unsigned char>(i) : 0;
    return;
  }
  const unsigned long long scaled =
      (static_cast<unsigned long long>(cdf[i] - cdf_min) * 255ull +
       range / 2) / range;
  lut[i] = static_cast<unsigned char>(scaled);
}

__global__ void ApplyLutKernel(const unsigned char* input,
                               unsigned char* output, const unsigned char* lut,
                               int num_pixels) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < num_pixels) output[i] = lut[input[i]];
}

// ---------------------------------------------------------------------------
// CPU reference implementations (same integer math as the kernels).
// ---------------------------------------------------------------------------

void CpuGaussianBlur(const unsigned char* input, unsigned char* output,
                     int width, int height) {
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int sum = 0;
      for (int dy = -kBlurRadius; dy <= kBlurRadius; ++dy) {
        const int yy = Clamp(y + dy, 0, height - 1);
        for (int dx = -kBlurRadius; dx <= kBlurRadius; ++dx) {
          const int xx = Clamp(x + dx, 0, width - 1);
          sum += kHostBlurWeights[(dy + kBlurRadius) * kBlurWidth + dx +
                                  kBlurRadius] *
                 input[yy * width + xx];
        }
      }
      output[y * width + x] = static_cast<unsigned char>(
          (sum + (1 << (kBlurShift - 1))) >> kBlurShift);
    }
  }
}

void CpuSobel(const unsigned char* input, unsigned char* output, int width,
              int height) {
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const int xl = Clamp(x - 1, 0, width - 1);
      const int xr = Clamp(x + 1, 0, width - 1);
      const int yu = Clamp(y - 1, 0, height - 1);
      const int yd = Clamp(y + 1, 0, height - 1);
      const int gx = (input[yu * width + xr] + 2 * input[y * width + xr] +
                      input[yd * width + xr]) -
                     (input[yu * width + xl] + 2 * input[y * width + xl] +
                      input[yd * width + xl]);
      const int gy = (input[yd * width + xl] + 2 * input[yd * width + x] +
                      input[yd * width + xr]) -
                     (input[yu * width + xl] + 2 * input[yu * width + x] +
                      input[yu * width + xr]);
      const float magnitude = std::sqrt(static_cast<float>(gx * gx + gy * gy));
      output[y * width + x] = static_cast<unsigned char>(
          Clamp(static_cast<int>(magnitude + 0.5f), 0, 255));
    }
  }
}

void CpuEqualize(const unsigned char* input, unsigned char* output,
                 int num_pixels) {
  unsigned int histogram[kHistogramBins] = {0};
  for (int i = 0; i < num_pixels; ++i) ++histogram[input[i]];

  unsigned int cdf[kHistogramBins];
  unsigned int running = 0;
  unsigned int cdf_min = 0;
  for (int i = 0; i < kHistogramBins; ++i) {
    running += histogram[i];
    cdf[i] = running;
    if (cdf_min == 0 && running > 0) cdf_min = running;
  }

  unsigned char lut[kHistogramBins];
  const unsigned int range = static_cast<unsigned int>(num_pixels) - cdf_min;
  for (int i = 0; i < kHistogramBins; ++i) {
    if (range == 0) {
      lut[i] = static_cast<unsigned char>(i);
    } else if (cdf[i] < cdf_min) {
      lut[i] = 0;
    } else {
      lut[i] = static_cast<unsigned char>(
          (static_cast<unsigned long long>(cdf[i] - cdf_min) * 255ull +
           range / 2) / range);
    }
  }
  for (int i = 0; i < num_pixels; ++i) output[i] = lut[input[i]];
}

// ---------------------------------------------------------------------------
// Command-line options.
// ---------------------------------------------------------------------------

struct Options {
  std::string input_dir = "data";
  std::string output_dir = "output";
  int num_streams = 4;
  int block_size = 16;
  int num_samples = 6;
  bool save_all = false;
};

void PrintUsage(const char* program) {
  std::printf(
      "Usage: %s --input <dir> [options]\n"
      "  --input <dir>    Folder searched recursively for .pgm images "
      "(default: data)\n"
      "  --output <dir>   Folder for results (default: output)\n"
      "  --streams <n>    Number of CUDA streams, 1-32 (default: 4)\n"
      "  --block <n>      2D thread block edge, 8/16/32 (default: 16)\n"
      "  --samples <n>    Sample images saved + shown in montage (default: 6)\n"
      "  --save_all       Also save every processed image as PGM\n"
      "  --help           Show this message\n",
      program);
}

bool ParseOptions(int argc, char** argv, Options* options) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const bool has_value = (i + 1 < argc);
    if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      std::exit(EXIT_SUCCESS);
    } else if (arg == "--save_all") {
      options->save_all = true;
    } else if (arg == "--input" && has_value) {
      options->input_dir = argv[++i];
    } else if (arg == "--output" && has_value) {
      options->output_dir = argv[++i];
    } else if (arg == "--streams" && has_value) {
      options->num_streams = std::atoi(argv[++i]);
    } else if (arg == "--block" && has_value) {
      options->block_size = std::atoi(argv[++i]);
    } else if (arg == "--samples" && has_value) {
      options->num_samples = std::atoi(argv[++i]);
    } else {
      std::fprintf(stderr, "Unknown or incomplete argument: %s\n",
                   arg.c_str());
      PrintUsage(argv[0]);
      return false;
    }
  }
  if (options->num_streams < 1 || options->num_streams > 32) {
    std::fprintf(stderr, "--streams must be between 1 and 32\n");
    return false;
  }
  if (options->block_size < 1 || options->block_size > 32) {
    std::fprintf(stderr, "--block must be between 1 and 32\n");
    return false;
  }
  if (options->num_samples < 0) options->num_samples = 0;
  return true;
}

// ---------------------------------------------------------------------------
// File helpers.
// ---------------------------------------------------------------------------

bool EndsWith(const std::string& text, const std::string& suffix) {
  return text.size() >= suffix.size() &&
         text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// Recursively collects all *.pgm files below directory.
void ListPgmFiles(const std::string& directory,
                  std::vector<std::string>* files) {
  DIR* dir = opendir(directory.c_str());
  if (dir == nullptr) return;
  while (struct dirent* entry = readdir(dir)) {
    const std::string name = entry->d_name;
    if (name == "." || name == "..") continue;
    const std::string path = directory + "/" + name;
    struct stat info;
    if (stat(path.c_str(), &info) != 0) continue;
    if (S_ISDIR(info.st_mode)) {
      ListPgmFiles(path, files);
    } else if (EndsWith(name, ".pgm")) {
      files->push_back(path);
    }
  }
  closedir(dir);
}

void MakeDirectory(const std::string& path) { mkdir(path.c_str(), 0755); }

std::string BaseName(const std::string& path) {
  const size_t slash = path.find_last_of('/');
  std::string name =
      (slash == std::string::npos) ? path : path.substr(slash + 1);
  const size_t dot = name.find_last_of('.');
  return (dot == std::string::npos) ? name : name.substr(0, dot);
}

int MaxAbsDifference(const std::vector<unsigned char>& a,
                     const std::vector<unsigned char>& b) {
  int max_diff = 0;
  for (size_t i = 0; i < a.size(); ++i) {
    max_diff = std::max(max_diff, std::abs(static_cast<int>(a[i]) - b[i]));
  }
  return max_diff;
}

// ---------------------------------------------------------------------------
// GPU pipeline.
// ---------------------------------------------------------------------------

struct ProcessedImage {
  std::vector<unsigned char> blurred;
  std::vector<unsigned char> edges;
  std::vector<unsigned char> equalized;
  float gpu_ms = 0.0f;  // Copies + kernels for this image, from CUDA events.
};

// Per-stream resources, sized for the largest image in the batch.
struct StreamContext {
  cudaStream_t stream;
  cudaEvent_t start;
  cudaEvent_t stop;
  unsigned char* host_input = nullptr;  // Pinned memory enables async copies.
  unsigned char* host_blurred = nullptr;
  unsigned char* host_edges = nullptr;
  unsigned char* host_equalized = nullptr;
  unsigned char* device_input = nullptr;
  unsigned char* device_blurred = nullptr;
  unsigned char* device_edges = nullptr;
  unsigned char* device_equalized = nullptr;
  unsigned char* device_lut = nullptr;
  unsigned int* device_histogram = nullptr;
  int image_index = -1;  // Image currently in flight on this stream.
};

void CreateStreamContext(size_t max_pixels, StreamContext* context) {
  CUDA_CHECK(cudaStreamCreateWithFlags(&context->stream,
                                       cudaStreamNonBlocking));
  CUDA_CHECK(cudaEventCreate(&context->start));
  CUDA_CHECK(cudaEventCreate(&context->stop));
  CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&context->host_input),
                            max_pixels));
  CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&context->host_blurred),
                            max_pixels));
  CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&context->host_edges),
                            max_pixels));
  CUDA_CHECK(cudaMallocHost(
      reinterpret_cast<void**>(&context->host_equalized), max_pixels));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&context->device_input),
                        max_pixels));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&context->device_blurred),
                        max_pixels));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&context->device_edges),
                        max_pixels));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&context->device_equalized),
                        max_pixels));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&context->device_lut),
                        kHistogramBins));
  CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&context->device_histogram),
                        kHistogramBins * sizeof(unsigned int)));
}

void DestroyStreamContext(StreamContext* context) {
  CUDA_CHECK(cudaFreeHost(context->host_input));
  CUDA_CHECK(cudaFreeHost(context->host_blurred));
  CUDA_CHECK(cudaFreeHost(context->host_edges));
  CUDA_CHECK(cudaFreeHost(context->host_equalized));
  CUDA_CHECK(cudaFree(context->device_input));
  CUDA_CHECK(cudaFree(context->device_blurred));
  CUDA_CHECK(cudaFree(context->device_edges));
  CUDA_CHECK(cudaFree(context->device_equalized));
  CUDA_CHECK(cudaFree(context->device_lut));
  CUDA_CHECK(cudaFree(context->device_histogram));
  CUDA_CHECK(cudaEventDestroy(context->start));
  CUDA_CHECK(cudaEventDestroy(context->stop));
  CUDA_CHECK(cudaStreamDestroy(context->stream));
}

// Queues the whole pipeline for one image on the context's stream. Nothing
// here blocks the host, so the next stream can be fed immediately.
void EnqueueImage(const image_io::GrayImage& image, int block_size,
                  StreamContext* context) {
  const int width = image.width;
  const int height = image.height;
  const int num_pixels = width * height;
  cudaStream_t stream = context->stream;

  std::memcpy(context->host_input, image.pixels.data(), num_pixels);
  CUDA_CHECK(cudaEventRecord(context->start, stream));
  CUDA_CHECK(cudaMemcpyAsync(context->device_input, context->host_input,
                             num_pixels, cudaMemcpyHostToDevice, stream));

  const dim3 block(block_size, block_size);
  const dim3 grid((width + block_size - 1) / block_size,
                  (height + block_size - 1) / block_size);
  const int linear_blocks = (num_pixels + 255) / 256;
  const int histogram_blocks = std::min(linear_blocks, kMaxHistogramBlocks);

  GaussianBlurKernel<<<grid, block, 0, stream>>>(
      context->device_input, context->device_blurred, width, height);
  SobelKernel<<<grid, block, 0, stream>>>(
      context->device_blurred, context->device_edges, width, height);
  CUDA_CHECK(cudaMemsetAsync(context->device_histogram, 0,
                             kHistogramBins * sizeof(unsigned int), stream));
  HistogramKernel<<<histogram_blocks, 256, 0, stream>>>(
      context->device_input, context->device_histogram, num_pixels);
  EqualizationLutKernel<<<1, kHistogramBins, 0, stream>>>(
      context->device_histogram, context->device_lut, num_pixels);
  ApplyLutKernel<<<linear_blocks, 256, 0, stream>>>(
      context->device_input, context->device_equalized, context->device_lut,
      num_pixels);
  CUDA_CHECK(cudaGetLastError());

  CUDA_CHECK(cudaMemcpyAsync(context->host_blurred, context->device_blurred,
                             num_pixels, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(context->host_edges, context->device_edges,
                             num_pixels, cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaMemcpyAsync(context->host_equalized,
                             context->device_equalized, num_pixels,
                             cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaEventRecord(context->stop, stream));
}

// Waits for the context's image to finish and copies the results out.
void CollectImage(const image_io::GrayImage& image, StreamContext* context,
                  ProcessedImage* result) {
  CUDA_CHECK(cudaStreamSynchronize(context->stream));
  const size_t num_pixels = static_cast<size_t>(image.width) * image.height;
  result->blurred.assign(context->host_blurred,
                         context->host_blurred + num_pixels);
  result->edges.assign(context->host_edges, context->host_edges + num_pixels);
  result->equalized.assign(context->host_equalized,
                           context->host_equalized + num_pixels);
  CUDA_CHECK(cudaEventElapsedTime(&result->gpu_ms, context->start,
                                  context->stop));
}

// Builds a grid image: one row per sample, columns = original | blurred |
// edges | equalized. Each cell is padded to the largest sample size.
void WriteMontage(const std::string& path,
                  const std::vector<image_io::GrayImage>& images,
                  const std::vector<ProcessedImage>& results,
                  const std::vector<int>& sample_indices) {
  if (sample_indices.empty()) return;
  constexpr int kColumns = 4;
  constexpr int kGap = 4;
  constexpr unsigned char kBackground = 40;
  int cell_width = 0;
  int cell_height = 0;
  for (int index : sample_indices) {
    cell_width = std::max(cell_width, images[index].width);
    cell_height = std::max(cell_height, images[index].height);
  }
  const int rows = static_cast<int>(sample_indices.size());
  const int montage_width = kColumns * cell_width + (kColumns + 1) * kGap;
  const int montage_height = rows * cell_height + (rows + 1) * kGap;
  std::vector<unsigned char> montage(
      static_cast<size_t>(montage_width) * montage_height, kBackground);

  for (int row = 0; row < rows; ++row) {
    const int index = sample_indices[row];
    const image_io::GrayImage& image = images[index];
    const unsigned char* columns[kColumns] = {
        image.pixels.data(), results[index].blurred.data(),
        results[index].edges.data(), results[index].equalized.data()};
    for (int column = 0; column < kColumns; ++column) {
      const int origin_x = kGap + column * (cell_width + kGap);
      const int origin_y = kGap + row * (cell_height + kGap);
      for (int y = 0; y < image.height; ++y) {
        std::memcpy(&montage[static_cast<size_t>(origin_y + y) *
                                 montage_width + origin_x],
                    columns[column] + static_cast<size_t>(y) * image.width,
                    image.width);
      }
    }
  }
  image_io::WritePngGray(path, montage_width, montage_height, montage.data());
}

double MillisecondsSince(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double, std::milli>(
             std::chrono::steady_clock::now() - start)
      .count();
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  if (!ParseOptions(argc, argv, &options)) return EXIT_FAILURE;

  // ---- Device information ----
  int device = 0;
  cudaDeviceProp properties;
  CUDA_CHECK(cudaGetDevice(&device));
  CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
  std::printf("=== GPU Batch Image Processor ===\n");
  std::printf("GPU: %s (compute capability %d.%d, %d SMs)\n",
              properties.name, properties.major, properties.minor,
              properties.multiProcessorCount);
  std::printf("Input: %s | Output: %s | Streams: %d | Block: %dx%d\n",
              options.input_dir.c_str(), options.output_dir.c_str(),
              options.num_streams, options.block_size, options.block_size);

  // ---- Load the dataset ----
  std::vector<std::string> files;
  ListPgmFiles(options.input_dir, &files);
  std::sort(files.begin(), files.end());

  std::vector<image_io::GrayImage> images;
  images.reserve(files.size());
  size_t max_pixels = 0;
  size_t total_pixels = 0;
  int skipped = 0;
  for (const std::string& file : files) {
    image_io::GrayImage image;
    if (!image_io::ReadPgm(file, &image)) {
      ++skipped;
      continue;
    }
    max_pixels = std::max(max_pixels, image.pixels.size());
    total_pixels += image.pixels.size();
    images.push_back(std::move(image));
  }
  std::printf("Found %zu PGM files, loaded %zu images (%d unreadable), "
              "%.2f megapixels in total\n",
              files.size(), images.size(), skipped, total_pixels / 1.0e6);
  if (images.empty()) {
    std::fprintf(stderr,
                 "No images found in '%s'. Run data/download_data.sh first.\n",
                 options.input_dir.c_str());
    return EXIT_FAILURE;
  }

  MakeDirectory(options.output_dir);
  const std::string samples_dir = options.output_dir + "/samples";
  const std::string all_dir = options.output_dir + "/all";
  MakeDirectory(samples_dir);
  if (options.save_all) MakeDirectory(all_dir);

  // ---- GPU processing: round-robin over the streams ----
  CUDA_CHECK(cudaMemcpyToSymbol(c_blur_weights, kHostBlurWeights,
                                sizeof(kHostBlurWeights)));
  std::vector<StreamContext> contexts(options.num_streams);
  for (StreamContext& context : contexts) {
    CreateStreamContext(max_pixels, &context);
  }

  std::vector<ProcessedImage> results(images.size());
  const auto gpu_start = std::chrono::steady_clock::now();
  for (size_t base = 0; base < images.size(); base += contexts.size()) {
    // Feed one image to every stream without waiting...
    for (size_t s = 0; s < contexts.size(); ++s) {
      const size_t index = base + s;
      contexts[s].image_index =
          (index < images.size()) ? static_cast<int>(index) : -1;
      if (contexts[s].image_index >= 0) {
        EnqueueImage(images[index], options.block_size, &contexts[s]);
      }
    }
    // ...then collect them, so copies and kernels of different images overlap.
    for (StreamContext& context : contexts) {
      if (context.image_index < 0) continue;
      CollectImage(images[context.image_index], &context,
                   &results[context.image_index]);
    }
  }
  const double gpu_wall_ms = MillisecondsSince(gpu_start);

  // ---- CPU reference + verification ----
  int failures = 0;
  double cpu_wall_ms = 0.0;
  std::vector<int> max_differences(images.size(), 0);
  for (size_t i = 0; i < images.size(); ++i) {
    const image_io::GrayImage& image = images[i];
    const size_t num_pixels = image.pixels.size();
    std::vector<unsigned char> blurred(num_pixels);
    std::vector<unsigned char> edges(num_pixels);
    std::vector<unsigned char> equalized(num_pixels);

    const auto cpu_start = std::chrono::steady_clock::now();
    CpuGaussianBlur(image.pixels.data(), blurred.data(), image.width,
                    image.height);
    CpuSobel(blurred.data(), edges.data(), image.width, image.height);
    CpuEqualize(image.pixels.data(), equalized.data(),
                static_cast<int>(num_pixels));
    cpu_wall_ms += MillisecondsSince(cpu_start);

    max_differences[i] =
        std::max({MaxAbsDifference(blurred, results[i].blurred),
                  MaxAbsDifference(edges, results[i].edges),
                  MaxAbsDifference(equalized, results[i].equalized)});
    if (max_differences[i] != 0) ++failures;
  }

  // ---- Outputs: per-image log + CSV, samples, montage ----
  const std::string csv_path = options.output_dir + "/timings.csv";
  FILE* csv = std::fopen(csv_path.c_str(), "w");
  if (csv != nullptr) {
    std::fprintf(csv, "index,file,width,height,gpu_ms,max_diff,verified\n");
  }
  double gpu_event_total_ms = 0.0;
  for (size_t i = 0; i < images.size(); ++i) {
    const image_io::GrayImage& image = images[i];
    gpu_event_total_ms += results[i].gpu_ms;
    const char* status = (max_differences[i] == 0) ? "OK" : "MISMATCH";
    std::printf("[%4zu/%zu] %-45s %3dx%-3d gpu %.3f ms  verify %s\n", i + 1,
                images.size(), image.path.c_str(), image.width, image.height,
                results[i].gpu_ms, status);
    if (csv != nullptr) {
      std::fprintf(csv, "%zu,%s,%d,%d,%.4f,%d,%s\n", i, image.path.c_str(),
                   image.width, image.height, results[i].gpu_ms,
                   max_differences[i], status);
    }
    if (options.save_all) {
      const std::string base = all_dir + "/" + BaseName(image.path);
      image_io::WritePgm(base + "_blur.pgm", image.width, image.height,
                         results[i].blurred.data());
      image_io::WritePgm(base + "_edges.pgm", image.width, image.height,
                         results[i].edges.data());
      image_io::WritePgm(base + "_equalized.pgm", image.width, image.height,
                         results[i].equalized.data());
    }
  }
  if (csv != nullptr) std::fclose(csv);

  // Samples: evenly spaced among the largest images (best for viewing).
  std::vector<int> large_indices;
  for (size_t i = 0; i < images.size(); ++i) {
    if (images[i].pixels.size() == max_pixels) large_indices.push_back(i);
  }
  std::vector<int> sample_indices;
  const int num_samples =
      std::min(options.num_samples, static_cast<int>(large_indices.size()));
  for (int k = 0; k < num_samples; ++k) {
    sample_indices.push_back(
        large_indices[static_cast<size_t>(k) * large_indices.size() /
                      num_samples]);
  }
  for (int index : sample_indices) {
    const image_io::GrayImage& image = images[index];
    const std::string base = samples_dir + "/" + BaseName(image.path);
    image_io::WritePngGray(base + "_original.png", image.width, image.height,
                           image.pixels.data());
    image_io::WritePngGray(base + "_blur.png", image.width, image.height,
                           results[index].blurred.data());
    image_io::WritePngGray(base + "_edges.png", image.width, image.height,
                           results[index].edges.data());
    image_io::WritePngGray(base + "_equalized.png", image.width,
                           image.height, results[index].equalized.data());
  }
  WriteMontage(options.output_dir + "/montage.png", images, results,
               sample_indices);

  // ---- Summary ----
  std::printf("\n=== Summary ===\n");
  std::printf("Images processed in this run : %zu\n", images.size());
  std::printf("Total pixels                 : %zu\n", total_pixels);
  std::printf("GPU wall time (all streams)  : %.3f ms\n", gpu_wall_ms);
  std::printf("Sum of per-image GPU times   : %.3f ms\n", gpu_event_total_ms);
  std::printf("Stream overlap factor        : %.2fx\n",
              gpu_wall_ms > 0 ? gpu_event_total_ms / gpu_wall_ms : 0.0);
  std::printf("CPU reference time           : %.3f ms\n", cpu_wall_ms);
  std::printf("GPU throughput               : %.1f images/s, %.1f MP/s\n",
              images.size() / (gpu_wall_ms / 1000.0),
              total_pixels / 1.0e6 / (gpu_wall_ms / 1000.0));
  std::printf("Verification vs CPU          : %s (%d mismatching images)\n",
              failures == 0 ? "PASSED" : "FAILED", failures);
  std::printf("Wrote %s, %s/montage.png and %zu sample sets in %s\n",
              csv_path.c_str(), options.output_dir.c_str(),
              sample_indices.size(), samples_dir.c_str());

  for (StreamContext& context : contexts) DestroyStreamContext(&context);
  CUDA_CHECK(cudaDeviceReset());
  return failures == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
