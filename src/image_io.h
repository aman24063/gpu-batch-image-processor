// Copyright 2026 Aman Pal
//
// Minimal, dependency-free I/O for 8-bit grayscale images.
//   ReadPgm      - reads binary (P5) and ASCII (P2) PGM files.
//   WritePgm     - writes binary (P5) PGM files.
//   WritePngGray - writes an 8-bit grayscale PNG using uncompressed
//                  ("stored") deflate blocks, so no zlib is needed and the
//                  result can be viewed directly on GitHub.

#ifndef SRC_IMAGE_IO_H_
#define SRC_IMAGE_IO_H_

#include <cctype>
#include <cstdint>
#include <fstream>
#include <string>
#include <vector>

namespace image_io {

struct GrayImage {
  std::string path;                   // File the image was loaded from.
  int width = 0;
  int height = 0;
  std::vector<unsigned char> pixels;  // width * height values, row-major.
};

// Reads the next whitespace-separated PGM header token, skipping comments.
inline bool ReadPgmToken(std::istream* in, std::string* token) {
  token->clear();
  char c;
  while (in->get(c)) {
    if (c == '#') {
      std::string comment;
      std::getline(*in, comment);
      continue;
    }
    if (std::isspace(static_cast<unsigned char>(c))) {
      if (!token->empty()) return true;
      continue;
    }
    token->push_back(c);
  }
  return !token->empty();
}

// Loads a PGM file into *image. Returns false on any parsing error.
inline bool ReadPgm(const std::string& path, GrayImage* image) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;

  std::string magic, width_str, height_str, maxval_str;
  if (!ReadPgmToken(&in, &magic) || (magic != "P5" && magic != "P2")) {
    return false;
  }
  if (!ReadPgmToken(&in, &width_str) || !ReadPgmToken(&in, &height_str) ||
      !ReadPgmToken(&in, &maxval_str)) {
    return false;
  }

  const int width = std::stoi(width_str);
  const int height = std::stoi(height_str);
  const int maxval = std::stoi(maxval_str);
  if (width <= 0 || height <= 0 || maxval <= 0 || maxval > 255) return false;

  image->path = path;
  image->width = width;
  image->height = height;
  image->pixels.assign(static_cast<size_t>(width) * height, 0);

  if (magic == "P5") {
    // ReadPgmToken already consumed the single whitespace after maxval.
    in.read(reinterpret_cast<char*>(image->pixels.data()),
            image->pixels.size());
    return in.gcount() == static_cast<std::streamsize>(image->pixels.size());
  }

  std::string value;
  for (size_t i = 0; i < image->pixels.size(); ++i) {
    if (!ReadPgmToken(&in, &value)) return false;
    image->pixels[i] = static_cast<unsigned char>(std::stoi(value));
  }
  return true;
}

// Writes width x height pixels as a binary (P5) PGM file.
inline bool WritePgm(const std::string& path, int width, int height,
                     const unsigned char* data) {
  std::ofstream out(path, std::ios::binary);
  if (!out) return false;
  out << "P5\n" << width << " " << height << "\n255\n";
  out.write(reinterpret_cast<const char*>(data),
            static_cast<std::streamsize>(width) * height);
  return static_cast<bool>(out);
}

// ---------------------------------------------------------------------------
// PNG writer helpers.
// ---------------------------------------------------------------------------

// Standard CRC-32 used by PNG chunks.
inline uint32_t PngCrc(const unsigned char* buffer, size_t length) {
  static uint32_t table[256];
  static bool initialized = false;
  if (!initialized) {
    for (uint32_t n = 0; n < 256; ++n) {
      uint32_t c = n;
      for (int k = 0; k < 8; ++k) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
      table[n] = c;
    }
    initialized = true;
  }
  uint32_t crc = 0xFFFFFFFFu;
  for (size_t i = 0; i < length; ++i) {
    crc = table[(crc ^ buffer[i]) & 0xFF] ^ (crc >> 8);
  }
  return crc ^ 0xFFFFFFFFu;
}

// Appends a 32-bit big-endian integer.
inline void PutU32(uint32_t value, std::vector<unsigned char>* out) {
  out->push_back((value >> 24) & 0xFF);
  out->push_back((value >> 16) & 0xFF);
  out->push_back((value >> 8) & 0xFF);
  out->push_back(value & 0xFF);
}

// Writes one PNG chunk: length, type, data, CRC(type + data).
inline void WritePngChunk(const char* type,
                          const std::vector<unsigned char>& data,
                          std::ofstream* out) {
  std::vector<unsigned char> chunk;
  PutU32(static_cast<uint32_t>(data.size()), &chunk);
  chunk.insert(chunk.end(), type, type + 4);
  chunk.insert(chunk.end(), data.begin(), data.end());
  PutU32(PngCrc(chunk.data() + 4, chunk.size() - 4), &chunk);
  out->write(reinterpret_cast<const char*>(chunk.data()), chunk.size());
}

// Writes width x height grayscale pixels as an 8-bit PNG file.
inline bool WritePngGray(const std::string& path, int width, int height,
                         const unsigned char* data) {
  std::ofstream out(path, std::ios::binary);
  if (!out) return false;

  const unsigned char kSignature[8] = {137, 80, 78, 71, 13, 10, 26, 10};
  out.write(reinterpret_cast<const char*>(kSignature), 8);

  std::vector<unsigned char> header;
  PutU32(width, &header);
  PutU32(height, &header);
  header.push_back(8);  // Bit depth.
  header.push_back(0);  // Color type: grayscale.
  header.push_back(0);  // Compression method.
  header.push_back(0);  // Filter method.
  header.push_back(0);  // Interlace method.
  WritePngChunk("IHDR", header, &out);

  // Raw scanlines, each prefixed with filter type 0 (None).
  std::vector<unsigned char> raw;
  raw.reserve(static_cast<size_t>(height) * (width + 1));
  for (int y = 0; y < height; ++y) {
    raw.push_back(0);
    const unsigned char* row = data + static_cast<size_t>(y) * width;
    raw.insert(raw.end(), row, row + width);
  }

  // zlib stream of stored deflate blocks (<= 65535 bytes) + Adler-32.
  std::vector<unsigned char> zlib = {0x78, 0x01};
  size_t pos = 0;
  do {
    size_t length = raw.size() - pos;
    if (length > 65535) length = 65535;
    const bool is_final = (pos + length == raw.size());
    zlib.push_back(is_final ? 1 : 0);
    zlib.push_back(length & 0xFF);
    zlib.push_back((length >> 8) & 0xFF);
    zlib.push_back(~length & 0xFF);
    zlib.push_back((~length >> 8) & 0xFF);
    zlib.insert(zlib.end(), raw.begin() + pos, raw.begin() + pos + length);
    pos += length;
  } while (pos < raw.size());

  uint32_t adler_a = 1;
  uint32_t adler_b = 0;
  for (unsigned char c : raw) {
    adler_a = (adler_a + c) % 65521;
    adler_b = (adler_b + adler_a) % 65521;
  }
  PutU32((adler_b << 16) | adler_a, &zlib);
  WritePngChunk("IDAT", zlib, &out);
  WritePngChunk("IEND", std::vector<unsigned char>(), &out);
  return static_cast<bool>(out);
}

}  // namespace image_io

#endif  // SRC_IMAGE_IO_H_
