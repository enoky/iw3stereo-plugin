// The reference file tools/dump_pipeline_reference.py writes, and its reader.
//
// Shared by test_pipeline.cpp and test_monobw_gpu.cu so both check the CPU and
// the GPU against the same bytes. The format is deliberately trivial:
//
//     "IW3P" | case count | then per case:
//     name length | name | int count | ints | input count | floats |
//     output count | floats

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace iw3test
{

struct Case
{
    std::string name;
    std::vector<int32_t> ints;
    std::vector<float> inputs;
    std::vector<float> outputs;
};

template <typename T>
inline bool readVector(std::ifstream& stream, std::vector<T>& out)
{
    int32_t count = 0;
    if (!stream.read(reinterpret_cast<char*>(&count), 4)) return false;
    out.resize(size_t(count));
    if (count > 0 && !stream.read(reinterpret_cast<char*>(out.data()),
                                  std::streamsize(count * sizeof(T))))
    {
        return false;
    }
    return true;
}

inline bool load(const std::string& path, std::vector<Case>& cases)
{
    std::ifstream stream(path, std::ios::binary);
    if (!stream)
    {
        std::printf("cannot open %s\n", path.c_str());
        return false;
    }
    char magic[4] = {};
    stream.read(magic, 4);
    if (std::memcmp(magic, "IW3P", 4) != 0)
    {
        std::printf("bad magic in %s\n", path.c_str());
        return false;
    }
    int32_t count = 0;
    stream.read(reinterpret_cast<char*>(&count), 4);
    for (int32_t i = 0; i < count; ++i)
    {
        Case entry;
        int32_t nameLength = 0;
        stream.read(reinterpret_cast<char*>(&nameLength), 4);
        entry.name.resize(size_t(nameLength));
        stream.read(entry.name.data(), nameLength);
        if (!readVector(stream, entry.ints)) return false;
        if (!readVector(stream, entry.inputs)) return false;
        if (!readVector(stream, entry.outputs)) return false;
        cases.push_back(std::move(entry));
    }
    return true;
}

}  // namespace iw3test
