// Copyright (c) 2026 FORMOSA-GPGPU Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "parseParameters.h"
#include "imageHelpers.h"

#include <vector>

static void resetSimtixOptions()
{
    gSimtixMode = false;
    gSimtixSamples = 64;
    gWimpyMode = false;
    gNumWorkerThreads = 0;
    gNumThreadPoolThreads = 0;
}

int main()
{
    {
        resetSimtixOptions();
        const char *argv[] = { "test", "--simtix" };
        int argc = 2;
        std::vector<std::string> removed;
        bool help = false;

        if (parseCommonParamAndGetRemovedArgs(argc, argv, removed, help) != 1
            || !gSimtixMode || gSimtixSamples != 64 || !gWimpyMode
            || gNumWorkerThreads != 1 || gNumThreadPoolThreads != 1)
            return 1;

        if (capSimtixNumElements(0x4000) != 64 || capSimtixNumElements(8) != 8
            || capSimtixDimension(1, 100) != 64
            || capSimtixDimension(2, 100) != 8
            || capSimtixDimension(3, 100) != 4 || capSimtixExponent(26) != 6)
            return 1;
    }

    {
        resetSimtixOptions();
        const char *argv[] = { "test", "--simtix-samples", "17", "--simtix" };
        int argc = 4;
        std::vector<std::string> removed;
        bool help = false;

        if (parseCommonParamAndGetRemovedArgs(argc, argv, removed, help) != 1
            || gSimtixSamples != 17 || !gSimtixMode)
            return 1;

        if (capSimtixNumElements(0x4000, 36) != 36) return 1;
    }

    {
        resetSimtixOptions();
        const char *argv[] = { "test", "--simtix", "--simtix-samples", "0" };
        int argc = 4;
        std::vector<std::string> removed;
        bool help = false;

        if (parseCommonParamAndGetRemovedArgs(argc, argv, removed, help) != -1)
            return 1;
    }

    {
        resetSimtixOptions();
        const char *argv[] = { "test", "--simtix-samples", "64" };
        int argc = 3;
        std::vector<std::string> removed;
        bool help = false;

        if (parseCommonParamAndGetRemovedArgs(argc, argv, removed, help) != -1)
            return 1;
    }

    {
        resetSimtixOptions();
        if (capSimtixNumElements(0x4000) != 0x4000
            || capSimtixDimension(2, 100) != 100 || capSimtixExponent(26) != 26)
            return 1;
    }

    {
        gSimtixMode = true;
        gSimtixSamples = 1;
        if (capSimtixNumElements(512, 11) != 11
            || capSimtixNumElements(512, 49) != 49
            || capSimtixNumElements(64) != 1)
            return 1;
        gSimtixSamples = 64;
        if (capSimtixNumElements(512, 49) != 64) return 1;
        gSimtixSamples = 1024;
        if (capSimtixNumElements(512, 49) != 512) return 1;
        resetSimtixOptions();
    }

    {
        gSimtixMode = true;
        const cl_image_format format{ CL_RGBA, CL_FLOAT };
        const cl_mem_object_type types[] = {
            CL_MEM_OBJECT_IMAGE1D, CL_MEM_OBJECT_IMAGE1D_BUFFER,
            CL_MEM_OBJECT_IMAGE2D, CL_MEM_OBJECT_IMAGE1D_ARRAY,
            CL_MEM_OBJECT_IMAGE3D, CL_MEM_OBJECT_IMAGE2D_ARRAY
        };
        for (size_t samples : { size_t(1), size_t(17), size_t(64) })
        {
            gSimtixSamples = samples;
            for (auto type : types)
            {
                size_t count = 0, sizes[2][3];
                get_max_sizes(&count, 2, sizes, 4096, 4096, 4096, 4096, 1 << 24,
                              1 << 26, type, &format, CL_TRUE);
                if (count != 2) return 1;
                for (size_t i = 0; i < count; ++i)
                    if (!sizes[i][0] || !sizes[i][1] || !sizes[i][2]
                        || sizes[i][0] * sizes[i][1] * sizes[i][2] > 64)
                        return 1;
            }
        }
        resetSimtixOptions();
    }

    return 0;
}
