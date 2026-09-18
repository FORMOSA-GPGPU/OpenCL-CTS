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

#include <vector>

int gInvalidObject = 0;

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

    return 0;
}
