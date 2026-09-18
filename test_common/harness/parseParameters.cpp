//
// Copyright (c) 2017 The Khronos Group Inc.
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
//
#include "parseParameters.h"

#include "errorHelpers.h"
#include "testHarness.h"
#include "ThreadPool.h"

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <sstream>
#include <sys/types.h>
#include <sys/stat.h>
#include <string.h>

#define DEFAULT_COMPILATION_PROGRAM "cl_offline_compiler"
#define DEFAULT_SPIRV_VALIDATOR "spirv-val"

CompilationMode gCompilationMode = kOnline;
CompilationCacheMode gCompilationCacheMode = kCacheModeCompileIfAbsent;
std::string gCompilationCachePath = ".";
std::string gCompilationProgram = DEFAULT_COMPILATION_PROGRAM;
bool gDisableSPIRVValidation = false;
std::string gSPIRVValidator = DEFAULT_SPIRV_VALIDATOR;
unsigned gNumWorkerThreads;
unsigned gNumThreadPoolThreads = 0;
bool gListTests = false;
bool gWimpyMode = false;
bool gSimtixMode = false;
size_t gSimtixSamples = 64;

int capSimtixNumElements(int num_elements, int minimum)
{
    if (!gSimtixMode || num_elements <= 0) return num_elements;

    const size_t floor = minimum > 0 ? static_cast<size_t>(minimum) : 1;
    const size_t limit = std::max(gSimtixSamples, floor);
    return limit >= static_cast<size_t>(num_elements)
        ? num_elements
        : static_cast<int>(limit);
}

unsigned capSimtixDimension(unsigned dimensions, unsigned max_dimension)
{
    if (!gSimtixMode || dimensions == 0 || max_dimension <= 1)
        return max_dimension;

    unsigned low = 1;
    unsigned high = max_dimension;
    while (low < high)
    {
        const unsigned middle = low + (high - low + 1) / 2;
        size_t work_items = 1;
        bool fits = true;
        for (unsigned dimension = 0; dimension < dimensions; ++dimension)
        {
            if (work_items > gSimtixSamples / middle)
            {
                fits = false;
                break;
            }
            work_items *= middle;
        }

        if (fits)
            low = middle;
        else
            high = middle - 1;
    }
    return low;
}

int capSimtixExponent(int max_exponent)
{
    if (!gSimtixMode || max_exponent <= 0) return max_exponent;

    int exponent = 0;
    size_t values = 1;
    while (exponent < max_exponent && values <= gSimtixSamples / 2)
    {
        values *= 2;
        ++exponent;
    }
    return exponent;
}

void helpInfo()
{
    log_info(
        R"(Common options:
    -h, --help
        This help
    --compilation-mode <mode>
        Specify a compilation mode.  Mode can be:
            online     Use online compilation (default)
            binary     Use binary offline compilation
            spir-v     Use SPIR-V offline compilation
    --num-worker-threads <num>
        Select parallel execution with the specified number of worker threads.
    --list
        List sub-tests
    --simtix
        Enable simulator mode (wimpy tests and one host worker/thread).
    --simtix-samples <num>
        Set the simulator workload/sample budget (default: 64).
    -w, --wimpy
        Enable wimpy mode. It does not impact all tests. Impacted tests will run
        with a very small subset of the tests. This option should not be used
        for conformance submission (default: disabled).
    -m, --disable-threadpool
        Disable multi-threading (using the ThreadPool API) within individual tests.
    -t, --num-threadpool-threads <num>
        Select the number of threads used by the ThreadPool API.
        default: All available threads

    --invalid-object-scenarios=<option_1>,<option_2>....
        Specify different scenarios to use when
        testing for object validity. Options can be:
           nullptr                 To use a nullptr (default)
           valid_object_wrong_type To use a valid_object which is not the correct type
        NOTE: valid_object_wrong_type option is not required for OpenCL conformance.

For offline compilation (binary and spir-v modes) only:
    --compilation-cache-mode <cache-mode>
        Specify a compilation caching mode:
            compile-if-absent
                Read from cache if already populated, or else perform
                offline compilation (default)
            force-read
                Force reading from the cache
            overwrite
                Disable reading from the cache
            dump-cl-files
                Dumps the .cl and build .options files used by the test suite
    --compilation-cache-path <path>
        Path for offline compiler output and CL source
    --compilation-program <prog>
        Program to use for offline compilation, defaults to:
            )" DEFAULT_COMPILATION_PROGRAM R"(

For spir-v mode only:
    --disable-spirv-validation
        Disable validation of SPIR-V using the SPIR-V validator
    --spirv-validator
        Path for SPIR-V validator, defaults to )" DEFAULT_SPIRV_VALIDATOR "\n"
        "\n");
}

static bool parsePositiveSize(const char *text, size_t &value)
{
    if (text == nullptr || text[0] == '\0' || text[0] == '-') return false;
    for (const char *p = text; *p != '\0'; ++p)
    {
        if (*p < '0' || *p > '9') return false;
    }

    errno = 0;
    char *end = nullptr;
    const unsigned long long parsed = strtoull(text, &end, 10);
    if (errno == ERANGE || end == text || *end != '\0' || parsed == 0
        || parsed > std::numeric_limits<size_t>::max()
        || parsed > std::numeric_limits<size_t>::max() / sizeof(float))
    {
        return false;
    }

    value = static_cast<size_t>(parsed);
    return true;
}

int parseCommonParamAndGetRemovedArgs(int argc, const char *argv[],
                                      std::vector<std::string> &removed_args,
                                      bool &help)
{
    int delArg = 0;
    help = false;
    bool simtixSamplesSpecified = false;

    for (int i = 1; i < argc; i++)
    {
        delArg = 0;
        size_t i_object_length = strlen("--invalid-object-scenarios=");

        if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
        {
            delArg++;
            if (!help)
            {
                help = true;
                removed_args.push_back("--help");
                helpInfo();
            }
        }
        else if (!strcmp(argv[i], "--list") || !strcmp(argv[i], "-list"))
        {
            delArg++;
            removed_args.push_back("--list");
            gListTests = true;
        }
        else if (!strcmp(argv[i], "--simtix"))
        {
            delArg++;
            removed_args.push_back("--simtix");
            gSimtixMode = true;
            gWimpyMode = true;
        }
        else if (!strcmp(argv[i], "--simtix-samples"))
        {
            delArg++;
            if ((i + 1) < argc)
            {
                delArg++;
                if (!parsePositiveSize(argv[i + 1], gSimtixSamples))
                {
                    log_error(
                        "A parameter to --simtix-samples must be a positive "
                        "integer that fits in the sample buffer.\n");
                    return -1;
                }
                simtixSamplesSpecified = true;
            }
            else
            {
                log_error(
                    "A parameter to --simtix-samples must be provided!\n");
                return -1;
            }
            removed_args.push_back(std::string(argv[i]) + " " + argv[i + 1]);
        }
        else if (!strcmp(argv[i], "--wimpy") || !strcmp(argv[i], "-w"))
        {
            delArg++;
            removed_args.push_back("--wimpy");
            gWimpyMode = true;
        }
        else if (!strcmp(argv[i], "-m")
                 || !strcmp(argv[i], "--disable-threadpool"))
        {
            if (gNumThreadPoolThreads != 0)
            {
                log_info(
                    "WARNING: -m/--disable-threadpool option overriding "
                    "previously set gNumThreadPoolThreads value of %d to 1.\n",
                    gNumThreadPoolThreads);
            }
            delArg++;
            removed_args.push_back("--disable-threadpool");
            gNumThreadPoolThreads = 1;
        }
        else if (!strcmp(argv[i], "-t")
                 || !strcmp(argv[i], "--num-threadpool-threads"))
        {
            delArg++;
            if ((i + 1) < argc)
            {
                delArg++;
                const char *numthstr = argv[i + 1];
                int new_value = atoi(numthstr);

                if (gNumThreadPoolThreads != 0)
                {
                    log_info("WARNING: -t/--num-threadpool-threads option "
                             "overriding "
                             "previously set gNumThreadPoolThreads value of %d "
                             "to %d.\n",
                             gNumThreadPoolThreads, new_value);
                }
                gNumThreadPoolThreads = new_value;
            }
            else
            {
                log_error("A parameter to --num-threadpool-threads must be "
                          "provided!\n");
                return -1;
            }
            removed_args.push_back(std::string(argv[i]) + " " + argv[i + 1]);
        }
        else if (!strcmp(argv[i], "--compilation-mode"))
        {
            delArg++;
            if ((i + 1) < argc)
            {
                delArg++;
                const char *mode = argv[i + 1];

                if (!strcmp(mode, "online"))
                {
                    gCompilationMode = kOnline;
                }
                else if (!strcmp(mode, "binary"))
                {
                    gCompilationMode = kBinary;
                }
                else if (!strcmp(mode, "spir-v"))
                {
                    gCompilationMode = kSpir_v;
                }
                else
                {
                    log_error("Compilation mode not recognized: %s\n", mode);
                    return -1;
                }
                log_info("Compilation mode specified: %s\n", mode);
            }
            else
            {
                log_error("Compilation mode parameters are incorrect. Usage:\n"
                          "  --compilation-mode <online|binary|spir-v>\n");
                return -1;
            }
            removed_args.push_back(std::string(argv[i]) + " " + argv[i + 1]);
        }
        else if (!strcmp(argv[i], "--num-worker-threads"))
        {
            delArg++;
            if ((i + 1) < argc)
            {
                delArg++;
                const char *numthstr = argv[i + 1];

                gNumWorkerThreads = atoi(numthstr);
            }
            else
            {
                log_error(
                    "A parameter to --num-worker-threads must be provided!\n");
                return -1;
            }
            removed_args.push_back(std::string(argv[i]) + " " + argv[i + 1]);
        }
        else if (!strcmp(argv[i], "--compilation-cache-mode"))
        {
            delArg++;
            if ((i + 1) < argc)
            {
                delArg++;
                const char *mode = argv[i + 1];

                if (!strcmp(mode, "compile-if-absent"))
                {
                    gCompilationCacheMode = kCacheModeCompileIfAbsent;
                }
                else if (!strcmp(mode, "force-read"))
                {
                    gCompilationCacheMode = kCacheModeForceRead;
                }
                else if (!strcmp(mode, "overwrite"))
                {
                    gCompilationCacheMode = kCacheModeOverwrite;
                }
                else if (!strcmp(mode, "dump-cl-files"))
                {
                    gCompilationCacheMode = kCacheModeDumpCl;
                }
                else
                {
                    log_error("Compilation cache mode not recognized: %s\n",
                              mode);
                    return -1;
                }
                log_info("Compilation cache mode specified: %s\n", mode);
            }
            else
            {
                log_error(
                    "Compilation cache mode parameters are incorrect. Usage:\n"
                    "  --compilation-cache-mode "
                    "<compile-if-absent|force-read|overwrite>\n");
                return -1;
            }
            removed_args.push_back(std::string(argv[i]) + " " + argv[i + 1]);
        }
        else if (!strcmp(argv[i], "--compilation-cache-path"))
        {
            delArg++;
            if ((i + 1) < argc)
            {
                delArg++;
                gCompilationCachePath = argv[i + 1];
            }
            else
            {
                log_error("Path argument for --compilation-cache-path was not "
                          "specified.\n");
                return -1;
            }
            removed_args.push_back(std::string(argv[i]) + " " + argv[i + 1]);
        }
        else if (!strcmp(argv[i], "--compilation-program"))
        {
            delArg++;
            if ((i + 1) < argc)
            {
                delArg++;
                gCompilationProgram = argv[i + 1];
            }
            else
            {
                log_error("Program argument for --compilation-program was not "
                          "specified.\n");
                return -1;
            }
            removed_args.push_back(std::string(argv[i]) + " " + argv[i + 1]);
        }
        else if (!strcmp(argv[i], "--disable-spirv-validation"))
        {
            delArg++;
            removed_args.push_back(argv[i]);
            gDisableSPIRVValidation = true;
        }
        else if (!strcmp(argv[i], "--spirv-validator"))
        {
            delArg++;
            if ((i + 1) < argc)
            {
                delArg++;
                gSPIRVValidator = argv[i + 1];
            }
            else
            {
                log_error("Program argument for --spirv-validator was not "
                          "specified.\n");
                return -1;
            }
            removed_args.push_back(std::string(argv[i]) + " " + argv[i + 1]);
        }
        else if (!strncmp(argv[i],
                          "--invalid-object-scenarios=", i_object_length))
        {
            if (strlen(argv[i]) > i_object_length)
            {
                delArg++;
                gInvalidObject = 0;
                std::string invalid_objects(argv[i]);

                if (invalid_objects.find("nullptr") != std::string::npos)
                {
                    gInvalidObject |= InvalidObject::Nullptr;
                }
                if (invalid_objects.find("valid_object_wrong_type")
                    != std::string::npos)
                {
                    gInvalidObject |= InvalidObject::ValidObjectWrongType;
                }
            }
            else
            {
                log_error("Program argument for --invalid-object-scenarios was "
                          "not specified.\n");
                return -1;
            }
            removed_args.push_back(argv[i]);
        }

        // cleaning parameters from argv tab
        for (int j = i; j < argc - delArg; j++) argv[j] = argv[j + delArg];
        argc -= delArg;
        i -= delArg;
    }

    if ((gCompilationCacheMode == kCacheModeForceRead
         || gCompilationCacheMode == kCacheModeOverwrite)
        && gCompilationMode == kOnline)
    {
        log_error("Compilation cache mode can only be specified when using an "
                  "offline compilation mode.\n");
        return -1;
    }

    if (gSimtixMode)
    {
        // Simulator runs must not create parallel host or in-test workloads.
        gWimpyMode = true;
        gNumWorkerThreads = 1;
        gNumThreadPoolThreads = 1;
    }
    else if (simtixSamplesSpecified)
    {
        log_error("--simtix-samples requires --simtix.\n");
        return -1;
    }

    return argc;
}

int parseCommonParam(int argc, const char *argv[])
{
    std::vector<std::string> unused1;
    bool unused2;
    return parseCommonParamAndGetRemovedArgs(argc, argv, unused1, unused2);
}

bool is_power_of_two(int number) { return number && !(number & (number - 1)); }

extern void parseWimpyReductionFactor(const char *&arg,
                                      int &wimpyReductionFactor)
{
    const char *arg_temp = strchr(&arg[1], ']');
    if (arg_temp != 0)
    {
        int new_factor = atoi(&arg[1]);
        arg = arg_temp; // Advance until ']'
        if (is_power_of_two(new_factor))
        {
            log_info("\n Wimpy reduction factor changed from %d to %d \n",
                     wimpyReductionFactor, new_factor);
            wimpyReductionFactor = new_factor;
        }
        else
        {
            log_info("\n WARNING: Incorrect wimpy reduction factor %d, must be "
                     "power of 2. The default value will be used.\n",
                     new_factor);
        }
    }
}
