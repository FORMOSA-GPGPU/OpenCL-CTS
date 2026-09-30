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
#include "testBase.h"
#include "harness/parseParameters.h"
#include "harness/typeWrappers.h"
#include "harness/testHarness.h"
#include "harness/conversions.h"

const char *test_kernels[] = { "__kernel void kernelA(__global uint *dst)\n"
                               "{\n"
                               "\n"
                               " dst[get_global_id(0)]*=3;\n"
                               "\n"
                               "}\n"
                               "__kernel void kernelB(__global uint *dst)\n"
                               "{\n"
                               "\n"
                               " dst[get_global_id(0)]++;\n"
                               "\n"
                               "}\n" };

#define TEST_SIZE    512
#define MAX_DEVICES 32
#define MAX_QUEUES 1000

static int test_device_set(size_t deviceCount, size_t queueCount,
                           cl_device_id *devices, int num_elements)
{
    int error;
    clContextWrapper context;
    clProgramWrapper program;
    clKernelWrapper kernels[2];
    clMemWrapper      stream;
    clCommandQueueWrapper queues[MAX_QUEUES] = {};
    size_t    threads[1], localThreads[1];
    const int testSize = capSimtixNumElements(TEST_SIZE);
    std::vector<cl_uint> data(testSize), outputData(testSize),
        expectedResults(testSize);
    std::vector<std::vector<cl_uint>> expectedResultsOneDevice(
        MAX_DEVICES, std::vector<cl_uint>(testSize));
    size_t i;

    if (gSimtixMode)
        queueCount = std::max(
            deviceCount,
            std::min(
                queueCount,
                size_t(std::max(2, capSimtixNumElements(MAX_QUEUES) / 16))));

    RandomSeed seed( gRandomSeed );

    if (deviceCount > MAX_DEVICES) {
        log_error("Number of devices in set (%zu) is greater than the number "
                  "for which the test was written (%d).",
                  deviceCount, MAX_DEVICES);
        return -1;
  }

  if (queueCount > MAX_QUEUES) {
      log_error("Number of queues (%zu) is greater than the number for which "
                "the test was written (%d).",
                queueCount, MAX_QUEUES);
      return -1;
  }

  log_info("Testing with %zu queues on %zu devices, %zu kernel executions.\n",
           queueCount, deviceCount,
           queueCount
               * (gSimtixMode ? std::max(1, num_elements / testSize)
                              : num_elements / testSize));

  for (i=0; i<deviceCount; i++) {
    char deviceName[4096] = "";
    error = clGetDeviceInfo(devices[i], CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);
    test_error(error, "clGetDeviceInfo CL_DEVICE_NAME failed");
    log_info("Device %zu is \"%s\".\n", i, deviceName);
  }

    /* Create a context */
    context = clCreateContext( NULL, (cl_uint)deviceCount, devices, notify_callback, NULL, &error );
    test_error( error, "Unable to create testing context" );

    /* Create our kernels (they all have the same arguments so we don't need multiple ones for each device) */
  if( create_single_kernel_helper( context, &program, &kernels[0], 1, test_kernels, "kernelA" ) != 0 )
  {
    return -1;
  }

  kernels[1] = clCreateKernel(program, "kernelB", &error);
  test_error(error, "clCreateKernel failed");


    /* Now create I/O streams */
  for (i = 0; i < size_t(testSize); i++) data[i] = genrand_int32(seed);

  stream = clCreateBuffer(context, CL_MEM_COPY_HOST_PTR,
                          sizeof(cl_uint) * testSize, data.data(), &error);
  test_error(error, "Unable to create test array");

  // Update the expected results
  for (i = 0; i < size_t(testSize); i++)
  {
      expectedResults[i] = data[i];
      for (size_t j = 0; j < deviceCount; j++)
          expectedResultsOneDevice[j][i] = data[i];
  }


  // Set the arguments
  error = clSetKernelArg( kernels[0], 0, sizeof( stream ), &stream);
  test_error( error, "Unable to set kernel arguments" );
  error = clSetKernelArg( kernels[1], 0, sizeof( stream ), &stream);
  test_error( error, "Unable to set kernel arguments" );

    /* Run the test */
  threads[0] = (size_t)testSize;

  error = get_max_common_work_group_size(context, kernels[0], threads[0],
                                         &localThreads[0]);
  test_error(error, "Unable to calc work group size");

  /* Create work queues */
  for (i = 0; i < queueCount; i++)
  {
      queues[i] =
          clCreateCommandQueue(context, devices[i % deviceCount], 0, &error);
      if (error != CL_SUCCESS || queues[i] == NULL)
      {
          log_info("Could not create queue[%d].\n", (int)i);
          queueCount = i;
          break;
      }
  }
  log_info("Testing with %d queues.\n", (int)queueCount);

    /* Enqueue executions */
  for (int z = 0; z < (gSimtixMode ? std::max(1, num_elements / testSize)
                                   : num_elements / testSize);
       z++)
  {
      for (i = 0; i < queueCount; i++)
      {
          // Randomly choose a kernel to execute.
          int kernel_selection = (int)get_random_float(0, 2, seed);
          error = clEnqueueNDRangeKernel(queues[i], kernels[kernel_selection],
                                         1, NULL, threads, localThreads, 0,
                                         NULL, NULL);
          test_error(error, "Kernel execution failed");

          // Update the expected results
          for (int j = 0; j < testSize; j++)
          {
              expectedResults[j] = (kernel_selection) ? expectedResults[j] + 1
                                                      : expectedResults[j] * 3;
              expectedResultsOneDevice[i % deviceCount][j] = (kernel_selection)
                  ? expectedResultsOneDevice[i % deviceCount][j] + 1
                  : expectedResultsOneDevice[i % deviceCount][j] * 3;
          }

          // Force the queue to finish so the next one will be in sync
          error = clFinish(queues[i]);
          test_error(error, "clFinish failed");
      }
  }

  /* Read results */
  int errors = 0;
  for (int q = 0; q<(int)queueCount; q++) {
      error = clEnqueueReadBuffer(queues[0], stream, CL_TRUE, 0,
                                  sizeof(cl_int) * testSize, outputData.data(),
                                  0, NULL, NULL);
      test_error(error, "Unable to get result data set");

      int errorsThisTime = 0;
      /* Verify all of the data now */
      for (i = 0; i < size_t(testSize); i++)
      {
          if (expectedResults[i] != outputData[i])
          {
              log_error(
                  "ERROR: Sample data did not verify for queue %d on device "
                  "%zu (sample %d, expected %d, got %d)\n",
                  q, q % deviceCount, (int)i, expectedResults[i],
                  outputData[i]);
              for (size_t j = 0; j < deviceCount; j++)
              {
                  if (expectedResultsOneDevice[j][i] == outputData[i])
                      log_info("Sample consistent with only device %zu having "
                               "modified the data.\n",
                               j);
              }
              errorsThisTime++;
              break;
          }
    }
    if (errorsThisTime)
      errors++;
  }

    /* All done now! */
  if (errors) return -1;
  return 0;
}

REGISTER_TEST(two_devices)
{
    cl_platform_id platform;
    cl_device_id devices[2];
    int err;
    cl_uint numDevices;

    err = clGetPlatformIDs(1, &platform, NULL);
    test_error( err, "Unable to get platform" );

    /* Get some devices */
    err = clGetDeviceIDs(platform,  CL_DEVICE_TYPE_ALL, 2, devices, &numDevices );
    test_error( err, "Unable to get 2 devices" );

    if( numDevices < 2 )
    {
        log_info( "WARNING: two device test unable to get two devices via CL_DEVICE_TYPE_ALL (got %d devices). Skipping test...\n", (int)numDevices );
        return 0;
    }
  else if (numDevices > 2)
  {
    log_info("Note: got %d devices, using just the first two.\n", (int)numDevices);
  }

    /* Run test */
    return test_device_set( 2, 2, devices, num_elements );
}

REGISTER_TEST(max_devices)
{
    cl_platform_id platform;
    cl_device_id devices[MAX_DEVICES];
    cl_uint deviceCount;
    int err;

    err = clGetPlatformIDs(1, &platform, NULL);
    test_error( err, "Unable to get platform" );

    /* Get some devices */
    err = clGetDeviceIDs(platform,  CL_DEVICE_TYPE_ALL, MAX_DEVICES, devices, &deviceCount );
    test_error( err, "Unable to get multiple devices" );

  log_info("Testing with %d devices.", deviceCount);

    /* Run test */
    return test_device_set( deviceCount, deviceCount, devices, num_elements );
}

REGISTER_TEST(hundred_queues)
{
    return test_device_set(1, 100, &device, num_elements);
}
