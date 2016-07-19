/*
  Copyright (C) 2010 Axel Arnold, ICP, University of Stuttgart, Pfaffenwaldring 27, 70569 Germany
  
  This file is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.
  
  This file is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  
  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>. 
*/

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <unistd.h>

#include "float.h"
#include "timing.h"
#include "cpu_gausseli.h"
#include "gpu_gausseli.h"

myfloat_t check(const char *who, const Matrix &A, const Vector &x, const Vector &b, bool v)
{
  myfloat_t res = 0;
  const int end = x.size();

  for (int r = 0; r < end; ++r) {
    myfloat_t cr = b[r];
    if (v) printf("%f ", b[r]);
    for (int s = 0; s < end; ++s) {
      cr -= A.get(r, s) *x[s];

      if (v) printf("- %f * %f ", A.get(r, s), x[s]);
    }
    if (v) printf("= %f (!= 0)\n", cr);
    if (fabs(cr) > res) res = fabs(cr);
  }
  printf("maximal error of %s: %f\n", who, res);
  return res;
}

void compare(const Matrix &A, const Matrix &A2,
	     const Vector &x, const Vector &x2,
	     const Vector &b, const Vector &b2, bool v)
{
  const int end = x.size();

  myfloat_t max_dev = 0, dev;
  for (int r = 0; r < end; ++r) {
    for (int c = 0; c < end; ++c) {
      if (c >= r) {
	myfloat_t dev = A.get(r, c) - A2.get(r, c);
	if (v) printf("%+9.5f ", dev);
	if (fabs(dev) > max_dev) max_dev = fabs(dev);
      }
      else if (v) printf("          ");
    }
    dev = x[r] - x2[r];
    if (v) printf(")*( %+9.5f ", dev);
    if (fabs(dev) > max_dev) max_dev = fabs(dev);

    dev = b[r] - b2[r];
    if (v) printf(")=( %+9.5f\n", dev);
    if (fabs(dev) > max_dev) max_dev = fabs(dev);
  }
  printf("maximal difference: %g\n", max_dev);
}

int main(int argc, char **argv) {
  int size = 3,
    rounds = 1;
  int seed = getpid();
  bool verbose = false;
  bool debug   = false;
  bool gpuonly = false;

  gpu_init(argc, argv);

  int opt;
  while ((opt = getopt(argc, argv, "s:r:i:vcg")) != -1) {
    switch (opt) {
    case 's':   size = atol(optarg); break;
    case 'r': rounds = atol(optarg); break;
    case 'i': seed   = atol(optarg); break;
    case 'v': verbose = true; break;
    case 'c': debug   = true; break;
    case 'g': gpuonly = true; break;
    case ':': fprintf(stderr, "parameter missing\n"); exit(-1);
    case '?': fprintf(stderr, "unknown option\n"); exit(-1);
    }
  }

  srand48(seed);

  double ttotCPU = 0, ttotGPU = 0;
  double tstart;
  for (int r = 0; r < rounds; ++r) {
    fprintf(stderr, "**** generating problem of size %d\n", size);

    // generate the problem
    Matrix A(size, size), finalA(size, size), finalA2(size, size);
    Vector b(size), finalb(size), finalb2(size);
    Vector x(size), x2(size);
    for (int i = 0; i < size; ++i) b[i] = drand48();
    for (int i = 0; i < size; ++i)
      for (int j = 0; j < size; ++j) {
	A.get(i, j) = drand48() - 0.5;
	if (i==j) A.get(i, j) += 0.1*size;
      }

    if (!gpuonly) {
      fprintf(stderr, "**** running on the CPU\n");

      tstart = wall_secs();
      cpu_solve(A, x, b, finalA, finalb);
      ttotCPU += wall_secs() - tstart;
      
      check("CPU", A, x, b, verbose);
    }

    fprintf(stderr, "**** running on the GPU\n");

    tstart = wall_secs();
    gpu_solve(A, x2, b, finalA2, finalb2);
    ttotGPU += wall_secs() - tstart;

    check("GPU", A, x2, b, verbose);

    if (debug && !gpuonly) {
      fprintf(stderr, "**** compare\n");
      compare(finalA, finalA2, x, x2, finalb, finalb2, verbose);
    }
  }

  if (!gpuonly) {
    printf("CPU solver time: %f s total\n", ttotCPU);
    printf("CPU solver %f GFlops\n", .66e-9*pow(size, 3)/(ttotCPU/rounds));
  }
  printf("GPU solver time: %f s total\n", ttotGPU);
  printf("GPU solver %f GFlops\n", .66e-9*pow(size, 3)/(ttotGPU/rounds));
  return 1;
}
