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

#include "cpu_gausseli.h"
#include <cmath>
#include <cstdio>
#ifdef HAVE_OMP
#include <omp.h>
#endif

using namespace std;

inline void print_state(const Matrix &A, const Vector &b)
{
  printf("----------------------------\n");    
  for (size_t r = 0; r < b.size(); ++r) {
    for (size_t s = 0; s < b.size(); ++s)
      printf("%f ", A.get(r, s));
    printf("= %f\n", b[r]);    
  }
  printf("----------------------------\n");    
}

void cpu_solve(const Matrix &A_in, Vector &x,
	       const Vector &b_in, Matrix &A, Vector &b)
{
  const int bottom = b.size() - 1;
  const int N = b.size();

  A = A_in; b = b_in;

  // forward elimination
  for (int i = 0; i < N; i++) {
#pragma omp parallel for
    for (int j = i + 1; j < N; j++) {
      myfloat_t p = A.get(j, i) / A.get(i, i);

      b[j] -= b[i] * p;
      for (int k = i+1; k < N; k++)
	A.get(j, k) -= A.get(i, k) * p;
    } 
  }

  // backward substitution

  for (int r = bottom; r >= 0; --r) {
    myfloat_t t = b[r];
    for (int s = bottom; s > r; --s)
      t -= A.get(r, s)*x[s];
    x[r] = t / A.get(r, r);
  }
}
