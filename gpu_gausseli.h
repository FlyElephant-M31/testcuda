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

#ifndef GPU_GAUSSELI_H
#define GPU_GAUSSELI_H

#include <vector>
#include "float.h"

void gpu_init(int argc, char **argv);

void gpu_solve(const Matrix &A, Vector &x, const Vector &b, Matrix &finalA, Vector &finalb);

#endif
