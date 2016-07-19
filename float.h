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

#ifndef float_h
#define float_h
#include <cstddef>
#include <vector>

typedef float myfloat_t;

typedef std::vector<myfloat_t> Vector;

class Matrix: public std::vector<myfloat_t>
{
  std::size_t _size1, _size2;

public:
  Matrix(std::size_t __s1, std::size_t __s2):
    std::vector<myfloat_t>(__s1*__s2), _size1(__s1), _size2(__s2) {}

  myfloat_t &get(std::size_t i, std::size_t j)       { return (*this)[i*_size2 + j]; }
  myfloat_t  get(std::size_t i, std::size_t j) const { return (*this)[i*_size2 + j]; }

  std::size_t size_x() const { return _size1; }
  std::size_t size_y() const { return _size2; }
};

#endif
