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

#ifndef TIMING_H
#define TIMING_H
#include <sys/time.h>
#include <sys/resource.h>

inline double user_secs() {
  struct rusage rus;
  getrusage(RUSAGE_SELF, &rus);
  return (rus.ru_utime.tv_sec + rus.ru_stime.tv_sec) + 1e-6*(rus.ru_utime.tv_usec + rus.ru_stime.tv_usec);
}

inline double wall_secs() {
  double mysecs;
  struct timeval tp;
  struct timezone tzp;
  gettimeofday (&tp, &tzp);
  mysecs  = tp.tv_sec;
  mysecs += tp.tv_usec *1e-6;
  return mysecs;
}

#endif
