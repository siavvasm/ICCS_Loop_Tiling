#include <math.h>
#define ceild(n,d)  ceil(((double)(n))/((double)(d)))
#define floord(n,d) floor(((double)(n))/((double)(d)))
#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))

/**
 * fdtd2d.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <unistd.h>
#include <sys/time.h>

#define POLYBENCH_TIME 1

#include "fdtd2d.h"
#include <polybench.h>
#include <polybenchUtilFuncts.h>

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 10.05

#define GPU_DEVICE 0

#define RUN_ON_CPU


void init_arrays(int tmax, int nx, int ny, DATA_TYPE POLYBENCH_1D(_fict_, TMAX, TMAX), DATA_TYPE POLYBENCH_2D(ex,NX,NY,nx,ny), 
		DATA_TYPE POLYBENCH_2D(ey,NX,NY,nx,ny), DATA_TYPE POLYBENCH_2D(hz,NX,NY,nx,ny))
{
	int i, j;

  	for (i = 0; i < tmax; i++)
	{
		_fict_[i] = (DATA_TYPE) i;
	}
	
	for (i = 0; i < nx; i++)
	{
		for (j = 0; j < ny; j++)
		{
			ex[i][j] = ((DATA_TYPE) i*(j+1) + 1) / NX;
			ey[i][j] = ((DATA_TYPE) (i-1)*(j+2) + 2) / NX;
			hz[i][j] = ((DATA_TYPE) (i-9)*(j+4) + 3) / NX;
		}
	}
}


void runFdtd(int tmax, int nx, int ny, DATA_TYPE POLYBENCH_1D(_fict_, TMAX, TMAX), DATA_TYPE POLYBENCH_2D(ex,NX,NY,nx,ny), 
	DATA_TYPE POLYBENCH_2D(ey,NX,NY,nx,ny), DATA_TYPE POLYBENCH_2D(hz,NX,NY,nx,ny))
{
	int t, i, j;
	
/* Copyright (C) 1991-2018 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it and/or
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <http://www.gnu.org/licenses/>.  */
/* This header is separate from features.h so that the compiler can
   include it implicitly at the start of every compilation.  It must
   not itself include <features.h> or any other header that includes
   <features.h> because the implicit include comes before any feature
   test macros that may be defined in a source file before it first
   explicitly includes a system header.  GCC knows the name of this
   header in order to preinclude it.  */
/* glibc's intent is to support the IEC 559 math functionality, real
   and complex.  If the GCC (4.9 and later) predefined macros
   specifying compiler intent are available, use them to determine
   whether the overall intent is to support these features; otherwise,
   presume an older compiler has intent to support these features and
   define these macros by default.  */
/* wchar_t uses Unicode 10.0.0.  Version 10.0 of the Unicode Standard is
   synchronized with ISO/IEC 10646:2017, fifth edition, plus
   the following additions from Amendment 1 to the fifth edition:
   - 56 emoji characters
   - 285 hentaigana
   - 3 additional Zanabazar Square characters */
/* We do not support C11 <threads.h>.  */
  int t1, t2, t3, t4, t5, t6;
 register int lbv, ubv;
/* Start of CLooG code */
if ((_PB_NY >= 1) && (_PB_TMAX >= 1)) {
  for (t1=0;t1<=floord(_PB_TMAX-1,32);t1++) {
    for (t2=t1;t2<=min(floord(_PB_TMAX+_PB_NY-2,32),floord(32*t1+_PB_NY+30,32));t2++) {
      for (t3=t1;t3<=min(min(floord(32*t1+31*_PB_NX,32),floord(32*t1+_PB_NX+30,32)),floord(32*t1+30*_PB_TMAX+31*_PB_NX-30,992));t3++) {
        if ((_PB_NX >= 2) && (_PB_NY >= 2) && (t1 == t3)) {
          for (t4=max(32*t1,32*t2-_PB_NY+1);t4<=min(_PB_TMAX-1,32*t1+30);t4++) {
            if (t1 == t2) {
              ey[0][0] = _fict_[t4];;
              for (t6=t4+1;t6<=min(32*t1+31,t4+_PB_NX-1);t6++) {
                ey[(-t4+t6)][0] = ey[(-t4+t6)][0] - 0.5*(hz[(-t4+t6)][0] - hz[((-t4+t6)-1)][0]);;
              }
            }
            for (t5=max(32*t2,t4+1);t5<=min(32*t2+31,t4+_PB_NY-1);t5++) {
              ex[0][(-t4+t5)] = ex[0][(-t4+t5)] - 0.5*(hz[0][(-t4+t5)] - hz[0][((-t4+t5)-1)]);;
              ey[0][(-t4+t5)] = _fict_[t4];;
              for (t6=t4+1;t6<=min(32*t1+31,t4+_PB_NX-1);t6++) {
                ey[(-t4+t6)][(-t4+t5)] = ey[(-t4+t6)][(-t4+t5)] - 0.5*(hz[(-t4+t6)][(-t4+t5)] - hz[((-t4+t6)-1)][(-t4+t5)]);;
                ex[(-t4+t6)][(-t4+t5)] = ex[(-t4+t6)][(-t4+t5)] - 0.5*(hz[(-t4+t6)][(-t4+t5)] - hz[(-t4+t6)][((-t4+t5)-1)]);;
                hz[(-t4+t6-1)][(-t4+t5-1)] = hz[(-t4+t6-1)][(-t4+t5-1)] - 0.7*(ex[(-t4+t6-1)][((-t4+t5-1)+1)] - ex[(-t4+t6-1)][(-t4+t5-1)] + ey[((-t4+t6-1)+1)][(-t4+t5-1)] - ey[(-t4+t6-1)][(-t4+t5-1)]);;
              }
            }
          }
        }
        if ((_PB_NX >= 2) && (_PB_NY == 1) && (t1 == t2) && (t1 == t3)) {
          for (t4=32*t1;t4<=min(_PB_TMAX-1,32*t1+30);t4++) {
            ey[0][0] = _fict_[t4];;
            for (t6=t4+1;t6<=min(32*t1+31,t4+_PB_NX-1);t6++) {
              ey[(-t4+t6)][0] = ey[(-t4+t6)][0] - 0.5*(hz[(-t4+t6)][0] - hz[((-t4+t6)-1)][0]);;
            }
          }
        }
        if ((_PB_NX >= 2) && (t1 == t3) && (t1 <= min(floord(_PB_TMAX-32,32),t2-1))) {
          for (t5=32*t2;t5<=min(32*t2+31,32*t1+_PB_NY+30);t5++) {
            ex[0][(-32*t1+t5-31)] = ex[0][(-32*t1+t5-31)] - 0.5*(hz[0][(-32*t1+t5-31)] - hz[0][((-32*t1+t5-31)-1)]);;
            ey[0][(-32*t1+t5-31)] = _fict_[(32*t1+31)];;
          }
        }
        if ((_PB_NX >= 2) && (t1 == t2) && (t1 == t3) && (t1 <= floord(_PB_TMAX-32,32))) {
          ey[0][0] = _fict_[(32*t1+31)];;
        }
        if ((_PB_NX == 1) && (_PB_NY >= 2) && (t1 == t3)) {
          for (t4=max(32*t1,32*t2-_PB_NY+1);t4<=min(min(_PB_TMAX-1,32*t1+31),32*t2+30);t4++) {
            if (t1 == t2) {
              ey[0][0] = _fict_[t4];;
            }
            for (t5=max(32*t2,t4+1);t5<=min(32*t2+31,t4+_PB_NY-1);t5++) {
              ex[0][(-t4+t5)] = ex[0][(-t4+t5)] - 0.5*(hz[0][(-t4+t5)] - hz[0][((-t4+t5)-1)]);;
              ey[0][(-t4+t5)] = _fict_[t4];;
            }
          }
        }
        if ((_PB_NX == 1) && (_PB_NY >= 2) && (t1 == t2) && (t1 == t3) && (t1 <= floord(_PB_TMAX-32,32))) {
          ey[0][0] = _fict_[(32*t1+31)];;
        }
        if ((_PB_NX == 0) && (_PB_NY >= 2) && (t1 == t3)) {
          for (t4=max(32*t1,32*t2-_PB_NY+1);t4<=min(_PB_TMAX-1,32*t1+31);t4++) {
            for (t5=max(32*t2,t4);t5<=min(32*t2+31,t4+_PB_NY-1);t5++) {
              ey[0][(-t4+t5)] = _fict_[t4];;
            }
          }
        }
        if ((_PB_NX <= 1) && (_PB_NY == 1) && (t1 == t2) && (t1 == t3)) {
          for (t4=32*t1;t4<=min(_PB_TMAX-1,32*t1+31);t4++) {
            ey[0][0] = _fict_[t4];;
          }
        }
        if ((_PB_NY >= 2) && (t1 <= t3-1)) {
          for (t4=max(max(32*t1,32*t2-_PB_NY+1),32*t3-_PB_NX+1);t4<=min(min(_PB_TMAX-1,32*t1+31),32*t2+30);t4++) {
            if (t1 == t2) {
              for (t6=32*t3;t6<=min(32*t3+31,t4+_PB_NX-1);t6++) {
                ey[(-t4+t6)][0] = ey[(-t4+t6)][0] - 0.5*(hz[(-t4+t6)][0] - hz[((-t4+t6)-1)][0]);;
              }
            }
            for (t5=max(32*t2,t4+1);t5<=min(32*t2+31,t4+_PB_NY-1);t5++) {
              for (t6=32*t3;t6<=min(32*t3+31,t4+_PB_NX-1);t6++) {
                ey[(-t4+t6)][(-t4+t5)] = ey[(-t4+t6)][(-t4+t5)] - 0.5*(hz[(-t4+t6)][(-t4+t5)] - hz[((-t4+t6)-1)][(-t4+t5)]);;
                ex[(-t4+t6)][(-t4+t5)] = ex[(-t4+t6)][(-t4+t5)] - 0.5*(hz[(-t4+t6)][(-t4+t5)] - hz[(-t4+t6)][((-t4+t5)-1)]);;
                hz[(-t4+t6-1)][(-t4+t5-1)] = hz[(-t4+t6-1)][(-t4+t5-1)] - 0.7*(ex[(-t4+t6-1)][((-t4+t5-1)+1)] - ex[(-t4+t6-1)][(-t4+t5-1)] + ey[((-t4+t6-1)+1)][(-t4+t5-1)] - ey[(-t4+t6-1)][(-t4+t5-1)]);;
              }
            }
          }
        }
        if ((_PB_NY >= 2) && (t1 == t2) && (t1 <= min(floord(_PB_TMAX-32,32),t3-1))) {
          for (t6=32*t3;t6<=min(32*t3+31,32*t1+_PB_NX+30);t6++) {
            ey[(-32*t1+t6-31)][0] = ey[(-32*t1+t6-31)][0] - 0.5*(hz[(-32*t1+t6-31)][0] - hz[((-32*t1+t6-31)-1)][0]);;
          }
        }
        if ((_PB_NY == 1) && (t1 == t2) && (t1 <= t3-1)) {
          for (t4=max(32*t1,32*t3-_PB_NX+1);t4<=min(_PB_TMAX-1,32*t1+31);t4++) {
            for (t6=32*t3;t6<=min(32*t3+31,t4+_PB_NX-1);t6++) {
              ey[(-t4+t6)][0] = ey[(-t4+t6)][0] - 0.5*(hz[(-t4+t6)][0] - hz[((-t4+t6)-1)][0]);;
            }
          }
        }
      }
      if (_PB_NX <= -1) {
        for (t4=max(32*t1,32*t2-_PB_NY+1);t4<=min(_PB_TMAX-1,32*t1+31);t4++) {
          for (t5=max(32*t2,t4);t5<=min(32*t2+31,t4+_PB_NY-1);t5++) {
            ey[0][(-t4+t5)] = _fict_[t4];;
          }
        }
      }
    }
  }
}
/* End of CLooG code */
	
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int nx,
		 int ny,
		 DATA_TYPE POLYBENCH_2D(hz,NX,NY,nx,ny))
{
  int i, j;

  for (i = 0; i < nx; i++)
    for (j = 0; j < ny; j++) {
         fprintf(stderr, DATA_PRINTF_MODIFIER, hz[i][j]);
      if ((i * nx + j) % 20 == 0) fprintf(stderr, "\n");
    }
  fprintf(stderr, "\n");
}


int main(int argc, char *argv[])
{
	int tmax = TMAX;
	int nx = NX;
	int ny = NY;

	POLYBENCH_1D_ARRAY_DECL(_fict_,DATA_TYPE,TMAX,TMAX);
	POLYBENCH_2D_ARRAY_DECL(ex,DATA_TYPE,NX,NY,nx,ny);
	POLYBENCH_2D_ARRAY_DECL(ey,DATA_TYPE,NX,NY,nx,ny);
	POLYBENCH_2D_ARRAY_DECL(hz,DATA_TYPE,NX,NY,nx,ny);
	POLYBENCH_2D_ARRAY_DECL(hz_outputFromGpu,DATA_TYPE,NX,NY,nx,ny);

	init_arrays(tmax, nx, ny, POLYBENCH_ARRAY(_fict_), POLYBENCH_ARRAY(ex), POLYBENCH_ARRAY(ey), POLYBENCH_ARRAY(hz));

	//GPU_argv_init();
	//fdtdCuda(tmax, nx, ny, POLYBENCH_ARRAY(_fict_), POLYBENCH_ARRAY(ex), POLYBENCH_ARRAY(ey), POLYBENCH_ARRAY(hz), POLYBENCH_ARRAY(hz_outputFromGpu));

	#ifdef RUN_ON_CPU

		/* Start timer. */
	  	polybench_start_instruments;

		runFdtd(tmax, nx, ny, POLYBENCH_ARRAY(_fict_), POLYBENCH_ARRAY(ex), POLYBENCH_ARRAY(ey), POLYBENCH_ARRAY(hz));

		/* Stop and print timer. */
		printf("CPU Time in seconds:\n");
	  	polybench_stop_instruments;
	 	polybench_print_instruments;
		
		//compareResults(nx, ny, POLYBENCH_ARRAY(hz), POLYBENCH_ARRAY(hz_outputFromGpu));

	#else //prevent dead code elimination

		polybench_prevent_dce(print_array(nx, ny, POLYBENCH_ARRAY(hz_outputFromGpu)));

	#endif //RUN_ON_CPU


	POLYBENCH_FREE_ARRAY(_fict_);
	POLYBENCH_FREE_ARRAY(ex);
	POLYBENCH_FREE_ARRAY(ey);
	POLYBENCH_FREE_ARRAY(hz);
	POLYBENCH_FREE_ARRAY(hz_outputFromGpu);

	return 0;
}

#include <polybench.c>
