#include <math.h>
#define ceild(n,d)  ceil(((double)(n))/((double)(d)))
#define floord(n,d) floor(((double)(n))/((double)(d)))
#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))

/**
 * bicg.cu: This file is part of the PolyBench/GPU 1.0 test suite.
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
#include <sys/time.h>

#define POLYBENCH_TIME 1

#include "bicg.h"
#include <polybench.h>
#include <polybenchUtilFuncts.h>

//Error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.5

#define GPU_DEVICE 0

#ifndef M_PI
#define M_PI 3.14159
#endif

#define RUN_ON_CPU


void init_array(int nx, int ny, DATA_TYPE POLYBENCH_2D(A,NX,NY,nx,ny), DATA_TYPE POLYBENCH_1D(p,NY,ny), DATA_TYPE POLYBENCH_1D(r,NX,nx))
{
	int i, j;
	
	for (i = 0; i < ny; i++)
	{
    		p[i] = i * M_PI;
	}

	for (i = 0; i < nx; i++)
	{
    		r[i] = i * M_PI;

    		for (j = 0; j < ny; j++)
		{
      			A[i][j] = ((DATA_TYPE) i*j) / NX;
		}
 	}
}

void bicg_cpu(int nx, int ny, DATA_TYPE POLYBENCH_2D(A,NX,NY,nx,ny), DATA_TYPE POLYBENCH_1D(r,NX,nx), DATA_TYPE POLYBENCH_1D(s,NY,ny), 
		DATA_TYPE POLYBENCH_1D(p,NY,ny), DATA_TYPE POLYBENCH_1D(q,NX,nx))
{
	int i,j;

  	for (i = 0; i < _PB_NY; i++)
	{
		s[i] = 0.0;
	}

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
  int t1, t2, t3, t4, t5;
 register int lbv, ubv;
/* Start of CLooG code */
if (_PB_NX >= 1) {
  for (t2=0;t2<=floord(_PB_NY-1,32);t2++) {
    for (t3=0;t3<=floord(_PB_NX-1,32);t3++) {
      for (t4=32*t2;t4<=min(_PB_NY-1,32*t2+31);t4++) {
        for (t5=32*t3;t5<=min(_PB_NX-1,32*t3+31);t5++) {
          s[t4] = s[t4] + r[t5] * A[t5][t4];;
        }
      }
    }
  }
  for (t2=0;t2<=floord(_PB_NX-1,32);t2++) {
    for (t3=32*t2;t3<=min(_PB_NX-1,32*t2+31);t3++) {
      q[t3] = 0.0;;
    }
  }
  if (_PB_NY >= 1) {
    for (t2=0;t2<=floord(_PB_NX-1,32);t2++) {
      for (t3=0;t3<=floord(_PB_NY-1,32);t3++) {
        for (t4=32*t2;t4<=min(_PB_NX-1,32*t2+31);t4++) {
          for (t5=32*t3;t5<=min(_PB_NY-1,32*t3+31);t5++) {
            q[t4] = q[t4] + A[t4][t5] * p[t5];;
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
void print_array(int nx, int ny,
		 DATA_TYPE POLYBENCH_1D(s,NY,ny),
		 DATA_TYPE POLYBENCH_1D(q,NX,nx))

{
  int i;

  for (i = 0; i < ny; i++) {
    fprintf (stderr, DATA_PRINTF_MODIFIER, s[i]);
    if (i % 20 == 0) fprintf (stderr, "\n");
  }
  for (i = 0; i < nx; i++) {
    fprintf (stderr, DATA_PRINTF_MODIFIER, q[i]);
    if (i % 20 == 0) fprintf (stderr, "\n");
  }
  fprintf (stderr, "\n");
}

int main(int argc, char** argv)
{
	int nx = NX;
	int ny = NY;

	POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NX,NY,nx,ny);
	POLYBENCH_1D_ARRAY_DECL(s,DATA_TYPE,NY,ny);
	POLYBENCH_1D_ARRAY_DECL(q,DATA_TYPE,NX,nx);
	POLYBENCH_1D_ARRAY_DECL(p,DATA_TYPE,NY,ny);
	POLYBENCH_1D_ARRAY_DECL(r,DATA_TYPE,NX,nx);
	POLYBENCH_1D_ARRAY_DECL(s_outputFromGpu,DATA_TYPE,NY,ny);
	POLYBENCH_1D_ARRAY_DECL(q_outputFromGpu,DATA_TYPE,NX,nx);

	init_array(nx, ny, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(p), POLYBENCH_ARRAY(r));

	//GPU_argv_init();

	//bicgCuda(nx, ny, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(r), POLYBENCH_ARRAY(s), POLYBENCH_ARRAY(p), POLYBENCH_ARRAY(q), 
	//	POLYBENCH_ARRAY(s_outputFromGpu), POLYBENCH_ARRAY(q_outputFromGpu));

	#ifdef RUN_ON_CPU

		/* Start timer. */
	  	polybench_start_instruments;

		bicg_cpu(nx, ny, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(r), POLYBENCH_ARRAY(s), POLYBENCH_ARRAY(p), POLYBENCH_ARRAY(q));

		/* Stop and print timer. */
		printf("CPU Time in seconds:\n");
	  	polybench_stop_instruments;
	 	polybench_print_instruments;
	
		//compareResults(nx, ny, POLYBENCH_ARRAY(s), POLYBENCH_ARRAY(s_outputFromGpu), POLYBENCH_ARRAY(q), 
		//	POLYBENCH_ARRAY(q_outputFromGpu));

	#else //prevent dead code elimination

		polybench_prevent_dce(print_array(nx, ny, POLYBENCH_ARRAY(s_outputFromGpu), POLYBENCH_ARRAY(q_outputFromGpu)));
	
	#endif //RUN_ON_CPU


	POLYBENCH_FREE_ARRAY(A);
	POLYBENCH_FREE_ARRAY(r);
	POLYBENCH_FREE_ARRAY(s);
	POLYBENCH_FREE_ARRAY(p);
	POLYBENCH_FREE_ARRAY(q);
	POLYBENCH_FREE_ARRAY(s_outputFromGpu);
	POLYBENCH_FREE_ARRAY(q_outputFromGpu);

  	return 0;
}

#include <polybench.c>
