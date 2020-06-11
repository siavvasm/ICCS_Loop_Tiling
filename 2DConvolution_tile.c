#include <math.h>
#define ceild(n,d)  ceil(((double)(n))/((double)(d)))
#define floord(n,d) floor(((double)(n))/((double)(d)))
#define max(x,y)    ((x) > (y)? (x) : (y))
#define min(x,y)    ((x) < (y)? (x) : (y))

/**
 * 2DConvolution.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Will Killian <killian@udel.edu>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

#define POLYBENCH_TIME 1

#include "2DConvolution.h"
#include <polybench.h>
#include <polybenchUtilFuncts.h>

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

#define RUN_ON_CPU


void conv2D(int ni, int nj, DATA_TYPE POLYBENCH_2D(A, NI, NJ, ni, nj), DATA_TYPE POLYBENCH_2D(B, NI, NJ, ni, nj))
{
	int i, j;
	DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;


	c11 = +0.2;  c21 = +0.5;  c31 = -0.8;
	c12 = -0.3;  c22 = +0.6;  c32 = -0.9;
	c13 = +0.4;  c23 = +0.7;  c33 = +0.10;

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
  int t1, t2, t3, t4;
 register int lbv, ubv;
/* Start of CLooG code */
if ((_PB_NI >= 3) && (_PB_NJ >= 3)) {
  for (t1=0;t1<=floord(_PB_NI-2,32);t1++) {
    for (t2=0;t2<=floord(_PB_NJ-2,32);t2++) {
      for (t3=max(1,32*t1);t3<=min(_PB_NI-2,32*t1+31);t3++) {
        for (t4=max(1,32*t2);t4<=min(_PB_NJ-2,32*t2+31);t4++) {
          B[t3][t4] = c11 * A[(t3 - 1)][(t4 - 1)] + c12 * A[(t3 + 0)][(t4 - 1)] + c13 * A[(t3 + 1)][(t4 - 1)] + c21 * A[(t3 - 1)][(t4 + 0)] + c22 * A[(t3 + 0)][(t4 + 0)] + c23 * A[(t3 + 1)][(t4 + 0)] + c31 * A[(t3 - 1)][(t4 + 1)] + c32 * A[(t3 + 0)][(t4 + 1)] + c33 * A[(t3 + 1)][(t4 + 1)];;
        }
      }
    }
  }
}
/* End of CLooG code */

}



void init(int ni, int nj, DATA_TYPE POLYBENCH_2D(A, NI, NJ, ni, nj))
{
	int i, j;

	for (i = 0; i < ni; ++i)
    	{
		for (j = 0; j < nj; ++j)
		{
			A[i][j] = (float)rand()/RAND_MAX;
        	}
    	}
}

/* DCE code. Must scan the entire live-out data.
   Can be used also to check the correctness of the output. */
static
void print_array(int ni, int nj,
		 DATA_TYPE POLYBENCH_2D(B,NI,NJ,ni,nj))
{
  int i, j;

  for (i = 0; i < ni; i++)
    for (j = 0; j < nj; j++) {
	fprintf (stderr, DATA_PRINTF_MODIFIER, B[i][j]);
	if ((i * ni + j) % 20 == 0) fprintf (stderr, "\n");
    }
  fprintf (stderr, "\n");
}


int main(int argc, char *argv[])
{
	/* Retrieve problem size */
	int ni = NI;
	int nj = NJ;

	POLYBENCH_2D_ARRAY_DECL(A,DATA_TYPE,NI,NJ,ni,nj);
  	POLYBENCH_2D_ARRAY_DECL(B,DATA_TYPE,NI,NJ,ni,nj);
  	POLYBENCH_2D_ARRAY_DECL(B_outputFromGpu,DATA_TYPE,NI,NJ,ni,nj);

	//initialize the arrays
	init(ni, nj, POLYBENCH_ARRAY(A));
	
	//GPU_argv_init();

	//convolution2DCuda(ni, nj, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(B_outputFromGpu));

	#ifdef RUN_ON_CPU
	
	 	/* Start timer. */
	  	polybench_start_instruments;

		conv2D(ni, nj, POLYBENCH_ARRAY(A), POLYBENCH_ARRAY(B));

		/* Stop and print timer. */
		printf("CPU Time in seconds:\n");
	  	polybench_stop_instruments;
	 	polybench_print_instruments;
	
		//compareResults(ni, nj, POLYBENCH_ARRAY(B), POLYBENCH_ARRAY(B_outputFromGpu));

	#else //prevent dead code elimination

		polybench_prevent_dce(print_array(ni, nj, POLYBENCH_ARRAY(B_outputFromGpu)));

	#endif //RUN_ON_CPU


	POLYBENCH_FREE_ARRAY(A);
  	POLYBENCH_FREE_ARRAY(B);
	POLYBENCH_FREE_ARRAY(B_outputFromGpu);
	
	return 0;
}

#include <polybench.c>
