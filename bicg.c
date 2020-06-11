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

	for (i = 0; i < _PB_NX; i++)
	{
		q[i] = 0.0;
		for (j = 0; j < _PB_NY; j++)
	  	{
	    		s[j] = s[j] + r[i] * A[i][j];
	    		q[i] = q[i] + A[i][j] * p[j];
	  	}
	}


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