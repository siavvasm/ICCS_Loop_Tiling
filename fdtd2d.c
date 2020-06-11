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
	

	for (t=0; t < _PB_TMAX; t++)  
	{
		for (j=0; j < _PB_NY; j++)
		{
			ey[0][j] = _fict_[t];
		}
	
		for (i = 1; i < _PB_NX; i++)
		{
       		for (j = 0; j < _PB_NY; j++)
			{
       			ey[i][j] = ey[i][j] - 0.5*(hz[i][j] - hz[(i-1)][j]);
        		}
		}

		for (i = 0; i < _PB_NX; i++)
		{
       		for (j = 1; j < _PB_NY; j++)
			{
				ex[i][j] = ex[i][j] - 0.5*(hz[i][j] - hz[i][(j-1)]);
			}
		}

		for (i = 0; i < _PB_NX-1; i++)
		{
			for (j = 0; j < _PB_NY-1; j++)
			{
				hz[i][j] = hz[i][j] - 0.7*(ex[i][(j+1)] - ex[i][j] + ey[(i+1)][j] - ey[i][j]);
			}
		}
	}

	
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