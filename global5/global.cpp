#include "hip/hip_runtime.h"
#include <stdio.h>
#include <stdint.h>
#include <assert.h>
#include "repeat.h"

#define SET_BIASING 1

// Radeon VII main memory size
const long long GPU_MEM_SIZE (16 * 1024 * 1024 * 1024ll);

const int page_size = 4;        // Scale stride and arrays by page size.
const int elt_size = 8;         // since addresses are 64 bits, elements are 8 bytes each


__global__ void global_latency (unsigned long long **my_array, int array_length, int iterations, int ignore_iterations, unsigned long long * duration) {
    unsigned int start_time, end_time;
    unsigned long long *j = (unsigned long long*)my_array;
    volatile unsigned long long sum_time;

    sum_time = 0;
    duration[0] = 0;
    asm volatile("s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t");

    for (int k = -ignore_iterations; k < iterations; k++) {
        if (k==0) {
            sum_time = 0; // ignore some iterations: cold icache misses
        }

        start_time = clock();
        asm volatile("s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t");
        // There are some array lengths that you will need to increase this number for, as doing 256 strides might not be enough for the experiment
        // In this case, change here and in the final timing calculation (last statement of parametric_measure_global)
        repeat256(j=*(unsigned long long **)j;)
        end_time = clock();
        asm volatile("s_waitcnt vmcnt(0) & lgkmcnt(0)\n\t");
        sum_time += (end_time - start_time);
    }

    ((unsigned long long*)my_array)[array_length] = (unsigned long long)j;
    ((unsigned long long*)my_array)[array_length+1] = (unsigned long long) sum_time;
    duration[0] = sum_time;
}

int gcf(int a, int b)
{
    if (a == 0) return b;
    return gcf(b % a, a);
}

/* Construct an array of N unsigned long longs, with array elements initialized
   so kernel will make stride accesses to the array. Then launch kernel
   10 times, each making iterations*256 global memory accesses. */
// stride: in number of elements. Eg stride=1 will move at a stride of 1 unsigned long long, or 8 bytes
// N: number of elements in array
// ignore_iterations: cache warmup, how many passes through the kernel we run before starting to time
// iterations: how many passes through kernerl we do (after ignore_iterations)
void parametric_measure_global(long long N, int iterations, int ignore_iterations, int stride) {
    long long i;
    unsigned long long * h_a;
    unsigned long long ** d_a;

    unsigned long long * duration;
    unsigned long long * latency;
    unsigned long long latency_sum = 0;

    //printf("N: %lld \n", N);
    // Don't die if too much memory was requested.
    if (N > GPU_MEM_SIZE) { printf ("OOM.\n"); return; }

    /* allocate arrays on CPU */
    // Allocate arrays on CPU with 2 MB alignment
    posix_memalign((void**)&h_a, 2 * 1024 * 1024, (N + 2) * sizeof(unsigned long long));
    //h_a = (unsigned long long *)calloc((N+2), sizeof(unsigned long long)); // TODO page align
    // unsigned long long *h_a_temp = (unsigned long long *)malloc(((N + 2) * sizeof(unsigned long long)) + 0x1000);
    //h_a = (unsigned long long *)(((((uintptr_t)h_a_temp) >> 12) << 12) + 0x1000);

    latency = (unsigned long long *)calloc(1, sizeof(unsigned long long));


    /* allocate arrays on GPU */
    hipError_t error_id = hipMallocManaged((void **) &d_a, sizeof(unsigned long long) * (N+2));
    if (error_id != hipSuccess) {
        printf("Error is %s\n", hipGetErrorString(error_id));
    }
    hipMalloc((void **) &duration, sizeof(unsigned long long));
    //printf("[GLOBAL] Base Address of device array: 0x%lld\n", (unsigned long long)(unsigned long long *)d_a);


#if SET_BIASING
    /* initialize array elements on CPU with pointers into d_a.*/
    int step = gcf(stride, N);    // Optimization: Initialize fewer elements.
    long long index;
    int count = 1;
    unsigned long long bias = 16; // add to base address to hopefully hit in different d$ sets. in bytes
    //  printf("h_a[index]\t\t&d_a[index]\t\tindex\n");

    for (i = 0; i < N; i += step) {
        // last iteration
        if((count % 2 == 1) && (i + step >= N)){
            index = i;
            if(index>=N) {
                printf("if2 : %lld\n", index);
            }
            assert(index<N);
            h_a[index] = ((unsigned long long)(unsigned long long *)d_a) + ((i + stride) % N)*sizeof(unsigned long long);
        }

            /**
            // second to last iteration
            else if(i + step + step >= N){
                index = (count % 2) ? i : ((i + bias) % N);
                if(index>=N) {
                    printf("if1 : %lld\n", index);
                }
                assert(index<N);
                //h_a[index] = ((unsigned long long)(unsigned long long *)d_a) + ((i + stride) % N)*sizeof(unsigned long long);
                h_a[index] = (count % 2) ? ((unsigned long long) (unsigned long long *) d_a) +
                                           ((i + stride + bias) % N) * sizeof(unsigned long long)
                                         : ((unsigned long long) (unsigned long long *) d_a) +
                                           ((i + stride) % N) * sizeof(unsigned long long);
            }
    **/
        else {
            // Device pointers are 64-bits
            index = (count % 2) ? i : ((i + bias)%N); // changed
            if(index>=N) {
                printf("if3 : %lld\n", index);
            }
            assert(index<N);
            h_a[index] = (count % 2) ? ((unsigned long long) (unsigned long long *) d_a) +
                                       ((i + stride + bias) % N) * sizeof(unsigned long long)
                                     : ((unsigned long long) (unsigned long long *) d_a) +
                                       ((i + stride) % N) * sizeof(unsigned long long);
        }

        // add condition if there is only one element being updated (step>=N) then h_a[index] = d_a[index];
        if(step>=N) h_a[index] = ((unsigned long long)(unsigned long long *)d_a) + (i % N)*sizeof(unsigned long long);

  //         printf("0x%llx\t\t0x%llx\t\th[%lld]\n", (unsigned long long)(unsigned long long *)h_a[index], (unsigned long long)(unsigned long long *)&d_a[index],index);
        //      fflush(stdout);
        count++;
    }
#else
    // initialize array elements on CPU with pointers into d_a.
  int step = gcf(stride, N);   // Optimization: Initialize fewer elements.
  for (i = 0; i < N; i += step) {
    // Device pointers are 64-bits
    h_a[i] = ((unsigned long long)(unsigned long long *)d_a) + ((i + stride) % N)*sizeof(unsigned long long);   // set
  }
#endif

    h_a[N] = 0;
    h_a[N+1] = 0;

    error_id = hipDeviceSynchronize();
    if (error_id != hipSuccess) {
        printf("Error is %s\n", hipGetErrorString(error_id));
    }

    /* copy array elements from CPU to GPU */
    error_id = hipMemcpy((void *)d_a, (void *)h_a, sizeof(unsigned long long) * (N+2), hipMemcpyHostToDevice);
    if (error_id != hipSuccess) {
        printf("Error is %s\n", hipGetErrorString(error_id));
    }



    /* Launch a multiple of 10 iterations of the same kernel and take the average to eliminate interconnect (TPCs) effects */
    int l;
    for (l=0; l <10; l++) {
        /* launch kernel*/
        dim3 Db = dim3(1);
        dim3 Dg = dim3(1,1,1);

        //printf("Launch kernel with parameters: %d, N: %lld, stride: %d\n", iterations, N, stride);
        hipLaunchKernelGGL(global_latency, dim3(Dg), dim3(Db), 0, 0, d_a, N, iterations, ignore_iterations, duration);

        error_id = hipDeviceSynchronize();
        if (error_id != hipSuccess) {
            printf("Error is %s\n", hipGetErrorString(error_id));
        }

        /* copy results from GPU to CPU */

        //hipMemcpy((void *)h_a, (void *)d_a, sizeof(unsigned long long) * (N+2), hipMemcpyDeviceToHost);
        hipMemcpy((void *)latency, (void *)duration, sizeof(unsigned long long), hipMemcpyDeviceToHost);
        unsigned long long old_latency = latency_sum;
        latency_sum+=latency[0];
        assert(latency_sum >old_latency);
    }

    /* free memory on GPU */
    hipFree(d_a);
    hipFree(duration);

    /*free memory on CPU */
    free(h_a);
    free(latency);

    // added debug print
    //printf("l: %d, iterations: %d, constant: 256, latency_sum: %llu\n", l, iterations, latency_sum);
    printf("%f\n", (double)(latency_sum/(l*256.0*iterations)) ); // 10 for l<10, 256 for repeat256
}

/* Test page size. Construct an access pattern of N elements spaced stride apart,
   followed by a gap of stride+offset, followed by N more elements spaced stride
   apart. */
void measure_pagesize(long long N, int stride, int offset) {
    unsigned long long * h_a;
    unsigned long long ** d_a;

    unsigned long long * duration;
    unsigned long long * latency;

    unsigned long long latency_sum = 0;

    const long long size = N * stride * 2 + offset + stride*2;
    const int iterations = 20;

    // Don't die if too much memory was requested.
    if (size > GPU_MEM_SIZE) { printf ("OOM.\n"); return; }

    /* allocate array on CPU */
    h_a = (unsigned long long *)calloc(size, sizeof(unsigned long long));
    latency = (unsigned long long *)calloc(1, sizeof(unsigned long long));

    /* allocate array on GPU */
    hipMalloc ((void **) &d_a, sizeof(unsigned long long) * size);
    hipMalloc ((void **) &duration, sizeof(unsigned long long));

    /* initialize array elements on CPU */
    for (int i=0;i<N; i++) {
        h_a[i * stride] = ((unsigned long long)(unsigned long long*)d_a) + ((i*stride + stride)*elt_size);
        //h_a[i * stride] = (unsigned long long)&d_a[(i*stride + stride)];
        fprintf(stdout, "Base d_a: %p, h_a[%d]: 0x%llx\n", (unsigned long long*)d_a, i, h_a[i]);
    }

    // point last element to stride+offset
    h_a[(N-1)*stride] = ((unsigned long long)(unsigned long long*)d_a) + ((N*stride + offset)*elt_size);
    //fprintf(stdout, "Base d_a: %p, h_a[%lld]: 0x%llx\n", (unsigned long long*)d_a, (N-1)*stride, h_a[(N-1)*stride]);

    for (int i=0;i<N; i++) {
        h_a[(i+N)*stride+offset] = ((unsigned long long)(unsigned long long*)d_a) + (((i+N)*stride + offset + stride)*elt_size);
        //fprintf(stdout, "Base d_a: %p, h_a[%d]: 0x%llx\n", (unsigned long long*)d_a, i, h_a[i]);
    }

    // wrap around
    h_a[(2*N-1)*stride+offset] = ((unsigned long long)(unsigned long long*)d_a);
    fprintf(stdout, "Base d_a: %p, h_a[%lld]: 0x%llx\n", (unsigned long long*)d_a, (2*N-1)*stride+offset, h_a[(2*N-1)*stride+offset]);


    /* copy array elements from CPU to GPU */
    hipMemcpy((void *)d_a, (void *)h_a, sizeof(unsigned long long) * size, hipMemcpyHostToDevice);


    //for (int l=0; l < 10 ; l++) {
        /* launch kernel*/
        dim3 Db = dim3(1);
        dim3 Dg = dim3(1,1,1);

        //printf("Launch kernel with parameters: %d, N: %lld, stride: %d\n", iterations, N, stride);
        hipLaunchKernelGGL(global_latency, dim3(Dg), dim3(Db), 0, 0, d_a, N, iterations, 1, duration);

        hipError_t error_id = hipDeviceSynchronize();
        if (error_id != hipSuccess) {
            printf("Error is %s\n", hipGetErrorString(error_id));
        }

        /* copy results from GPU to CPU */

        //hipMemcpy((void *)h_a, (void *)d_a, sizeof(unsigned long long) * N, hipMemcpyDeviceToHost);
        hipMemcpy((void *)latency, (void *)duration, sizeof(unsigned long long), hipMemcpyDeviceToHost);


        latency_sum+=latency[0];
    //}

    /* free memory on GPU */
    hipFree(d_a);
    hipFree(duration);

    /*free memory on CPU */
    free(h_a);
    free(latency);

    printf("%f\n", (double)(latency_sum/(10.0*256*iterations)));
}

void measure_global1() {
    // we will measure latency of global memory
    // One thread that accesses an array.
    // loads are dependent on the previously loaded values

    int N, iterations, stride;

    // initialize upper bounds here
    int stride_upper_bound;

    printf("Global1: Global memory latency for 1 KB array and varying strides.\n");
    printf("   stride (bytes), latency (clocks)\n");

    N=256;                // 131072;
    iterations = 4;
    stride_upper_bound = N;
    for (stride = 1; stride <= (stride_upper_bound) ; stride+=1) {
        printf ("  %5d, ", stride*elt_size);
        parametric_measure_global(N, iterations, 1, stride);
    }
}
/**
void measure_global5() {
    int N, iterations, stride, stride_in_kb, ignore_iterations;
    stride_in_kb = 32;
    stride = stride_in_kb * 1024 / elt_size; // stride in number of elements

    // initialize upper bounds here
    printf("[GLOBAL]   Global5: Global memory latency for %d KB stride.\n", stride*elt_size/1024);
    printf("[GLOBAL]   Array size (KB), latency (clocks)\n");

    iterations = 10;
    ignore_iterations = 1;
    for (N = (stride*1); N <= (stride*256); N += stride) {
        //if (((N*elt_size/1024 * page_size/4) >= 2016) && ((N*elt_size/1024 * page_size/4) <= 2044)) {
          //  printf ("insert hello");
            printf ("  %5d, ", N*elt_size/1024 * page_size/4);
            parametric_measure_global(N, iterations, ignore_iterations, stride);
//      }
    }
}
**/

void measure_global5() {
    int N, iterations, stride, stride_in_kb, ignore_iterations;
    stride_in_kb = 32;
    //stride_in_kb = 256;
    stride = stride_in_kb * 1024 / elt_size; // stride in number of elements

    // initialize upper bounds here
    printf("[GLOBAL]   Global5: Global memory latency for %d KB stride.\n", stride*elt_size/1024);
    printf("[GLOBAL]   Array size (KB), latency (clocks)\n");

    iterations = 10;
    ignore_iterations = 1;
    //for (N = (stride*1); N <= (stride*256); N += stride) {
    for (N = (stride*1); N <= (stride*256); N += stride) { 
   // if((N*elt_size/1024 * page_size/4 >= 8000)) {
   	if((N*elt_size/1024 * page_size/4 >= 3776) && (N*elt_size/1024 * page_size/4 <= 4192)) {
        //if((N*elt_size/1024 * page_size/4 >= 1376) && (N*elt_size/1024 * page_size/4 <= 1536)) {
	//if((N*elt_size/1024 * page_size/4 >= 1952) && (N*elt_size/1024 * page_size/4 <= 2144)) {
    	  printf ("[GLOBAL]   %5d, ", N*elt_size/1024 * page_size/4);
          parametric_measure_global(N, iterations, ignore_iterations, stride);
     //    }
       }
    }
}

void measure_global_near_2mb() {
    int iterations, stride_in_kb, ignore_iterations;
    long long N;
    stride_in_kb = 32;  // 32 KB stride
    int stride = stride_in_kb * 1024 / elt_size; // Stride in number of elements

    printf("[GLOBAL] Global memory latency near 2 MB size.\n");
    printf("[GLOBAL] Array size (KB), latency (clocks)\n");

    iterations = 10;
    ignore_iterations = 1;

    // Test from 1.5 MB to 3 MB in 0.25 MB increments
    for (N = (1.5 * 1024 * 1024 / sizeof(unsigned long long));  // 1.5 MB
         N <= (3.0 * 1024 * 1024 / sizeof(unsigned long long)); // 3 MB
         N += (0.25 * 1024 * 1024 / sizeof(unsigned long long))) { // 0.25 MB increments

        printf("[GLOBAL] %5lld KB, ", (N * sizeof(unsigned long long)) / 1024);
        parametric_measure_global(N, iterations, ignore_iterations, stride);
    }
}

void measure_global6() {
    int N, stride, entries, top, bottom;
    top = 65;
    bottom = 64;



    printf("\n[GLOBAL] Global6: Testing associativity of L1 TLB, %d and %d elements accessesed.\n", bottom, top);
    printf("[GLOBAL]   entries accessed, array size (KB), stride (KB), latency\n");

    for (entries = bottom; entries <= top; entries++) {
        for (stride = 1; stride < (512*1024); stride *= 2 ) {
            for (int substride = 1; substride < 16; substride *= 2 ){
                int stride2 = stride * sqrt(sqrt(substride)) + 0.5;
                N = entries * stride2;
                printf ("  %d, %7.3f, %7f, ", entries, N*elt_size/1024.0*page_size/4, stride2*elt_size/1024.0*page_size/4);
                parametric_measure_global(N*page_size/4, 4, 1, stride2*page_size/4);
            }
        }
    }
}

void measure_global4()
{
    printf ("\nGlobal4: Measuring L2 TLB page size using %d MB stride\n", 2 * page_size/4);
    printf ("  offset (bytes), latency (clocks)\n");

    // Small offsets (approx. page size) are interesting. Search much bigger offsets to
    // ensure nothing else interesting happens.
    for (int offset = -2048/4; offset <= (2097152+1536)/4; offset += (offset < 1536) ? 128/4 : 4096/4)
    {
        printf ("[GLOBAL]  %d, ", offset*elt_size *page_size/4);
        measure_pagesize(10, 2097152/4 *page_size/4, offset* page_size/4);
        //measure_pagesize(10, 32786, offset* page_size/4);
    }
}

int main(int argc, char *argv[]) {
    printf("START OF GLOBAL\n");
    printf("Assuming page size is %d KB\n", page_size);

    if (argc < 2) {
        printf("No flags provided. Exiting.\n");
        return 1;
    }

    // Parse command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-N") == 0) {
            printf("[GLOBAL]  Global1: L1 TLB page size.\n");
            measure_global1(); // L1 TLB page size
        } else if (strcmp(argv[i], "-Fo") == 0) {
            printf("[GLOBAL]   Global4: L2 TLB page size.\n");
            measure_global4(); // L2 TLB page size
        } else if (strcmp(argv[i], "-F") == 0) {
            printf("[GLOBAL]   Global5: Page table size.\n");
            measure_global5(); // page table size
        } else if (strcmp(argv[i], "-S") == 0) {
            printf("[GLOBAL] Global6: L1 TLB associativity.\n");
            measure_global6(); // L1 TLB associativity
        } else if(strcmp(argv[i], "-Ch") == 0) {
	    printf("[GLOBAL] Global5 2mb.\n");
	    measure_global_near_2mb();
	}
	else {
            printf("Unknown flag: %s\n", argv[i]);
        }
    }

    printf("END OF GLOBAL\n");
    return 0;
}
