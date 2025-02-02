
#include "cudaLib.cuh"

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort)
{
	if (code != cudaSuccess) 
	{
		fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

__global__ 
void saxpy_gpu (float* x, float* y, float scale, int size) {
	//	Insert GPU SAXPY kernel code here

		int i = threadIdx.x + blockIdx.x*blockDim.x;

	if (i <size){
		y[i] += scale*x[i];
	}

	__syncthreads();
}

int runGpuSaxpy(int vectorSize) {

	std::cout << "Hello GPU Saxpy!\n";

	//	Insert code here
	std::cout << "Lazy, you are!\n";
	std::cout << "Write code, you must\n";

	int size = vectorSize;
	float scale = 1.5;
	float *h_x  = (float *)malloc(size* sizeof(float));
	float *h_y  = (float *)malloc(size* sizeof(float));
	float *h_y_temp  = (float *)malloc(size* sizeof(float));

	// if (h_x == NULL || h_y == NULL ) {
	// 	printf("Unable to malloc memory ... Exiting!");
	// 	return -1;
	// }

	vectorInit(h_x, vectorSize);
	vectorInit(h_y, vectorSize);
	std::memcpy(h_y_temp, h_y, vectorSize * sizeof(float));

	for(int i = 0;i<10;i++){
		printf("%3.4f, ",h_x[i]);
	}
	printf("\n");
	for(int i = 0;i<10;i++){
		printf("%3.4f, ",h_y[i]);
	}

	printf("\n");
	float *d_x,*d_y;

 	cudaMalloc((void**)&d_x, size*sizeof(float));
	cudaMalloc((void**)&d_y, size*sizeof(float));

	cudaMemcpy(d_x,h_x,size*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(d_y,h_y,size*sizeof(float),cudaMemcpyHostToDevice);

	int threads_per_block = 1024;
	int blocks_per_grid = (size+threads_per_block-1)/threads_per_block;

	saxpy_gpu<<<blocks_per_grid, threads_per_block>>>(d_x,d_y,scale,size);


	cudaMemcpy(h_y,d_y,size*sizeof(float),cudaMemcpyDeviceToHost);

	int errorCount = verifyVector(h_x, h_y_temp, h_y, scale, vectorSize);
	std::cout << "Found " << errorCount << " / " << vectorSize << " errors \n";

	for(int i = 0;i<10;i++){
		
		printf("%3.4f, ",h_y[i]);
	}
	cudaFree(d_x);
	cudaFree(d_y);
	free(h_x);
	free(h_y);

	return 0;
}

/* 
 Some helpful definitions

 generateThreadCount is the number of threads spawned initially. Each thread is responsible for sampleSize points. 
 *pSums is a pointer to an array that holds the number of 'hit' points for each thread. The length of this array is pSumSize.

 reduceThreadCount is the number of threads used to reduce the partial sums.
 *totals is a pointer to an array that holds reduced values.
 reduceSize is the number of partial sums that each reduceThreadCount reduces.

*/

__global__
void generatePoints (uint64_t * pSums, uint64_t pSumSize, uint64_t sampleSize) {
	//	Insert code here

	int thread_id = threadIdx.x + blockIdx.x*blockDim.x;

    // Initialize curand state
    curandState_t rng;
	curand_init(clock64(), thread_id, 0, &rng);
	uint64_t hits = 0;
    if (thread_id >= pSumSize) return;

    for (int i = 0; i < sampleSize; i++) {
        float x = curand_uniform(&rng);
        float y = curand_uniform(&rng);

        if ((x * x + y * y) <= 1.0f) {
            hits++;
        }
    }

    // Store the count of hits per thread
    pSums[thread_id] = hits;
}
// __syncthreads();


__global__ 
void reduceCounts (uint64_t * pSums, uint64_t * totals, uint64_t pSumSize, uint64_t reduceSize) {
	//	Insert code here
int thread_id = threadIdx.x + blockIdx.x*blockDim.x;
int start_index = reduceSize*thread_id;
totals[thread_id] = 0;
	for (int i = 0;i < reduceSize;i++){
		totals[thread_id] += pSums[start_index + i];
	}

}

int runGpuMCPi (uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {

	//  Check CUDA device presence
	int numDev;
	cudaGetDeviceCount(&numDev);
	if (numDev < 1) {
		std::cout << "CUDA device missing!\n";
		return -1;
	}

	auto tStart = std::chrono::high_resolution_clock::now();
		
	float approxPi = estimatePi(generateThreadCount, sampleSize, 
		reduceThreadCount, reduceSize);
	
	std::cout << "Estimated Pi = " << approxPi << "\n";

	auto tEnd= std::chrono::high_resolution_clock::now();

	std::chrono::duration<double> time_span = (tEnd- tStart);
	std::cout << "It took " << time_span.count() << " seconds.";

	return 0;
}

double estimatePi(uint64_t generateThreadCount, uint64_t sampleSize, 
	uint64_t reduceThreadCount, uint64_t reduceSize) {
	
	double approxPi = 0;

	//      Insert code here
	std::cout << "Sneaky, you are ...\n";
	std::cout << "Compute pi, you must!\n";
	
	
	// int size = generateThreadCount;

	uint64_t *d_pSums;
	cudaMalloc((void**)&d_pSums, generateThreadCount * sizeof(uint64_t));

	int threads_per_block = 1024;
	int blocks_per_grid = (generateThreadCount+threads_per_block-1)/threads_per_block;

	generatePoints <<<blocks_per_grid, threads_per_block>>>(d_pSums,generateThreadCount, sampleSize);

	// uint64_t *h_pSums  = (uint64_t *)malloc(generateThreadCount* sizeof(uint64_t));
	// cudaMemcpy(h_pSums,d_pSums, generateThreadCount * sizeof(uint64_t), cudaMemcpyDeviceToHost);
	// 	for (int i = 0;i < generateThreadCount;i++){
	// 	approxPi += h_pSums[i];
	// 	// printf("%d ,",h_pSums[i]);

	// }



	// cudaFree(d_pSums);
	// free(h_pSums);

	

	
	uint64_t *h_reducethreads  = (uint64_t *)malloc(reduceThreadCount* sizeof(uint64_t));


	uint64_t *d_reducethreads;
	cudaMalloc((void**)&d_reducethreads, reduceThreadCount * sizeof(uint64_t));



	
	
	blocks_per_grid = (reduceThreadCount+threads_per_block-1)/threads_per_block;
	reduceCounts <<<blocks_per_grid, threads_per_block>>>(d_pSums, d_reducethreads, generateThreadCount, reduceSize);

	cudaMemcpy(h_reducethreads,d_reducethreads, reduceThreadCount * sizeof(uint64_t), cudaMemcpyDeviceToHost);

	for (int i = 0;i < reduceThreadCount;i++){
		approxPi += h_reducethreads[i];
	}



	cudaFree(d_pSums);
	cudaFree(d_reducethreads);
	free(h_reducethreads);

	approxPi = (4*approxPi)/(generateThreadCount*sampleSize);

	printf("%3.7f, ",approxPi);


	return approxPi;

	
	
}
