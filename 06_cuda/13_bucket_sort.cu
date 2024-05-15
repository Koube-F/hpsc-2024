#include <cstdio>
//#include <vector>

__global__ void count_bucket(int n, int *key, int range, int *bucket){
  const int index = blockIdx.x*blockDim.x +threadIdx.x;
  //for(...; index <n; ...)
  if(index < n){ //# of threads is not n, but multiple ofTHREAD_PER_BLOCK
    //bucket[key[i]]++;
    atomicAdd(&bucket[key[index]], 1);
  }
}

__global__ void write_key(int n, int *key, int range, int *bucket){ 
  const int index = blockIdx.x*blockDim.x +threadIdx.x;
  if(index >= n){ return; }
  int i = 0;
  for(int sum=0; sum<=index; ++i){
    sum += bucket[i];
  }
  key[index] = i-1;

  /*
  this function(simple_sum(O(range)) and liner_search(O(range))) is O(range)

  if range is very large,
  scan(O(log(range))) and binary_search(O(log(range))) is better
  
  */
}

int main(){
  int n = 50;
  int range = 5;
  //std::vector<int> key(n);
  int *key;
  cudaMallocManaged(&key, n*sizeof(int));
  for(int i=0; i<n; i++){
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  int *bucket;
  cudaMalloc(&bucket, range*sizeof(int));
  //bucket[i] = 0;
  cudaMemset(bucket, 0, range*sizeof(int));

  const int THREAD_PER_BLOCK = 32;
  const int m = (n+THREAD_PER_BLOCK-1)/THREAD_PER_BLOCK;
  //ceil(n/THREAD_PER_BLOCK)
  count_bucket<<<m, THREAD_PER_BLOCK>>>(n, key, range, bucket);
  cudaDeviceSynchronize();
  write_key<<<m, THREAD_PER_BLOCK>>>(n, key, range, bucket);

  cudaFree(bucket);

  for(int i=0; i<n; i++){
    printf("%d ",key[i]);
  }
  printf("\n");

  cudaFree(key);
}
