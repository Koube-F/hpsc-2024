#include <cstdio>
#include <cstdlib>
#include <vector>

int main() {
  int n = 50;
  int range = 5;
  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ",key[i]);
  }
  printf("\n");

  //this is counting sort rather than bucket sort
  std::vector<int> bucket(range,0); 
#pragma omp parallel for shared(bucket)
  for (int i=0; i<n; i++)
#pragma omp atomic update
    bucket[key[i]]++;

  std::vector<int> offset(range,0);
  //offset[i] = sum(k=0 to i-1 ,bucket[k])
/*
  for (int i=1; i<range; i++) 
    offset[i] = offset[i-1] + bucket[i-1];
*/
#pragma omp parallel for
  for(int i=1; i<range; ++i)
    offset[i] = bucket[i-1];
  std::vector<int> scan_tmp(range,0);
#pragma omp parallel
  for(int j=1; j<range; j<<=1){
#pragma omp for
    for(int i=0; i<range; i++)
      scan_tmp[i] = offset[i];
#pragma omp for
    for(int i=j; i<range; i++)
      offset[i] += scan_tmp[i-j];
  }
  
#pragma omp parallel for
  for (int i=0; i<range; i++) {
    int j = offset[i];
    for (; bucket[i]>0; bucket[i]--) {
      key[j++] = i;
    }
  }

  for (int i=0; i<n; i++) {
    printf("%d ",key[i]);
  }
  printf("\n");
}
