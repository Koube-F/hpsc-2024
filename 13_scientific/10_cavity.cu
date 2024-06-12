#include <fstream>
#include <iostream>

#define NX 41
#define NY 41
#define NT 500
#define NIT 50
#define DX (2.0/(NX-1))
#define DY (2.0/(NY-1))
#define DT 0.01
#define RHO 1.0
#define NU 0.02

#define POW2(x) ((x)*(x))

__global__ void calc_b(double *b, double *u, double *v){
  const int j = threadIdx.x +1;
  const int i = blockIdx.x +1;
  b[j*NX +i] = RHO * (1 / DT *
    ((u[j*NX +i+1] - u[j*NX +i-1]) / (2 * DX) + (v[(j+1)*NX +i] - v[(j-1)*NX +i]) / (2 * DY)) -
    POW2((u[j*NX +i+1] - u[j*NX +i-1]) / (2 * DX)) - 2 * ((u[(j+1)*NX +i] - u[(j-1)*NX +i]) / (2 * DY) *
    (v[j*NX +i+1] - v[j*NX +i-1]) / (2 * DX)) - POW2(v[(j+1)*NX +i] - v[(j-1)*NX +i] / (2 * DY)) );
}

__global__ void calc_p(double *p, double *b, double *pn){
  const int j = threadIdx.x +1;
  const int i = blockIdx.x +1;
  p[j*NX +i] = (POW2(DY) * (pn[j*NX +i+1] + pn[j*NX +i-1]) +
    POW2(DX) * (pn[(j+1)*NX +i] + pn[(j-1)*NX +i]) -
    b[j*NX +i] * POW2(DX) * POW2(DY) )
    / (2 * (POW2(DX) + POW2(DY) ));
}

__global__ void p_all_m1(double *p){
  p[threadIdx.x*NX +NX-1] = p[threadIdx.x*NX +NX-2];
}
__global__ void p_0_all(double *p){
  p[0*NX +threadIdx.x] = p[1*NX +threadIdx.x];
}
__global__ void p_all_0(double *p){
  p[threadIdx.x*NX +0] = p[threadIdx.x*NX +1];
}
__global__ void p_m1_all(double *p){
  p[(NY-1)*NX +threadIdx.x] = 0;
}

__global__ void calc_u(double *u, double *p, double *un){ 
  const int j = threadIdx.x +1;
  const int i = blockIdx.x +1;
  u[j*NX +i] = un[j*NX +i] - un[j*NX +i] * DT / DX * (un[j*NX +i] - un[j*NX +i-1])
    -un[j*NX +i] * DT / DY * (un[j*NX +i] - un[(j-1)*NX +i])
    - DT / (2 * RHO * DX) * (p[j*NX +i+1] - p[j*NX +i-1])
    + NU * DT / POW2(DX) * (un[j*NX +i+1] - 2 * un[j*NX +i] + un[j*NX +i-1])
    + NU * DT / POW2(DY) * (un[(j+1)*NX +i] - 2 * un[j*NX +i] + un[(j-1)*NX +i]);
}
__global__ void calc_v(double *v, double *p, double *vn){ 
  const int j = threadIdx.x +1;
  const int i = blockIdx.x +1;
  v[j*NX +i] = vn[j*NX +i] - vn[j*NX +i] * DT / DX * (vn[j*NX +i] - vn[j*NX +i-1])
    -vn[j*NX +i] * DT / DY * (vn[j*NX +i] - vn[(j-1)*NX +i])
    - DT / (2 * RHO * DX) * (p[(j+1)*NX +i] - p[(j-1)*NX +i])
    + NU * DT / POW2(DX) * (vn[j*NX +i+1] - 2 * vn[j*NX +i] + vn[j*NX +i-1])
    + NU * DT / POW2(DY) * (vn[(j+1)*NX +i] - 2 * vn[j*NX +i] + vn[(j-1)*NX +i]);
}

__global__ void u_0_all(double *u){
  u[0*NX +threadIdx.x] = 0;
}
__global__ void u_all_0(double *u){
  u[threadIdx.x*NX +0] = 0;
}
__global__ void u_all_m1(double *u){
  u[threadIdx.x*NX +NX-1] = 0;
}
__global__ void u_m1_all(double *u){
  u[(NY-1)*NX +threadIdx.x] = 1;
}

__global__ void v_0_all(double *v){
  v[0*NX +threadIdx.x] = 0;
}
__global__ void v_m1_all(double *v){
  v[(NY-1)*NX +threadIdx.x] = 0;
}
__global__ void v_all_0(double *v){
  v[threadIdx.x*NX +0] = 0;
}
__global__ void v_all_m1(double *v){
  v[threadIdx.x*NX +NX-1] = 0;
}

int main(){
  //open file
  std::ofstream ufile("u.dat");
  std::ofstream vfile("v.dat");
  std::ofstream pfile("p.dat");
  if(!ufile){
    std::cerr << "fail open u.dat" << std::endl;
    return 1;
  }
  if(!vfile){
    std::cerr << "fail open v.dat" << std::endl;
    return 1;
  }
  if(!pfile){
    std::cerr << "fail open p.dat" << std::endl;
    return 1;
  }

  //double buffering
  double *u[2];
  double *v[2];
  double *p[2];
  double *b;
  cudaMallocManaged(&u[0], NX*NY*sizeof(double));
  cudaMallocManaged(&u[1], NX*NY*sizeof(double));
  cudaMallocManaged(&v[0], NX*NY*sizeof(double));
  cudaMallocManaged(&v[1], NX*NY*sizeof(double));
  cudaMallocManaged(&p[0], NX*NY*sizeof(double));
  cudaMallocManaged(&p[1], NX*NY*sizeof(double));
  cudaMallocManaged(&b, NX*NY*sizeof(double));
  //np.zeros
  cudaMemset(u[0], 0, NX*NY*sizeof(double));
  cudaMemset(v[0], 0, NX*NY*sizeof(double));
  cudaMemset(p[0], 0, NX*NY*sizeof(double));
  //only fill the critical sections with 0
  p[1][0*NX +NX-2] = 0;
  p[1][(NY-1)*NX +NX-2] = 0;
  p[1][1*NX +0] = 0;
  p[1][(NY-1)*NX +1] = 0;
  //b[:,-1], b[0,:], b[:,0], b[-1,:] might not be referenced?
  cudaDeviceSynchronize();

  for(int n=0; n<NT; ++n){
    calc_b<<<NX-2, NY-2>>>(b, u[n&1], v[n&1]); 
    cudaDeviceSynchronize();

    for(int it=0; it<NIT; ++it){
      //(n*NIT +it)%2 == ((n&NIT)^it)&1
      calc_p<<<NX-2, NY-2>>>(p[(~((n&NIT)^it))&1], b, p[((n&NIT)^it)&1]);
      cudaDeviceSynchronize();
      p_all_m1<<<1, NY>>>(p[(~((n&NIT)^it))&1]);
      p_0_all<<<1, NX>>>(p[(~((n&NIT)^it))&1]);
      p_all_0<<<1, NY>>>(p[(~((n&NIT)^it))&1]);
      p_m1_all<<<1, NX>>>(p[(~((n&NIT)^it))&1]);
      cudaDeviceSynchronize();
    }

    calc_u<<<NX-2, NY-2>>>(u[(~n)&1], p[(n&NIT)&1], u[n&1]);
    calc_v<<<NX-2, NY-2>>>(v[(~n)&1], p[(n&NIT)&1], v[n&1]);
    u_0_all<<<1, NX>>>(u[(~n)&1]);
    u_all_0<<<1, NY>>>(u[(~n)&1]);
    u_all_m1<<<1, NY>>>(u[(~n)&1]);
    u_m1_all<<<1, NX>>>(u[(~n)&1]);
    v_0_all<<<1, NX>>>(v[(~n)&1]);
    v_m1_all<<<1, NX>>>(v[(~n)&1]);
    v_all_0<<<1, NY>>>(v[(~n)&1]);
    v_all_m1<<<1, NY>>>(v[(~n)&1]);
    cudaDeviceSynchronize();

    //write file
    for(int i=0; i<NX*NY; ++i){
      ufile << u[(~n)&1][i] << ' ';
    }
    ufile << '\n';
    for(int i=0; i<NX*NY; ++i){
      vfile << v[(~n)&1][i] << ' ';
    }
    vfile << '\n';
    for(int i=0; i<NX*NY; ++i){
      pfile << p[(n&NIT)&1][i] << ' ';
    }
    pfile << '\n';
  }

  cudaFree(u[0]);
  cudaFree(u[1]);
  cudaFree(v[0]);
  cudaFree(v[1]);
  cudaFree(p[0]);
  cudaFree(p[1]);
  cudaFree(b);
  
  //close file
  ufile.close();
  vfile.close();
  pfile.close();
  return 0;
}
