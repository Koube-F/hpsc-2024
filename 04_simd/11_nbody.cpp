#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <x86intrin.h>

int main() {
  const int N = 16;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
  }
  for(int i=0; i<N; i++) {
    /*
    for(int j=0; j<N; j++) {
      if(i != j) {
        float rx = x[i] - x[j];
        float ry = y[i] - y[j];
        float r = std::sqrt(rx * rx + ry * ry);
        fx[i] -= rx * m[j] / (r * r * r);
        fy[i] -= ry * m[j] / (r * r * r);
      }
    }
    */

    __mmask16 mask = _mm512_cmp_epi32_mask(
      _mm512_set1_epi32(i),
      _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15),
      _MM_CMPINT_NE
    );
    __m512 xivec = _mm512_set1_ps(x[i]);
    __m512 yivec = _mm512_set1_ps(y[i]);
    __m512 xjvec = _mm512_load_ps(x);
    __m512 yjvec = _mm512_load_ps(y);
    __m512 rxvec = _mm512_sub_ps(xivec, xjvec);
    __m512 ryvec = _mm512_sub_ps(yivec, yjvec);
    __m512 rvec = _mm512_rsqrt14_ps(
      _mm512_add_ps(
        _mm512_mul_ps(rxvec, rxvec),
        _mm512_mul_ps(ryvec, ryvec)
      )
    ); // calculate 1/r
    __m512 mvec = _mm512_load_ps(m);
    __m512 diff = _mm512_mul_ps(
        _mm512_mul_ps(mvec, rvec),
        _mm512_mul_ps(rvec, rvec)
    );
    __m512 diffx = _mm512_mul_ps(rxvec, diff);
    __m512 diffy = _mm512_mul_ps(ryvec, diff);
    diffx = _mm512_mask_blend_ps(mask, _mm512_setzero_ps(), diffx);
    diffy = _mm512_mask_blend_ps(mask, _mm512_setzero_ps(), diffy);
    fx[i] -= _mm512_reduce_add_ps(diffx);
    fy[i] -= _mm512_reduce_add_ps(diffy);

    printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
