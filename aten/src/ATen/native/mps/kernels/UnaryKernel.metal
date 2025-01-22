#include <metal_stdlib>
using namespace metal;

constant float a[4] = {0.886226899, -1.645349621, 0.914624893, -0.140543331};
constant float b[4] = {-2.118377725, 1.442710462, -0.329097515, 0.012229801};
constant float c[4] = {-1.970840454, -1.624906493, 3.429567803, 1.641345311};
constant float d[2] = {3.543889200, 1.637067800};

template <typename T0, typename T1>
kernel void erfinv_kernel( device T0* output [[buffer(0)]],
                           device T1* input [[buffer(1)]],
                           uint index [[thread_position_in_grid]]) {
  float y = input[index];
  float x, z, num, dem; /*working variables */
  /* coefficients in rational expansion */

  float y_abs = abs(y);
  if (y_abs >= 1.0f) {
    output[index] = T0( y_abs > 1.0f ? NAN : copysign(INFINITY, y));
    return;
  }
  if (y_abs <= 0.7f) {
    z = y * y;
    num = ((a[3] * z + a[2]) * z + a[1])*z + a[0];
    dem = (((b[3] * z + b[2]) * z + b[1]) * z +b[0]) * z + 1.0f;
    x = y * num / dem;
  } else {
    z = sqrt(-1.0f*log((1.0-y_abs)/2.0));
    num = ((c[3] * z + c[2]) * z + c[1]) * z + c[0];
    dem = (d[1] * z + d[0]) * z + 1.0f;
    x = copysign(num, y) / dem;
  }

  output[index] = T0(x);
}

template <typename T0, typename T1>
kernel void exp_kernel( device T0* output [[buffer(0)]],
                        device T1* input [[ buffer(1)]],
                        uint index [[thread_position_in_grid]]) {
  output[index] = T0(precise::exp(input[index]));
}


template <typename T>
struct vectypes {};

template <>
struct vectypes<float> {
  using type4 = float4;
  using type3 = float3;
  using type2 = float2;
};

template <>
struct vectypes<half> {
  using type4 = half4;
  using type3 = half3;
  using type2 = half2;
};

#if __METAL_VERSION__ >= 310
template <>
struct vectypes<bfloat> {
  using type4 = bfloat4;
  using type3 = bfloat3;
  using type2 = bfloat2;
};
#endif

template <>
struct vectypes<short> {
  using type4 = short4;
  using type3 = short3;
  using type2 = short2;
};

template <typename T>
using vec2type_t = typename vectypes<T>::type2;

template <typename T>
using vec4type_t = typename vectypes<T>::type4;

template <typename T0>
kernel void exp_complex_kernel( device vec2type_t<T0> *output [[buffer(0)]],
                                device vec2type_t<T0> *input [[ buffer(1)]],
                                uint index [[thread_position_in_grid]]) {
  output[index].x = T0(precise::exp(input[index].x)*precise::cos(input[index].y));
  output[index].y = T0(precise::exp(input[index].x)*precise::sin(input[index].y));
}

template <typename T0, typename T1>
kernel void tanh_kernel( device T0* output [[buffer(0)]],
                        device T1* input [[ buffer(1)]],
                        uint index [[thread_position_in_grid]]) {
  output[index] = T0(precise::tanh(input[index]));
}


#if __METAL_VERSION__ >= 310
bfloat dot(bfloat2 a, bfloat2 b) {
  return a.x * b.x + a.y * b.y;
}
#endif

short dot(short2 a, short2 b) {
  return a.x * b.x + a.y * b.y;
}

template<typename T>
T complex_div(T a, T b) {
  auto denom = dot(b, b);
  return T(dot(a, b), a.y * b.x - a.x * b.y)/denom;
}

template <typename T0>
kernel void tanh_complex_kernel( device vec2type_t<T0> *output [[buffer(0)]],
                                 device vec2type_t<T0> *input [[ buffer(1)]],
                                 uint index [[thread_position_in_grid]]) {
  //tanh(x+iy)=(tanh(x)+itan(y))/(1+itahnh(x)*tan(y));
  auto tanh_x = T0(precise::tanh(input[index].x));
  auto tan_y = T0(precise::tan(input[index].y));
  output[index] = complex_div(vec2type_t<T0>(tanh_x, tan_y), vec2type_t<T0>(T0(1), tanh_x * tan_y));
}


#define INSTANTIATE_UNARY_KERNELS2(DTYPE0, DTYPE1)                                   \
  template [[host_name("erfinv_" #DTYPE0 "_" #DTYPE1)]] kernel void erfinv_kernel(   \
      device DTYPE0* output [[buffer(0)]],                                            \
      device DTYPE1* input  [[buffer(1)]],                                           \
      uint id [[thread_position_in_grid]]);					     \
  template [[host_name("exp_" #DTYPE0 "_" #DTYPE1)]] kernel void exp_kernel(   \
      device DTYPE0* output [[buffer(0)]],                                            \
      device DTYPE1* input  [[buffer(1)]],                                           \
      uint id [[thread_position_in_grid]]);					     \
  template [[host_name("tanh_" #DTYPE0 "_" #DTYPE1)]] kernel void tanh_kernel(   \
      device DTYPE0* output [[buffer(0)]],                                            \
      device DTYPE1* input [[buffer(1)]],                                           \
      uint id [[thread_position_in_grid]]);

#if __METAL_VERSION__ >= 310
INSTANTIATE_UNARY_KERNELS2(bfloat, bfloat);
#endif
INSTANTIATE_UNARY_KERNELS2(half, half);
INSTANTIATE_UNARY_KERNELS2(float, float);
INSTANTIATE_UNARY_KERNELS2(float, bool);
INSTANTIATE_UNARY_KERNELS2(float, uchar);
INSTANTIATE_UNARY_KERNELS2(float, char);
INSTANTIATE_UNARY_KERNELS2(float, short);
INSTANTIATE_UNARY_KERNELS2(float, int);
INSTANTIATE_UNARY_KERNELS2(float, long);

#define INSTANTIATE_UNARY_KERNELS_VEC2(DTYPE0, DTYPE1)				\
  template [[host_name("exp_complex_" #DTYPE0 "_" #DTYPE1)]] kernel void exp_complex_kernel<DTYPE0>(   \
      device vec2type_t<DTYPE0>* output [[buffer(0)]],                                            \
      device vec2type_t<DTYPE0>* input [[buffer(1)]],                                           \
      uint did [[thread_position_in_grid]]);							\
  template [[host_name("tanh_complex_" #DTYPE0 "_" #DTYPE1)]] kernel void tanh_complex_kernel<DTYPE0>(   \
      device vec2type_t<DTYPE0>* output [[buffer(0)]],                                            \
      device vec2type_t<DTYPE0>* input [[buffer(1)]],                                           \
      uint did [[thread_position_in_grid]]);

INSTANTIATE_UNARY_KERNELS_VEC2(short, short);
INSTANTIATE_UNARY_KERNELS_VEC2(float, float);
