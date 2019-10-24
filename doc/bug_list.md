#### Better Compatibility for `__CUDA_ARCH__` identification
[rasterize_cuda_kernel.cu](../thirdparty/neural_renderer/neural_renderer/cuda/rasterize_cuda_kernel.cu)
```
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else
static __inline__ __device__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                __double_as_longlong(val + __longlong_as_double(assumed)));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN) } while (assumed != old);
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

```

to 

```cpp
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600

#else
static __inline__ __device__ double atomicAdd(double *address, double val) {
unsigned long long int* address_as_ull = (unsigned long long int*)address;
unsigned long long int old = *address_as_ull, assumed;
if (val==0.0)
    return __longlong_as_double(old);
do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val +__longlong_as_double(assumed)));
} while (assumed != old);
return __longlong_as_double(old);
}

#endif
```