#include <stdint.h>

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <ATen/cuda/CUDAContext.h>
#include <torch/torch.h>

#include <c10/cuda/CUDAGuard.h> // support multiple GPUs

#include <algorithm>
#include <stdexcept>

#include <cstdio>

// OpenGL use left-bottom as origin
// OpenCV use left-top as origin

#define LEFT_TOP_AS_ORIGIN


#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be a contiguous tensor")
#define CHECK_IS_INT(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Int, #x " must be an int tensor")
#define CHECK_IS_FLOATING(x) TORCH_CHECK(x.scalar_type() == at::ScalarType::Float || x.scalar_type() == at::ScalarType::Half || x.scalar_type() == at::ScalarType::Double, #x " must be a floating tensor")

//inline constexpr __device__ float PI() { return 3.141592653589793f; }

//// just for compatability of half precision in AT_DISPATCH_FLOATING_TYPES_AND_HALF... program will never reach here!
// __device__ inline at::Half atomicAdd(at::Half *address, at::Half val) {
//  // requires CUDA >= 10 and ARCH >= 70
//  // this is very slow compared to float or __half2, never use it.
//  //return atomicAdd(reinterpret_cast<__half*>(address), val);
//}


//__device__ inline at::Half atomicAdd(at::Half *address, at::Half val) {
//	__half v;
//	*(reinterpret_cast<at::Half *>(&v)) = val;
//	return atomicAdd(reinterpret_cast<__half*>(address),v);
//}

template <typename T>
__host__ __device__ T div_round_up(T val, T divisor) {
    return (val + divisor - 1) / divisor;
}


/*
	+x 0
	-x 1
	+y 2
	-y 3
	+z 4
	-z 5
	+x  div x, flip u,v // div -x
	-x  div x, flip u
	+y  div y
	-y  div y, flip u,v
	+z  div z, flip v
	-z  div z
*/


__device__ void EdgeTable(int L, int flag, int * index_xy) {
	int input_face = index_xy[0];
	int input_x = index_xy[1];
	int input_y = index_xy[2];
#ifdef LEFT_TOP_AS_ORIGIN
	if (input_face == 0) {
		if (flag == 1) { index_xy[0] = 4; index_xy[1] = L - 1; index_xy[2] = input_y; }
		else if (flag == 2) { index_xy[0] = 5; index_xy[1] = 0; index_xy[2] = input_y; }
		else if (flag == 4) { index_xy[0] = 3; index_xy[1] = L - 1; index_xy[2] = input_x; }
		else { index_xy[0] = 2; index_xy[1] = L - 1; index_xy[2] = input_x; }
	}
	else if (input_face == 1) {
		if (flag == 1) { index_xy[0] = 5; index_xy[1] = L - 1; index_xy[2] = input_y; }
		else if (flag == 2) { index_xy[0] = 4; index_xy[1] = 0; index_xy[2] = input_y; }
		else if (flag == 4) { index_xy[0] = 3; index_xy[1] = 0; index_xy[2] = L - 1 - input_x; }
		else { index_xy[0] = 2; index_xy[1] = 0; index_xy[2] = L - 1 - input_x; }
	}
	else if (input_face == 2) {
		if (flag == 1) { index_xy[0] = 1; index_xy[1] = L -1-input_y; index_xy[2] = L-1; }
		else if (flag == 2) { index_xy[0] = 0; index_xy[1] = input_y; index_xy[2] = L-1; }
		else if (flag == 4) { index_xy[0] = 4; index_xy[1] = input_x; index_xy[2] = L - 1; }
		else { index_xy[0] = 5; index_xy[1] = L - 1 - input_x; index_xy[2] = L - 1; }
	}
	else if (input_face == 3) {
		if (flag == 1) { index_xy[0] = 1; index_xy[1] = L-1-input_y; index_xy[2] = 0; }
		else if (flag == 2) { index_xy[0] = 0; index_xy[1] = input_y; index_xy[2] = 0; }
		else if (flag == 4) { index_xy[0] = 4; index_xy[1] = input_x; index_xy[2] = 0; }
		else { index_xy[0] = 5; index_xy[1] = L-1-input_x;  index_xy[2] = 0; }
	}
	else if (input_face == 4) {
		if (flag == 1) { index_xy[0] = 1; index_xy[1] = L - 1; index_xy[2] = input_y; }
		else if (flag == 2) { index_xy[0] = 0; index_xy[1] = 0; index_xy[2] = input_y; }
		else if (flag == 4) { index_xy[0] = 3; index_xy[1] = input_x; index_xy[2] =0; }
		else { index_xy[0] = 2; index_xy[1] = input_x; index_xy[2] = 0; }
	}
	else {
		if (flag == 1) { index_xy[0] = 0; index_xy[1] = L - 1; index_xy[2] = input_y; }
		else if (flag == 2) { index_xy[0] = 1; index_xy[1] = 0; index_xy[2] = input_y; }
		else if (flag == 4) { index_xy[0] = 3; index_xy[1] = L-1-input_x; index_xy[2] = L-1; }
		else { index_xy[0] = 2; index_xy[1] = L - 1 - input_x; index_xy[2] = L-1; }
	}
#else
	if (input_face == 0) {
		if (flag == 1) { index_xy[0] = 4; index_xy[1] = L-1 ; index_xy[2] = input_y; }
		else if (flag == 2) { index_xy[0] = 5; index_xy[1] = 0; index_xy[2] = input_y; }
		else if (flag == 4) { index_xy[0] = 2; index_xy[1] = L - 1; index_xy[2] = L - 1 - input_x; }
		else { index_xy[0] = 3; index_xy[1] = L - 1; index_xy[2] = L - 1 - input_x; }
	}
	else if (input_face == 1) {
		if (flag == 1) { index_xy[0] = 5; index_xy[1] = L - 1; index_xy[2] = input_y; }
		else if (flag == 2) { index_xy[0] = 4; index_xy[1] = 0; index_xy[2] = input_y; }
		else if (flag == 4) { index_xy[0] = 2; index_xy[1] = 0; index_xy[2] = input_x; }
		else { index_xy[0] = 3; index_xy[1] = 0; index_xy[2] = input_x; }
	}
	else if (input_face == 2) {
		if (flag == 1) { index_xy[0] = 1; index_xy[1] = input_y; index_xy[2] = 0; }
		else if (flag == 2) { index_xy[0] = 0; index_xy[1] = L-1-input_y; index_xy[2] = 0; }
		else if (flag == 4) { index_xy[0] = 5; index_xy[1] = L - 1 - input_x; index_xy[2] = 0; }
		else { index_xy[0] = 4; index_xy[1] = input_x; index_xy[2] = 0; }
	}
	else if (input_face == 3) {
		if (flag == 1) { index_xy[0] = 1; index_xy[1] = input_y; index_xy[2] = L - 1; }
		else if (flag == 2) { index_xy[0] = 0; index_xy[1] = L - 1 - input_y; index_xy[2] = L - 1; }
		else if (flag == 4) { index_xy[0] = 5; index_xy[1] = L - 1 - input_x; index_xy[2] = L - 1; }
		else { index_xy[0] = 4; index_xy[1] = input_x;  index_xy[2] = L - 1; }
	}
	else if (input_face == 4) {
		if (flag == 1) { index_xy[0] = 1; index_xy[1] = L - 1; index_xy[2] = input_y; }
		else if (flag == 2) { index_xy[0] = 0; index_xy[1] = 0; index_xy[2] = input_y; }
		else if (flag == 4) { index_xy[0] = 2; index_xy[1] = input_x; index_xy[2] = L - 1; }
		else { index_xy[0] = 3; index_xy[1] = input_x; index_xy[2] = L - 1; }
	} 
	else {
		if (flag == 1) { index_xy[0] = 0; index_xy[1] = L - 1; index_xy[2] = input_y; }
		else if (flag == 2) { index_xy[0] = 1; index_xy[1] = 0; index_xy[2] = input_y; }
		else if (flag == 4) { index_xy[0] = 2; index_xy[1] = L - 1 - input_x; index_xy[2] = 0; }
		else { index_xy[0] = 3; index_xy[1] = L - 1 - input_x; index_xy[2] = 0; }	
	}
#endif
}

template<typename scalar_t>
__device__ void Compute_Cubemap_UV(scalar_t x, scalar_t y, scalar_t z, scalar_t * uv, int * index)
{
	int max_dim = 0;
	scalar_t x_ = abs(x);
	scalar_t y_ = abs(y);
	scalar_t z_ = abs(z);
	scalar_t max_v = x_;
	if (y_ > max_v) { max_v = y_; max_dim = 1; }
	if (z_ > max_v) { max_v = z_; max_dim = 2; }
	if (max_dim == 0) {
		uv[0] = z / x;
		uv[1] = y / x;
		if (x >= 0.f) {
			*index = 0; uv[0] = -uv[0]; uv[1] = -uv[1];
		}
		else {
			*index = 1; uv[0] = -uv[0];
		}
	}
	else if (max_dim == 1) {
		uv[0] = x / y;
		uv[1] = z / y;
		if (y >= 0.f) {
			*index = 2; 
		}
		else {
			*index = 3; uv[0] = -uv[0]; uv[1] = -uv[1];
		}
	}
	else {
		uv[0] = x / z;
		uv[1] = y / z;
		if (z >= 0.f) {
			*index = 4; uv[1] = -uv[1];
		}
		else {
			*index = 5;
		}
	}
}

template<typename scalar_t>
__device__ bool Compute_Seamless_Index(
	int index, int L, const scalar_t * uv, int * index_xy, scalar_t * kxky, int * out_flag
) {
	scalar_t loc_uv[2]; // copy uv
	loc_uv[0] = uv[0]; loc_uv[1] = uv[1];
	
	int uy_0, ux_0;
	int uy_1, ux_1;
	scalar_t kx, ky;
	int flag = 0;
	bool is_vertex = false;

#ifdef LEFT_TOP_AS_ORIGIN
	loc_uv[1] = -loc_uv[1];
#endif
	loc_uv[0] = (loc_uv[0] * 0.5f + 0.5f) * scalar_t(L);
	loc_uv[1] = (loc_uv[1] * 0.5f + 0.5f) * scalar_t(L);
	ux_0 = int(floor(loc_uv[0] - 0.5f)); uy_0 = int(floor(loc_uv[1] - 0.5f));
	ux_1 = ux_0 + 1; uy_1 = uy_0 + 1;
	kx = loc_uv[0] - scalar_t(ux_0) - 0.5f;
	ky = loc_uv[1] - scalar_t(uy_0) - 0.5f;
	if (ux_0 < 0) ux_0 = 0;
	if (ux_0 >= L) ux_0 = L - 1;
	if (ux_1 < 0) ux_1 = 0;
	if (ux_1 >= L) ux_1 = L - 1;

	if (uy_0 < 0) uy_0 = 0;
	if (uy_0 >= L) uy_0 = L - 1;
	if (uy_1 < 0) uy_1 = 0;
	if (uy_1 >= L) uy_1 = L - 1;

	if (loc_uv[0] < 0.5f) {
		flag = flag | 0x01;
		kx = 0.5f - loc_uv[0];
	}
	else if (loc_uv[0] >= scalar_t(L) - 0.5f) {
		flag = flag | 0x02;
	}
	if (loc_uv[1] < 0.5f) {
		flag = flag | 0x04;
		ky = 0.5f - loc_uv[1];
	}
	else if (loc_uv[1] >= scalar_t(L) - 0.5f) {
		flag = flag | 0x08;
	}
	if ((flag & 0x03) && (flag & 0x0C)) { // vertex case
		is_vertex = true;
		index_xy[0] = index; index_xy[1] = ux_0; index_xy[2] = uy_0;
		index_xy[3] = index; index_xy[4] = ux_0; index_xy[5] = uy_0; EdgeTable(L, flag & 0x03, &index_xy[3]);
		index_xy[6] = index; index_xy[7] = ux_0; index_xy[8] = uy_0; EdgeTable(L, flag & 0x0C, &index_xy[6]);	
	}
	else if ((flag & 0x03)) { // edge case u style
		index_xy[0] = index; index_xy[1] = ux_0; index_xy[2] = uy_0;
		index_xy[3] = index; index_xy[4] = ux_0; index_xy[5] = uy_0; EdgeTable(L, flag, &index_xy[3]);
		index_xy[6] = index; index_xy[7] = ux_0; index_xy[8] = uy_1;
		index_xy[9] = index; index_xy[10] = ux_0; index_xy[11] = uy_1; EdgeTable(L, flag, &index_xy[9]);	
	}
	else if ((flag & 0x0C)) { // edge case v style
		index_xy[0] = index; index_xy[1] = ux_0; index_xy[2] = uy_0;
		index_xy[3] = index; index_xy[4] = ux_1; index_xy[5] = uy_0;
		index_xy[6] = index; index_xy[7] = ux_0; index_xy[8] = uy_0; EdgeTable(L, flag, &index_xy[6]);
		index_xy[9] = index; index_xy[10] = ux_1; index_xy[11] = uy_0; EdgeTable(L, flag, &index_xy[9]);	
	}
	else {
		index_xy[0] = index; index_xy[1] = ux_0; index_xy[2] = uy_0;
		index_xy[3] = index; index_xy[4] = ux_1; index_xy[5] = uy_0;
		index_xy[6] = index; index_xy[7] = ux_0; index_xy[8] = uy_1;
		index_xy[9] = index; index_xy[10] = ux_1; index_xy[11] = uy_1;
	}
	kxky[0] = kx; kxky[1] = ky;

	*out_flag = flag;
	return is_vertex;	
}

template<typename scalar_t>
__device__ void Compute_Cubemap_UV_Backward(
	int index, scalar_t x, scalar_t y, scalar_t z, 
	scalar_t * uv, scalar_t * grad_xyz
) {
	int face = index / 2;
	if (face == 0) {//0,1
		if (index == 0) {	uv[0] = -uv[0]; uv[1] = -uv[1];	}
		else {	uv[0] = -uv[0]; 	}
		grad_xyz[0] = -(z * uv[0] + y * uv[1]) / (x*x);
		grad_xyz[1] = 1.f / x * uv[1];
		grad_xyz[2] = 1.f / x * uv[0];
	}
	else if (face == 1) {//2,3
		if (index == 2) {}
		else { uv[0] = -uv[0]; uv[1] = -uv[1]; }
		grad_xyz[0] = 1.f / y * uv[0];
		grad_xyz[1] = -(x * uv[0] + z * uv[1]) / (y*y);
		grad_xyz[2] = 1.f / y * uv[1];
	}
	else if (face == 2) {//4,5
		if (index == 4) { uv[1] = -uv[1]; }
		else {}
		grad_xyz[0] = 1.f / z * uv[0];
		grad_xyz[1] = 1.f / z * uv[1];
		grad_xyz[2] = -(x*uv[0] + y * uv[1]) / (z*z);
	}
}

////////////////////////////////////
// forward

template<typename scalar_t>
__global__ void Cubemap_Bilinear_Seamless_Kernel(
	const scalar_t * __restrict__ inputs, // [B,3]
	const scalar_t * __restrict__ cubemap, // [6,C,L,L]
	const scalar_t * __restrict__ fail_value, // [C]
	scalar_t * __restrict__ outputs, // [C,B]
	const uint32_t B, const uint32_t C, const uint32_t L
){
	const uint32_t n = blockIdx.x * blockDim.x + threadIdx.x;
	if(n < B){
		scalar_t vx = inputs[n * 3 + 0];
		scalar_t vy = inputs[n * 3 + 1];
		scalar_t vz = inputs[n * 3 + 2];
		
		if (vx == 0.f && vy == 0.f && vz == 0.f) {
			for (int iC = 0; iC < C; iC++) {
				outputs[iC * B + n] = fail_value[iC];
			}
			return;
		}
		scalar_t uv[2]; int cube_idx;
		int index_xy[3 * 4]; scalar_t kxky[2]; int flag;
		Compute_Cubemap_UV(vx, vy, vz, uv, &cube_idx);
		bool is_vertex = Compute_Seamless_Index(cube_idx, L, uv, index_xy, kxky, &flag);
		for (int iC = 0; iC < C; iC++) {
			scalar_t v00, v01, v10, v11;
			v00 = cubemap[((index_xy[0] * C + iC)* L + index_xy[2]) * L + index_xy[1]];
			v01 = cubemap[((index_xy[3] * C + iC)* L + index_xy[5]) * L + index_xy[4]];
			v10 = cubemap[((index_xy[6] * C + iC)* L + index_xy[8]) * L + index_xy[7]];
			if (is_vertex) {
				v11 = (v00 + v01 + v10) / 3.f;
			}else {
				v11 = cubemap[((index_xy[9] * C + iC)* L + index_xy[11]) * L + index_xy[10]];
			}
			outputs[iC * B + n] = (1 - kxky[1])*((1 - kxky[0])* v00 + kxky[0] * v01) + kxky[1] * ((1 - kxky[0])*v10 + kxky[0] * v11);
		}		
	}
}

template<typename scalar_t>
__global__ void Cubemap_Bilinear_Kernel(
	const scalar_t * __restrict__ inputs, // [B,3]
	const scalar_t * __restrict__ cubemap, // [6,C,L,L]
	const scalar_t * __restrict__ fail_value, // [C]
	scalar_t * __restrict__ outputs, // [C,B]
	const uint32_t B, const uint32_t C, const uint32_t L
){
	const uint32_t n = blockIdx.x * blockDim.x + threadIdx.x;
	if(n < B){
		scalar_t vx = inputs[n * 3 + 0];
		scalar_t vy = inputs[n * 3 + 1];
		scalar_t vz = inputs[n * 3 + 2];
		
		if (vx == 0.f && vy == 0.f && vz == 0.f) {
			for (int iC = 0; iC < C; iC++) {
				outputs[iC * B + n] = fail_value[iC];
			}
			return;
		}
		scalar_t uv[2]; int cube_idx;
		Compute_Cubemap_UV(vx, vy, vz, uv, &cube_idx);
#ifdef LEFT_TOP_AS_ORIGIN
		uv[1] = -uv[1];
#endif
		// -> pixel space
		uv[0] = (uv[0] * 0.5f + 0.5f) * scalar_t(L);
		uv[1] = (uv[1] * 0.5f + 0.5f) * scalar_t(L);
		int uy_0, ux_0;
		int uy_1, ux_1;
		ux_0 = int(floor(uv[0]-0.5f)); uy_0 = int(floor(uv[1]-0.5f));
		ux_1 = ux_0 + 1; uy_1 = uy_0 + 1;
		scalar_t kx = uv[0] - scalar_t(ux_0) - 0.5f;
		scalar_t ky = uv[1] - scalar_t(uy_0) - 0.5f;
		if (ux_0 < 0) ux_0 = 0;
		if (ux_0 >= L) ux_0 = L - 1;
		if (ux_1 < 0) ux_1 = 0;
		if (ux_1 >= L) ux_1 = L - 1;
		
		if (uy_0 < 0) uy_0 = 0;
		if (uy_0 >= L) uy_0 = L - 1;
		if (uy_1 < 0) uy_1 = 0;
		if (uy_1 >= L) uy_1 = L - 1;
		for (int iC = 0; iC < C; iC++) {
			scalar_t v00 = cubemap[((cube_idx * C + iC)* L + uy_0) * L + ux_0];
			scalar_t v01 = cubemap[((cube_idx * C + iC)* L + uy_0) * L + ux_1];
			scalar_t v10 = cubemap[((cube_idx * C + iC)* L + uy_1) * L + ux_0];
			scalar_t v11 = cubemap[((cube_idx * C + iC)* L + uy_1) * L + ux_1];
			outputs[iC * B + n] = (1 - ky)*((1 - kx)* v00 + kx * v01) + ky * ((1 - kx)*v10 + kx * v11); 
		}
	}
}

template<typename scalar_t>
__global__ void Cubemap_Nearest_Kernel(
	const scalar_t * __restrict__ inputs, // [B,3]	
	const scalar_t * __restrict__ cubemap, // [6,C,L,L]
	const scalar_t * __restrict__ fail_value, // [C]
	scalar_t * __restrict__ outputs, // [C,B]
	const uint32_t B, const uint32_t C, const uint32_t L
){
	const uint32_t n = blockIdx.x * blockDim.x + threadIdx.x;
	if(n < B){
		scalar_t vx = inputs[n * 3 + 0];
		scalar_t vy = inputs[n * 3 + 1];
		scalar_t vz = inputs[n * 3 + 2];
		
		if (vx == 0.f && vy == 0.f && vz == 0.f) {
			for (int iC = 0; iC < C; iC++) {
				outputs[iC * B + n] = fail_value[iC];
			}
			return;
		}
		scalar_t uv[2]; int cube_idx;
		Compute_Cubemap_UV(vx, vy, vz, uv, &cube_idx);
#ifdef LEFT_TOP_AS_ORIGIN
		uv[1] = -uv[1];
#endif
		// -> pixel space
		uv[0] = (uv[0] * 0.5f + 0.5f) * scalar_t(L);
		uv[1] = (uv[1] * 0.5f + 0.5f) * scalar_t(L);		
		int uy, ux;
		ux = int(uv[0]); uy = int(uv[1]);
		if (ux < 0) ux = 0;
		if (ux >= L) ux = L - 1;
		if (uy < 0) uy = 0;
		if (uy >= L) uy = L - 1;
		for (int iC = 0; iC < C; iC++) {
			outputs[iC * B + n] = cubemap[((cube_idx * C + iC)* L + uy) * L + ux];
		}
	}	
}


void cubemap_encode_forward(
	const at::Tensor inputs, const at::Tensor cubemap, const at::Tensor fail_value,
	at::Tensor outputs,const uint32_t interp, const uint32_t seamless,
	const uint32_t B, const uint32_t C, const uint32_t L 
){
	CHECK_CUDA(inputs);
	CHECK_CUDA(cubemap);
	CHECK_CUDA(fail_value);
	CHECK_CUDA(outputs);

	CHECK_CONTIGUOUS(inputs);
	CHECK_CONTIGUOUS(cubemap);
	CHECK_CONTIGUOUS(fail_value);
	CHECK_CONTIGUOUS(outputs);

	CHECK_IS_FLOATING(inputs);
	CHECK_IS_FLOATING(cubemap);
	CHECK_IS_FLOATING(fail_value);
	CHECK_IS_FLOATING(outputs);

	const at::cuda::OptionalCUDAGuard device_guard(device_of(cubemap));
	static constexpr uint32_t threads = 256;
	uint32_t blocks = uint32_t((B + threads - 1) / threads);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    cubemap.scalar_type(), "cubemap_encode_forward", ([&] {
		
		if(interp == 0){
			// nearest
			Cubemap_Nearest_Kernel<scalar_t><<<blocks, threads>>>(
				inputs.data_ptr<scalar_t>(),
				cubemap.data_ptr<scalar_t>(),
				fail_value.data_ptr<scalar_t>(),
				outputs.data_ptr<scalar_t>(),
				B,C,L				
			);
		}
        else{
			// bilinear
			if (seamless == 0){
				Cubemap_Bilinear_Kernel<scalar_t><<<blocks, threads>>>(
					inputs.data_ptr<scalar_t>(),
					cubemap.data_ptr<scalar_t>(),
					fail_value.data_ptr<scalar_t>(),
					outputs.data_ptr<scalar_t>(),
					B, C, L
				);
			}else{
				Cubemap_Bilinear_Seamless_Kernel<scalar_t><<<blocks, threads>>>(
					inputs.data_ptr<scalar_t>(),
					cubemap.data_ptr<scalar_t>(),
					fail_value.data_ptr<scalar_t>(),
					outputs.data_ptr<scalar_t>(),
					B, C, L
				);
			}
		}
    }));		
}

////////////////////////////////////
// backward

// TODO, how to support half ?
// See also grid_encoder

// atomicAdd for __half is slow (especially for large values), so we use __half2 if N_C % 2 == 0
// TODO: use float which is better than __half, if N_C % 2 != 0
//if (std::is_same<scalar_t, at::Half>::value && C % 2 == 0){
//	for(int iC = 0; iC < C ; iC+=2){
//		__half2 v = {(__half)grad_outputs[iC * B + n], (__half)grad_outputs[(iC + 1) * B + n)]};	
//		atomicAdd((__half2*)&grad_fail[iC], v);
//	}
//}else{
//	for (int iC = 0; iC < C; iC++) {
//		atomicAdd(&grad_fail[iC], grad_outputs[iC * B + n]);
//	}
//}

template<typename scalar_t>
__global__ void Cubemap_Bilinear_Seamless_Backward_Kernel(
	const scalar_t * __restrict__ grad_outputs, // [C,B]
	const scalar_t * __restrict__ cubemap, // [6,C,L,L]
	const scalar_t * __restrict__ inputs, // [B,3]
	scalar_t * __restrict__ grad_cubemap, // [6,C,L,L]
	scalar_t * __restrict__ grad_inputs, // [B,3]
	scalar_t * __restrict__ grad_fail, // [C]
	const uint32_t B, const uint32_t C, const uint32_t L
){
	const uint32_t n = blockIdx.x * blockDim.x + threadIdx.x;
	if (n < B){
		scalar_t vx = inputs[n * 3 + 0];
		scalar_t vy = inputs[n * 3 + 1];
		scalar_t vz = inputs[n * 3 + 2];

		if (vx == 0.f && vy == 0.f && vz == 0.f) {
			for (int iC = 0; iC < C; iC++) {
				atomicAdd(grad_fail + iC, grad_outputs[iC * B + n]);
			}
			grad_inputs[n * 3 + 0] = 0;
		    grad_inputs[n * 3 + 1] = 0;
		    grad_inputs[n * 3 + 2] = 0;
			return;
		}
		scalar_t uv[2]; int cube_idx;
		int index_xy[3 * 4]; scalar_t kxky[2]; int flag;
		Compute_Cubemap_UV(vx, vy, vz, uv, &cube_idx);
		bool is_vertex = Compute_Seamless_Index(cube_idx, L, uv, index_xy, kxky, &flag);
		scalar_t grad_view_[3] = { 0,0,0 };
		for (int iC = 0; iC < C; iC++) {
			scalar_t v00, v01, v10, v11;
			//scalar_t grad_input = grad_image[(n*C + iC)*h*w + idx];
			scalar_t grad_input = grad_outputs[iC * B + n];
			v00 = cubemap[((index_xy[0] * C + iC)* L + index_xy[2]) * L + index_xy[1]];
			v01 = cubemap[((index_xy[3] * C + iC)* L + index_xy[5]) * L + index_xy[4]];
			v10 = cubemap[((index_xy[6] * C + iC)* L + index_xy[8]) * L + index_xy[7]];
			if (is_vertex) {
				v11 = (v00 + v01 + v10) / 3.f;
				scalar_t extra_g = kxky[1] * kxky[0] / 3.f;
				atomicAdd(grad_cubemap + ((index_xy[0] * C + iC)* L + index_xy[2]) * L + index_xy[1], ((1 - kxky[1])* (1 - kxky[0]) + extra_g) * grad_input);
				atomicAdd(grad_cubemap + ((index_xy[3] * C + iC)* L + index_xy[5]) * L + index_xy[4], ((1 - kxky[1])*kxky[0] + extra_g) * grad_input);
				atomicAdd(grad_cubemap + ((index_xy[6] * C + iC)* L + index_xy[8]) * L + index_xy[7], ((kxky[1] * (1 - kxky[0])) + extra_g) * grad_input);
			}
			else {
				v11 = cubemap[((index_xy[9] * C + iC)* L + index_xy[11]) * L + index_xy[10]];
				atomicAdd(grad_cubemap + ((index_xy[0] * C + iC)* L + index_xy[2]) * L + index_xy[1], (1 - kxky[1])* (1 - kxky[0]) * grad_input);
				atomicAdd(grad_cubemap + ((index_xy[3] * C + iC)* L + index_xy[5]) * L + index_xy[4], (1 - kxky[1])* kxky[0] * grad_input);
				atomicAdd(grad_cubemap + ((index_xy[6] * C + iC)* L + index_xy[8]) * L + index_xy[7], kxky[1] * (1 - kxky[0]) * grad_input);
				atomicAdd(grad_cubemap + ((index_xy[9] * C + iC)* L + index_xy[11]) * L + index_xy[10], kxky[1] * kxky[0] * grad_input);
			}
			scalar_t loc_grad[2];// ux,uy
			loc_grad[0] = (1 - kxky[1]) * (v01 - v00) + kxky[1] * (v11 - v10);
			loc_grad[1] = (1 - kxky[0]) * (v10 - v00) + kxky[0] * (v11 - v01);
			loc_grad[0] *= 0.5f * scalar_t(L) * grad_input;
			loc_grad[1] *= 0.5f * scalar_t(L) * grad_input;
			if (flag & 0x01) {
				loc_grad[0] = -loc_grad[0];
			}
			if (flag & 0x04) {
				loc_grad[1] = -loc_grad[1];
			}
#ifdef LEFT_TOP_AS_ORIGIN
			loc_grad[1] = -loc_grad[1];
#endif
			scalar_t loc_grad_view[3];
			Compute_Cubemap_UV_Backward(
				cube_idx, vx, vy, vz, loc_grad, loc_grad_view
			);
			grad_view_[0] += loc_grad_view[0];
			grad_view_[1] += loc_grad_view[1];
			grad_view_[2] += loc_grad_view[2];
		}
		grad_inputs[n * 3 + 0] = grad_view_[0];
		grad_inputs[n * 3 + 1] = grad_view_[1];
		grad_inputs[n * 3 + 2] = grad_view_[2];
	}
}

template<typename scalar_t>
__global__ void Cubemap_Bilinear_Backward_Kernel(
	const scalar_t * __restrict__ grad_outputs, // [C,B]
	const scalar_t * __restrict__ cubemap, // [6,C,L,L]
	const scalar_t * __restrict__ inputs, // [B,3]
	scalar_t * __restrict__ grad_cubemap, // [6,C,L,L]
	scalar_t * __restrict__ grad_inputs, // [B,3]
	scalar_t * __restrict__ grad_fail, // [C]
	const uint32_t B, const uint32_t C, const uint32_t L
){
	const uint32_t n = blockIdx.x * blockDim.x + threadIdx.x;
	if (n < B){
		scalar_t vx = inputs[n * 3 + 0];
		scalar_t vy = inputs[n * 3 + 1];
		scalar_t vz = inputs[n * 3 + 2];
		
		if (vx == 0.f && vy == 0.f && vz == 0.f) {
			for (int iC = 0; iC < C; iC++) {
				atomicAdd(grad_fail + iC, grad_outputs[iC * B + n]);
			}
			grad_inputs[n * 3 + 0] = 0;
		    grad_inputs[n * 3 + 1] = 0;
		    grad_inputs[n * 3 + 2] = 0;
			return;
		}
		scalar_t uv[2]; int cube_idx;
		Compute_Cubemap_UV(vx, vy, vz, uv, &cube_idx);
#ifdef LEFT_TOP_AS_ORIGIN
		uv[1] = -uv[1];
#endif
		// -> pixel space
		uv[0] = (uv[0] * 0.5f + 0.5f) * scalar_t(L);
		uv[1] = (uv[1] * 0.5f + 0.5f) * scalar_t(L);
		int uy_0, ux_0;
		int uy_1, ux_1;
		ux_0 = int(floor(uv[0] - 0.5f)); uy_0 = int(floor(uv[1] - 0.5f));
		ux_1 = ux_0 + 1; uy_1 = uy_0 + 1;
		scalar_t kx = uv[0] - scalar_t(ux_0) - 0.5f;
		scalar_t ky = uv[1] - scalar_t(uy_0) - 0.5f;
		if (ux_0 < 0) ux_0 = 0;
		if (ux_0 >= L) ux_0 = L - 1;
		if (ux_1 < 0) ux_1 = 0;
		if (ux_1 >= L) ux_1 = L - 1;

		if (uy_0 < 0) uy_0 = 0;
		if (uy_0 >= L) uy_0 = L - 1;
		if (uy_1 < 0) uy_1 = 0;
		if (uy_1 >= L) uy_1 = L - 1;

		scalar_t grad_view_[3] = { 0,0,0 };
		for (int iC = 0; iC < C; iC++) {
			scalar_t v00 = cubemap[((cube_idx * C + iC)* L + uy_0) * L + ux_0];
			scalar_t v01 = cubemap[((cube_idx * C + iC)* L + uy_0) * L + ux_1];
			scalar_t v10 = cubemap[((cube_idx * C + iC)* L + uy_1) * L + ux_0];
			scalar_t v11 = cubemap[((cube_idx * C + iC)* L + uy_1) * L + ux_1];
			//scalar_t grad_input = grad_image[(n*C + iC)*h*w + idx];
			scalar_t grad_input = grad_outputs[iC * B + n];
			atomicAdd(grad_cubemap + ((cube_idx * C + iC)* L + uy_0) * L + ux_0, (1 - ky)* (1 - kx) * grad_input);
			atomicAdd(grad_cubemap + ((cube_idx * C + iC)* L + uy_0) * L + ux_1, (1 - ky)* kx * grad_input);
			atomicAdd(grad_cubemap + ((cube_idx * C + iC)* L + uy_1) * L + ux_0, ky * (1 - kx) * grad_input);
			atomicAdd(grad_cubemap + ((cube_idx * C + iC)* L + uy_1) * L + ux_1, ky * kx * grad_input);
			scalar_t loc_grad[2];// ux,uy
			loc_grad[0] = (1 - ky) * (v01 - v00) + ky * (v11 - v10);
			loc_grad[1] = (1 - kx) * (v10 - v00) + kx * (v11 - v01);
			loc_grad[0] *= 0.5f * scalar_t(L) * grad_input; 
			loc_grad[1] *= 0.5f * scalar_t(L) * grad_input;			
#ifdef LEFT_TOP_AS_ORIGIN
			loc_grad[1] = -loc_grad[1];
#endif
			scalar_t loc_grad_view[3];
			Compute_Cubemap_UV_Backward(
				cube_idx, vx, vy, vz, loc_grad, loc_grad_view
			);
			grad_view_[0] += loc_grad_view[0];
			grad_view_[1] += loc_grad_view[1];
			grad_view_[2] += loc_grad_view[2];
		}
		grad_inputs[n * 3 + 0] = grad_view_[0];
		grad_inputs[n * 3 + 1] = grad_view_[1];
		grad_inputs[n * 3 + 2] = grad_view_[2];
	}
}

template<typename scalar_t>
__global__ void Cubemap_Nearest_Backward_Kernel(
	const scalar_t * __restrict__ grad_outputs, // [C,B]
	const scalar_t * __restrict__ cubemap, // [6,C,L,L]
	const scalar_t * __restrict__ inputs, // [B,3]
	scalar_t * __restrict__ grad_cubemap, // [6,C,L,L]
	scalar_t * __restrict__ grad_fail, // [C]
	const uint32_t B, const uint32_t C, const uint32_t L
){
	const uint32_t n = blockIdx.x * blockDim.x + threadIdx.x;
	if (n < B){
		scalar_t vx = inputs[n * 3 + 0];
		scalar_t vy = inputs[n * 3 + 1];
		scalar_t vz = inputs[n * 3 + 2];
		
		if (vx == 0.f && vy == 0.f && vz == 0.f) {			
			for (int iC = 0; iC < C; iC++) {
				atomicAdd(grad_fail + iC, grad_outputs[iC * B + n]);
			}
			return;
		}		
		scalar_t uv[2]; int cube_idx;
		Compute_Cubemap_UV(vx, vy, vz, uv, &cube_idx);
#ifdef LEFT_TOP_AS_ORIGIN
		uv[1] = -uv[1];
#endif
		// -> pixel space
		uv[0] = (uv[0] * 0.5f + 0.5f) * scalar_t(L);
		uv[1] = (uv[1] * 0.5f + 0.5f) * scalar_t(L);
		int uy, ux;
		ux = int(uv[0]); uy = int(uv[1]);
		if (ux < 0) ux = 0;
		if (ux >= L) ux = L - 1;
		if (uy < 0) uy = 0;
		if (uy >= L) uy = L - 1;
		for (int iC = 0; iC < C; iC++) {
			atomicAdd(grad_cubemap + ((cube_idx * C + iC)*L + uy) * L + ux, grad_outputs[iC * B + n]);
		}		
	}
}


void cubemap_encode_backward(
	const at::Tensor grad_outputs, const at::Tensor inputs, const at::Tensor cubemap,
	at::Tensor grad_cubemap, at::Tensor grad_inputs, at::Tensor grad_fail,
	const uint32_t interp, const uint32_t seamless,
	const uint32_t B, const uint32_t C, const uint32_t L
){
	CHECK_CUDA(grad_outputs);
	CHECK_CUDA(inputs);
	CHECK_CUDA(cubemap);
	CHECK_CUDA(grad_cubemap);
	CHECK_CUDA(grad_inputs);
	CHECK_CUDA(grad_fail);

	CHECK_CONTIGUOUS(grad_outputs);
	CHECK_CONTIGUOUS(inputs);
	CHECK_CONTIGUOUS(cubemap);
	CHECK_CONTIGUOUS(grad_cubemap);
	CHECK_CONTIGUOUS(grad_inputs);
	CHECK_CONTIGUOUS(grad_fail);

	CHECK_IS_FLOATING(grad_outputs);
	CHECK_IS_FLOATING(inputs);
	CHECK_IS_FLOATING(cubemap);
	CHECK_IS_FLOATING(grad_cubemap);
	CHECK_IS_FLOATING(grad_inputs);
	CHECK_IS_FLOATING(grad_fail);

	const at::cuda::OptionalCUDAGuard device_guard(device_of(cubemap));
	static constexpr uint32_t threads = 256;
	uint32_t blocks = uint32_t((B + threads - 1) / threads);

    //AT_DISPATCH_FLOATING_TYPES_AND_HALF(
    //cubemap.scalar_type(), "cubemap_encode_backward", ([&] {
	#define scalar_t float
		// suppose grad_cubemap is zero tensor
		// suppose grad_inputs is empty tensor
		// suppose grad_fail is zero tensor
		if( interp == 0){
			grad_inputs.fill_(0); // NEAREST has no grad 
		}		
		if( interp == 0){
			// nearest
			Cubemap_Nearest_Backward_Kernel<scalar_t><<<blocks, threads>>>(
				grad_outputs.data<scalar_t>(), cubemap.data<scalar_t>(), inputs.data<scalar_t>(),
				grad_cubemap.data<scalar_t>(),	grad_fail.data<scalar_t>(),
				B, C, L
			);
		}
		else {
			// bilinear
			if (seamless == 0){
				Cubemap_Bilinear_Backward_Kernel<scalar_t><<<blocks, threads>>>(
					grad_outputs.data<scalar_t>(), cubemap.data<scalar_t>(), inputs.data<scalar_t>(),
					grad_cubemap.data<scalar_t>(), grad_inputs.data<scalar_t>(), grad_fail.data<scalar_t>(),
					B, C, L
				);
			}else{
				Cubemap_Bilinear_Seamless_Backward_Kernel<scalar_t><<<blocks, threads>>>(
					grad_outputs.data<scalar_t>(), cubemap.data<scalar_t>(), inputs.data<scalar_t>(),
					grad_cubemap.data<scalar_t>(), grad_inputs.data<scalar_t>(), grad_fail.data<scalar_t>(),
					B, C, L
				);				
			}
		}		
	#undef scalar_t
    //}));	
}




/*
template<typename scalar_t>
__global__ void Cubemap_Nearest_Kernel(
	const float * __restrict__ inputs, // [B,3]
	const scalar_t * __restrict__ embeddings, // [?,C]
	const int * __restrict__ offsets,
	scalar_t * __restrict__ outputs, // [L,C,B]
	const uint32_t B, const uint32_t C, const uint32_t L, const uint32_t nP,
) {
	const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;

	if (b >= B) return;

	const uint32_t level = blockIdx.y;
	const uint32_t H = offsets[level];
	const uint32_t offset = offsets[L + level];

	// locate
	inputs += b * 3;
	outputs += level * C * B + b; // l,?,b

	scalar_t vx = inputs[0];
	scalar_t vy = inputs[1];
	scalar_t vz = inputs[2];

	if (vx == 0 && vy == 0 && vz == 0) {
		for (uint32_t iC = 0; iC < C; iC++) {
			outputs[iC * B] = 0;
		}
		return;
	}

	scalar_t uv[2]; uint32_t cube_idx;
	Compute_Cubemap_UV(vx, vy, vz, uv, &cube_idx);
#ifdef LEFT_TOP_AS_ORIGIN
	uv[1] = -uv[1];
#endif

	// TODO
}
*/
