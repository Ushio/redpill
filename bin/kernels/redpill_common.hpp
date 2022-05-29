#pragma once

#if defined( __HIPCC__ ) || defined( __CUDACC__ )
#define DEVICE __device__
#define DEVICE_INLINE __device__ inline
typedef unsigned int uint32_t;
#else
#define DEVICE
#define DEVICE_INLINE inline
#include <cstdint>
#endif

#define SHARED_TENSOR_ROW 16

namespace rpml
{
	const float pi = 3.14159265358979323846f;

    DEVICE_INLINE int div_round_up( int val, int divisor )
    {
        return ( val + divisor - 1 ) / divisor;
    }
    DEVICE_INLINE int next_multiple( int val, int divisor )
    {
        return div_round_up( val, divisor ) * divisor;
    }
    struct GPUMat
	{
		int m_row; // = inputs
		int m_paddedRow;
		int m_col; // = outputs
		int m_location;
	};
    struct MLPForwardFusedArg
	{
		GPUMat inputMat;
		GPUMat outputMat;
		GPUMat m_Ws[16];
		GPUMat m_Bs[16];
		int nLayer;
		int padd0;
		int padd1;
		int padd2;
	};
	struct MLPEncodingArg
	{
		int mode;
		int frequency_N;
		int padd1;
		int padd2;
	};

	DEVICE_INLINE int elem( int x, int y, GPUMat mat )
	{
		return mat.m_location + mat.m_paddedRow * x + y;
	}

	// 438976903 PRIMES[0] = 1 is very good for perf?
	constexpr uint32_t PRIMES[7] = { 1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737 };
	class DimensionHasher
	{
	public:
		DEVICE
		void add( uint32_t xs, int d )
		{
			m_h ^= xs * PRIMES[ d ];
		}

		DEVICE
		uint32_t value() const { return m_h; }
	private:
		uint32_t m_h = 0;
	};
}