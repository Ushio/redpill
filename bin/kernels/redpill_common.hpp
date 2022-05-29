#pragma once

#if defined( __HIPCC__ ) || defined( __CUDACC__ )
#define IS_HOST 0
#define DEVICE __device__
#define DEVICE_INLINE __device__ inline
typedef unsigned int uint32_t;
#else
#define IS_HOST 1
#define DEVICE
#define DEVICE_INLINE inline
#include <cstdint>
#endif

#define SHARED_TENSOR_ROW 16

namespace rpml
{
	const int GPU_MAT_ALIGNMENT = 8;
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
	struct MLPTrainArg
	{
		GPUMat inputMat;
		GPUMat outputMat;
		GPUMat m_Ws[16];
		GPUMat m_Bs[16];
		GPUMat m_Is[16];
		int nLayer;
		int padd0;
		int padd1;
		int padd2;
	};

	DEVICE_INLINE int elem( int x, int y, GPUMat mat )
	{
		return mat.m_location + mat.m_paddedRow * x + y;
	}

	DEVICE_INLINE GPUMat allocateGPUMat( int *location, int row, int col )
	{
		GPUMat m;
		m.m_row = row;
		m.m_col = col;
		m.m_paddedRow = next_multiple( row, GPU_MAT_ALIGNMENT );
		m.m_location = *location;
		*location += m.m_paddedRow * m.m_col;
		return m;
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

	class HashGridEvaluator
	{
	public:
		DEVICE
		HashGridEvaluator( int dim ) : m_dim( dim ), m_bits( 0xFFFFFFFF )
		{
		}
		DEVICE
		bool moveNext()
		{
			m_bits++;
			return m_bits < ( 1 << m_dim );
		}
		DEVICE
		void evaluate( float* weight, uint32_t* hashValue, int resolution, float* input )
		{
			DimensionHasher hasher;
			float w = 1.0f;
			for( int d = 0; d < m_dim; ++d )
			{
				float x_in = input[d];

				float xf = x_in * resolution;
				uint32_t xi = xf;
				float u = xf - xi;

#if IS_HOST
				RPML_ASSERT( 0.0f <= u && u <= 1.0f );
#endif

				if( m_bits & ( 1 << d ) )
				{
					w *= u;
					hasher.add( xi + 1, d );
				}
				else
				{
					w *= 1.0f - u;
					hasher.add( xi, d );
				}
			}
			*weight = w;
			*hashValue = hasher.value();
		}

	private:
		int m_dim;
		uint32_t m_bits;
	};


}