#pragma once

#if defined( __HIPCC__ ) || defined( __CUDACC__ )
#define IS_HOST 0
#define DEVICE __device__
#define DEVICE_INLINE __device__ inline
typedef unsigned int uint32_t;

#define INTRIN_SINF( x ) __sinf( x )
#define INTRIN_COSF( x ) __cosf( x )
#define INTRIN_POWF( x, y ) __powf( x, y )
#else
#define IS_HOST 1
#define DEVICE
#define DEVICE_INLINE inline
#include <cstdint>

#define INTRIN_SINF( x ) std::sinf( x )
#define INTRIN_COSF( x ) std::cosf( x )
#define INTRIN_POWF( x, y ) std::powf( x, y )
#endif

#define SHARED_TENSOR_ROW 16

namespace rpml
{
	const int GPU_MAT_ALIGNMENT = 8;
	const float pi = 3.14159265358979323846f;

	enum class EncoderType
	{
		None,
		Frequency,
		MultiResolutionHash
	};

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
    struct MLPForwardArg
	{
		GPUMat inputMat;
		GPUMat outputMat;
		GPUMat m_Ws[16];
		GPUMat m_Bs[16];
		int nLayer;
		EncoderType encoder;
		int gridFeatureLocation;
		int padd2;
	};
	struct MLPTrainArg
	{
		GPUMat inputMat;
		GPUMat inputRefMat;
		GPUMat m_Ws[16];
		GPUMat m_Bs[16];
		GPUMat m_Is[16];
		int nLayer;
		EncoderType encoder;
		int gridFeatureLocation;
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

	const float ADAM_BETA1 = 0.9f;
	const float ADAM_BETA2 = 0.999f;
	const float ADAM_E = 10.0e-15f;
	struct Adam
	{
		float m_m;
		float m_v;

		DEVICE
		float optimize( float value, float g, float alpha, float beta1t, float beta2t )
		{
			float s = alpha;
			float m = m_m = ADAM_BETA1 * m_m + ( 1.0f - ADAM_BETA1 ) * g;
			float v = m_v = ADAM_BETA2 * m_v + ( 1.0f - ADAM_BETA2 ) * g * g;
			float m_hat = m / ( 1.0f - beta1t );
			float v_hat = v / ( 1.0f - beta2t );
			return value - s * m_hat / ( sqrt( v_hat ) + ADAM_E );
		}
	};

	// float encode_frequency( int inputDim, int xi, )
	DEVICE_INLINE
	int frequencyOutputDim( int inputDim, int N )
	{
		return inputDim * N * 2;
	}

	// [ sin( 2*pi*x ), cos( 2*pi*x ), sin( 4*pi*x ), cos( 4*pi*x )... sin( 2*pi*y ), cos( 2*pi*y )... ]
	struct Frequency
	{
		DEVICE
		Frequency( int inputDim, int xi, int N )
		:m_inputDim( inputDim )
		,m_xi( xi )
		,m_N( N )
		{
		}

		DEVICE
		int dimIwant() const
		{
			return m_xi / ( m_N * 2 );
		}

		DEVICE
		float encode( float v ) const
		{
			int baseDim = m_xi % ( m_N * 2 );
			int i = baseDim / 2;
			float k = 2.0f * pi * INTRIN_POWF( 2.0f, (float)i );
			float theta = k * v;
			if( baseDim % 2 ) { theta += pi * 0.5f; }
			return INTRIN_SINF( theta );
		}

		DEVICE
		int outputDim() const
		{
			return frequencyOutputDim( m_inputDim, m_N );
		}
		int m_inputDim;
		int m_xi;
		int m_N;
	};

	DEVICE_INLINE
	int multiResolutionHashOutputDim( int L, int F )
	{
		return L * F;
	}
}