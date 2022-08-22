#pragma once

#if defined( __HIPCC__ ) || defined( __CUDACC__ )
#define IS_HOST 0
#define DEVICE __device__
#define DEVICE_INLINE __device__ inline
typedef unsigned int uint32_t;
#define FLT_MAX 3.402823466e+38F // max value

#define INTRIN_SINF( x ) __sinf( x )
#define INTRIN_COSF( x ) __cosf( x )
#define INTRIN_POWF( x, y ) __powf( x, y )
#define INTRIN_EXPF( x ) __expf( x )
// #define INTRIN_SINF( x ) sinf( x )
// #define INTRIN_COSF( x ) cosf( x )
// #define INTRIN_POWF( x, y ) powf( x, y )
// #define INTRIN_EXPF( x ) expf( x )

#include "helper_math.h"

#else
#define IS_HOST 1
#define DEVICE
#define DEVICE_INLINE inline
#include <stdint.h>
#include <math.h>

#define INTRIN_SINF( x ) std::sinf( x )
#define INTRIN_COSF( x ) std::cosf( x )
#define INTRIN_POWF( x, y ) std::powf( x, y )
#define INTRIN_EXPF( x ) std::expf( x )
#endif

#define SHARED_TENSOR_ROW 16

#define ENABLE_DEMO_BIAS 0
#define ENABLE_NERF_BIAS 0

#define NERF_AVG_BATCH 128

#define SAMPLING_DISTANCE_PROPORTIONAL 1

namespace rpml
{
	const int GPU_MAT_ALIGNMENT = 16; // assume == Mat::ROW_MULTIPLE for convinience 
	const int WMMA_ALIGNMENT = 16;
	constexpr float pi = 3.14159265358979323846f;

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

	DEVICE_INLINE float hash1( uint32_t n ) 
	{
		// integer hash copied from Hugo Elias
		n = (n << 13U) ^ n;
		n = n * (n * n * 15731U + 789221U) + 1376312589U;
		return float( n & uint32_t(0x7fffffffU))/float(0x7fffffff);
	}

#if IS_HOST == 0
	DEVICE_INLINE float3 pcg3df( uint3 v )
	{
		v = v * 1664525u + 1013904223u;
		v.x += v.y * v.z;
		v.y += v.z * v.x;
		v.z += v.x * v.y;
		
		v.x = v.x ^ ( v.x >> 16u );
		v.y = v.y ^ ( v.y >> 16u );
		v.z = v.z ^ ( v.z >> 16u );

		v.x += v.y * v.z;
		v.y += v.z * v.x;
		v.z += v.x * v.y;
		
		return make_float3( 
			v.x & uint32_t( 0x7fffffffU ),
			v.y & uint32_t( 0x7fffffffU ),
			v.z & uint32_t( 0x7fffffffU )
		) / make_float3( (float)0x7fffffffU );
	}
#endif

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
		//if( mat.m_col <= x )
		//{
		//	printf( "invalid: col=%d x=%d\n", mat.m_col, x );
		//}
		//if( mat.m_paddedRow <= y )
		//{
		//	printf( "invalid: row=%d y=%d\n", mat.m_paddedRow, y );
		//}
		return mat.m_location + mat.m_paddedRow * x + y;
	}

	DEVICE_INLINE GPUMat allocateGPUMat( int *location, int row, int col, int colPadding = 1 )
	{
		GPUMat m;
		m.m_row = row;
		m.m_col = col;
		m.m_paddedRow = next_multiple( row, GPU_MAT_ALIGNMENT );
		m.m_location = *location;

		int paddedCol = next_multiple( col, colPadding );

		*location += m.m_paddedRow * paddedCol;
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
	class SequencialHasher
	{
	public:
		DEVICE
		void add( uint32_t xs, uint32_t resolution )
		{
			m_h += m_base * xs;
			m_base *= resolution;
		}
		DEVICE
		uint32_t value() const { return m_h; }

	private:
		uint32_t m_base = 1;
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

	// batch sequencial load
	DEVICE
	float BSL4( float* matBuffer, int x, int y, GPUMat m, float vals[4] )
	{
		if( ( y % 4 ) == 0 )
		{
#if IS_HOST
			for( int i = 0; i < 4; i++ )
			{
				vals[i] = matBuffer[elem( x, y + i, m )];
			}
#else
			float4 vec = *(float4*)&matBuffer[elem( x, y, m )];
			vals[0] = vec.x;
			vals[1] = vec.y;
			vals[2] = vec.z;
			vals[3] = vec.w;
#endif
		}
		return vals[y % 4];
	}

	const float ADAM_BETA1 = 0.9f;
	const float ADAM_BETA2 = 0.99f;
	const float ADAM_E = 1.0e-15f;
	struct Adam
	{
		float m_m;
		float m_v;

		DEVICE
		float optimize( float value, float g, float alpha, float beta1t, float beta2t )
		{
			float s = alpha;
			float m = ADAM_BETA1 * m_m + ( 1.0f - ADAM_BETA1 ) * g;
			float v = ADAM_BETA2 * m_v + ( 1.0f - ADAM_BETA2 ) * g * g;
			m_m = m;
			m_v = v;
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

	DEVICE_INLINE float huber_loss( float y, float y_ref, float delta ) 
	{
		float d = y - y_ref;
		return fabs( d ) < delta ? d : copysignf( delta, d );
	}

	const int NERF_DENSITY_LAYER_BEG = 0;
	const int NERF_DENSITY_LAYER_END = 2;
	const int NERF_COLOR_LAYER_BEG = 2;
	const int NERF_COLOR_LAYER_END = 5;
	const int MLP_STEP = 1024;
	const float MLP_CONSTANT_STEP_SIZE = 1.73205080756887729353f /*sqrt(3)*/ / MLP_STEP;

	const int NERF_OCCUPANCY_GRID_MIN_RES = 128;
	const int NERF_OCCUPANCY_GRID_T = NERF_OCCUPANCY_GRID_MIN_RES * NERF_OCCUPANCY_GRID_MIN_RES * NERF_OCCUPANCY_GRID_MIN_RES;
	const int NERF_OCCUPANCY_GRID_L = 1;
	

	struct NeRFInput
	{
		float ro[3]; float pad0;
		float rd[3]; float pad1;
	};
	struct NeRFOutput
	{
		float color[4];
	};
	struct NeRFMarching
	{
		int beg;
		int end;
	};
	struct NeRFRay
	{
		int eval_beg;
		int eval_end;
	};
    struct NeRFForwardArg
	{
		GPUMat inputMat;
		GPUMat outputMat;
		GPUMat dirMat;
		GPUMat m_Ws[16];
		GPUMat m_Bs[16];
		int gridFeatureLocation;
		int padd0;
		int padd1;
		int padd2;
	};
	struct NeRFTrainArg
	{
		GPUMat inputMat;
		GPUMat outputMat;
		GPUMat dirMat;
		GPUMat m_Ws[16];
		GPUMat m_Bs[16];
		GPUMat m_Is[16];
		GPUMat m_DoAs[16]; // derivative of activation
		int nLayer;
		int gridFeatureLocation;
		int padd1;
		int padd2;
	};
#if IS_HOST == 0
	// DEVICE_INLINE
	// float3 operator*( float3 a, float3 b )
	// {
	// 	float3 r;
	// 	r.x = a.x * b.x;
	// 	r.y = a.y * b.y;
	// 	r.z = a.z * b.z;
	// 	return r;
	// }
	// DEVICE_INLINE
	// float3 operator*( float3 a, float b )
	// {
	// 	float3 r;
	// 	r.x = a.x * b;
	// 	r.y = a.y * b;
	// 	r.z = a.z * b;
	// 	return r;
	// }
	// 	DEVICE_INLINE
	// float3 operator*( float a, float3 b )
	// {
	// 	float3 r;
	// 	r.x = b.x * a;
	// 	r.y = b.y * a;
	// 	r.z = b.z * a;
	// 	return r;
	// }
	// DEVICE_INLINE
	// float3 operator-( float3 a, float3 b )
	// {
	// 	float3 r;
	// 	r.x = a.x - b.x;
	// 	r.y = a.y - b.y;
	// 	r.z = a.z - b.z;
	// 	return r;
	// }
	// DEVICE_INLINE
	// float3 operator+( float3 a, float3 b )
	// {
	// 	float3 r;
	// 	r.x = a.x + b.x;
	// 	r.y = a.y + b.y;
	// 	r.z = a.z + b.z;
	// 	return r;
	// }
	// DEVICE_INLINE
	// float3 operator/( float3 a, float3 b )
	// {
	// 	float3 r;
	// 	r.x = a.x / b.x;
	// 	r.y = a.y / b.y;
	// 	r.z = a.z / b.z;
	// 	return r;
	// }
	DEVICE_INLINE
	float3 fmaxf3( float3 a, float3 b )
	{
		float3 r;
		r.x = fmax( a.x, b.x );
		r.y = fmax( a.y, b.y );
		r.z = fmax( a.z, b.z );
		return r;
	}
	DEVICE_INLINE
	float3 fminf3( float3 a, float3 b )
	{
		float3 r;
		r.x = fmin( a.x, b.x );
		r.y = fmin( a.y, b.y );
		r.z = fmin( a.z, b.z );
		return r;
	}
	DEVICE_INLINE
	float3 clampf3( float3 x, float3 a, float3 b )
	{
		return fmaxf3(a, fminf3(b, x));
	}
	DEVICE_INLINE
	float fclamp( float x, float a, float b )
	{
		return fmin( fmax( x, a ), b );
	}
	DEVICE_INLINE
	float compMin( float3 v )
	{
		return fmin( fmin( v.x, v.y ), v.z );
	}
	DEVICE_INLINE
	float compMax( float3 v )
	{
		return fmax( fmax( v.x, v.y ), v.z );
	}
	DEVICE_INLINE
	float2 slabs( float3 p0, float3 p1, float3 ro, float3 one_over_rd )
	{
		float3 t0 = ( p0 - ro ) * one_over_rd;
		float3 t1 = ( p1 - ro ) * one_over_rd;

		float3 tmin = fminf3( t0, t1 );
		float3 tmax = fmaxf3( t0, t1 );
		float region_min = compMax( tmin );
		float region_max = compMin( tmax );

		region_min = fmax( region_min, 0.0f );

		return make_float2( region_min, region_max );
	}
	DEVICE_INLINE
	float3 safe_inv_rd( float3 rd )
	{
		return clampf3( make_float3( 1.0f, 1.0f, 1.0f ) / rd, make_float3( -FLT_MAX, -FLT_MAX, -FLT_MAX ), make_float3( FLT_MAX, FLT_MAX, FLT_MAX ) );
	}
	DEVICE_INLINE
	float nerfDensityActivation(float x)
	{
		return INTRIN_EXPF( x );
	}
	DEVICE_INLINE
	float nerfDensityActivationDerivative( float x )
	{
		return INTRIN_EXPF( fclamp( x, -15.0f, 15.0f ) );
	}
	DEVICE_INLINE
	float nerfRgbActivation( float x )
	{
		return 1.0f / ( 1.0f + INTRIN_EXPF( -x ) );
	}
	DEVICE_INLINE
	float nerfRgbActivationDrivativeY( float Y )
	{
		return Y * ( 1.0f - Y );
	}
#endif

	DEVICE_INLINE
	bool isOccupied( float px, float py, float pz, float dt, float* occupancyGrid, float avg )
	{
		const float sqrt3 = 1.73205080756887729353f;
		if( sqrt3 / NERF_OCCUPANCY_GRID_MIN_RES < dt )
		{
			return true;
		}

		SequencialHasher hasher;
		hasher.add( px * NERF_OCCUPANCY_GRID_MIN_RES, NERF_OCCUPANCY_GRID_MIN_RES );
		hasher.add( py * NERF_OCCUPANCY_GRID_MIN_RES, NERF_OCCUPANCY_GRID_MIN_RES );
		hasher.add( pz * NERF_OCCUPANCY_GRID_MIN_RES, NERF_OCCUPANCY_GRID_MIN_RES );

		uint32_t index = hasher.value() % NERF_OCCUPANCY_GRID_T;
		return avg < occupancyGrid[index];
	}

	DEVICE_INLINE float srgbLinear2Gamma( float c )
	{
		if( c < 0.0031308f )
		{
			return c * 12.92f;
		}
		return INTRIN_POWF( c * 1.055f, 1.0f / 2.4f ) - 0.055f;
	}

	DEVICE_INLINE void adobeRGB2sRGB( float* outR, float* outG, float* outB, float r, float g, float b, float linearScaling )
	{
		r = INTRIN_POWF( r, 563.0f / 256.0f );
		g = INTRIN_POWF( g, 563.0f / 256.0f );
		b = INTRIN_POWF( b, 563.0f / 256.0f );

		float lr = 1.39821f * r - 0.3998298f * g - 0.000039774f * b;
		float lg = 0.000136554f * r + 0.999946f * g - 0.000006455f * b;
		float lb = 0.000033869f * r - 0.0429264f * g + 1.04297f * b;

		*outR = srgbLinear2Gamma( lr * linearScaling );
		*outG = srgbLinear2Gamma( lg * linearScaling );
		*outB = srgbLinear2Gamma( lb * linearScaling );
	}
}