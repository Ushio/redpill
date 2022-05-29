#pragma once

#include <memory>
#include <string>
#include <random>
#include <mutex>
#include <algorithm>
#include <atomic>
#include <chrono>

#include "redpill_common.hpp"

#include "prth.hpp"

#if defined( RPML_DISABLE_ASSERT )
#define RPML_ASSERT( ExpectTrue ) ;
#else
#include <intrin.h>
#define RPML_ASSERT( ExpectTrue ) \
	if( ( ExpectTrue ) == 0 )     \
	{                             \
		__debugbreak();           \
	}
#endif

#define FOR_EACH_ELEMENT( m, ix, iy )     \
	for( int ix = 0; ix < ( m ).col(); ix++ ) \
		for( int iy = 0; iy < ( m ).row(); iy++ )

#define ENABLE_SIMD 1

#if ENABLE_SIMD
#include <immintrin.h>
#endif

#define ENABLE_TRACE 0
#if ENABLE_TRACE
#include "prtr.hpp"
#endif

#define NOMINMAX
#include <Windows.h>

namespace rpml
{
	enum class OptimizerType
	{
		SGD,
		Adam,
	};
	enum class InitializationType
	{
		Xavier,
		He
	};
	enum class ActivationType
	{
		ReLU,
		LeakyReLU,
		Tanh,
		Sigmoid
	};
	enum class EncoderType
	{
		None,
		Frequency,
		MultiResolutionHash
	};


	//inline int div_round_up( int val, int divisor )
	//{
	//	return ( val + divisor - 1 ) / divisor;
	//}
	//inline int next_multiple( int val, int divisor )
	//{
	//	return div_round_up( val, divisor ) * divisor;
	//}

	constexpr float abs_constant( float x )
	{
		return x < 0.0f ? -x : x;
	}
	constexpr float newton_sqrt_r( float xn, float a, int e )
	{
		float xnp1 = xn - ( xn * xn - a ) * 0.5f / xn;
		float e0 = abs_constant( xn * xn - a );
		float e1 = abs_constant( xnp1 * xnp1 - a );
		return ( e1 < e0 )
				   ? newton_sqrt_r( xnp1, a, e )
				   : ( e < 4 /* magic number */ ? newton_sqrt_r( xnp1, a, e + 1 ) : xn );
	}
	constexpr float newton_sqrt( float x )
	{
		bool valid =
			0.0f <= x &&
			x < std::numeric_limits<float>::infinity() &&
			x == x; // nan
		return valid
				   ? ( x == 0.0f ? 0.0f : newton_sqrt_r( x, x, 0 ) )
				   : std::numeric_limits<double>::quiet_NaN();
	}

	void sh_L4( float v[16], float x, float y, float z )
	{
		constexpr float rpi = newton_sqrt( pi );
		constexpr float r3 = newton_sqrt( 3.0f );
		constexpr float r15 = newton_sqrt( 15.0f );
		constexpr float r5 = newton_sqrt( 5.0f );
		constexpr float r2 = newton_sqrt( 2.0f );
		constexpr float r35 = newton_sqrt( 35.0f );
		constexpr float r105 = newton_sqrt( 105.0f );
		constexpr float r21 = newton_sqrt( 21.0f );
		constexpr float r7 = newton_sqrt( 7 );

		const float xy = x * y;
		const float yz = y * z;
		const float xz = x * z;
		const float xx = x * x;
		const float yy = y * y;
		const float zz = z * z;
		const float xyz = xy * z;

		// L=0
		v[0] = +1.0f / ( 2.0f * rpi ); // M=0

		// L=1
		v[1] = -( r3 / ( 2.0f * rpi ) ) * y;
		v[2] = +( r3 / ( 2.0f * rpi ) ) * z;
		v[3] = -( r3 / ( 2.0f * rpi ) ) * x;

		// L=2
		v[4] = +( r15 / ( 2.0f * rpi ) ) * xy;
		v[5] = -( r15 / ( 2.0f * rpi ) ) * yz;
		v[6] = +( r5 / ( 4.0f * rpi ) ) * ( 3.0f * z * z - 1.0f );
		v[7] = -( r15 / ( 2.0f * rpi ) ) * xz;
		v[8] = +( r15 / ( 4.0f * rpi ) ) * ( xx - yy );

		// L=3
		v[9] = -( r2 * r35 / ( 8.0f * rpi ) ) * y * ( 3.0f * xx - yy );
		v[10] = +( r105 / ( 2.0f * rpi ) ) * xyz;
		v[11] = -( r2 * r21 / ( 8.0f * rpi ) ) * y * ( -1.0f + 5.0f * zz );
		v[12] = +( r7 / ( 4.0f * rpi ) ) * z * ( 5.0f * z * z - 3.0f );
		v[13] = -( r2 * r21 / ( 8.0f * rpi ) ) * x * ( -1.0f + 5.0f * zz );
		v[14] = +( r105 / ( 4.0f * rpi ) ) * ( xx - yy ) * z;
		v[15] = -( r2 * r35 / ( 8.0f * rpi ) ) * x * ( xx - 3.0f * yy );
	}

	namespace details
	{
		class Stopwatch
		{
		public:
			using clock = std::chrono::steady_clock;
			Stopwatch() : _started( clock::now() ) {}

			// seconds
			double elapsed() const
			{
				auto microseconds = std::chrono::duration_cast<std::chrono::microseconds>( clock::now() - _started ).count();
				return (double)microseconds * 0.001 * 0.001;
			}

		private:
			clock::time_point _started;
		};
	}

	struct Mat
	{
		enum
		{
			ROW_MULTIPLE = 8
		};
		Mat() {}
		Mat( int row, int col ) 
			: m_row( row ) 
			, m_paddedRow( next_multiple( row, ROW_MULTIPLE ) )
			, m_col( col )
			, m_data( col * next_multiple( row, ROW_MULTIPLE ) ) 
		{
		}
		Mat( int row, int col, const std::vector<float>& data /* col major */ ) 
			: m_row( row )
			, m_paddedRow( next_multiple( row, ROW_MULTIPLE ) )
			, m_col( col )
			, m_data( col * next_multiple( row, ROW_MULTIPLE ) ) 
		{
			RPML_ASSERT( m_row * m_col == data.size() ); 

			FOR_EACH_ELEMENT( *this, ix, iy )
			{
				( *this )( ix, iy ) = data[ix * row + iy];
			}
		};

		void setShape( int row, int col )
		{
			if( m_row == row && m_col == col )
			{
				return;
			}

			m_row = row;
			m_paddedRow = next_multiple( row, ROW_MULTIPLE );
			m_col = col;
			m_data.resize( col * m_paddedRow );
		}
		void fill( float x )
		{
#if ENABLE_SIMD
			__m256 v = _mm256_set1_ps( x );
			for( int i = 0; i < m_data.size(); i += 8 )
			{
				_mm256_storeu_ps( &m_data[i], v );
			}
#else
			for( int i = 0; i < m_data.size(); i++ )
			{
				m_data[i] = x ;
			}
#endif
		}

		float& operator()( int x, int y )
		{
			RPML_ASSERT( 0 <= y && y < m_row );
			RPML_ASSERT( 0 <= x && x < m_col );
			return m_data[m_paddedRow * x + y];
		}
		const float& operator()( int x, int y ) const
		{
			RPML_ASSERT( 0 <= y && y < m_row );
			RPML_ASSERT( 0 <= x && x < m_col );
			return m_data[m_paddedRow * x + y];
		}
		int row() const { return m_row; }
		int paddedRow() const { return m_paddedRow; }
		int col() const { return m_col; }

		void swap( Mat& rhs )
		{
			std::swap( m_data, rhs.m_data );
			std::swap( m_row, rhs.m_row );
			std::swap( m_paddedRow, rhs.m_paddedRow );
			std::swap( m_col, rhs.m_col );
		}
		
		const float* data() const
		{
			return m_data.data();
		}
		float* data()
		{
			return m_data.data();
		}
		int bytes() const
		{
			return m_data.size() * sizeof( float );
		}
		bool noPadding() const
		{
			return m_row == m_paddedRow;
		}
	private:
		std::vector<float> m_data;
		int m_row = 0;
		int m_paddedRow = 0;
		int m_col = 0;
	};

	inline void transpose( Mat *r, const Mat& m )
	{
		( *r ).setShape( m.col(), m.row() );
		FOR_EACH_ELEMENT( *r, ix, iy )
		{
			( *r )( ix, iy ) = m( iy, ix );
		}
	}

	inline Mat fromRowMajor( int row, int col, std::initializer_list<float> init )
	{
		Mat t;
		transpose( &t, Mat( col, row, init ) );
		return t;
	}
	inline Mat fromColMajor( int row, int col, std::initializer_list<float> init )
	{
		return Mat( row, col, init );
	}

	inline void mulNaive( Mat *r, const Mat& ma, const Mat& mb )
	{
		RPML_ASSERT( ma.col() == mb.row() );

		( *r ).setShape( ma.row(), mb.col() );

		FOR_EACH_ELEMENT( ( *r ), ix, iy )
		{
			const int n = ma.col();
			float v = 0.0f;
			for( int i = 0; i < n; i++ )
			{
				v += ma( i, iy ) * mb( ix, i );
			}
			( *r )( ix, iy ) = v;
		}
	}

#if ENABLE_SIMD
	inline void mulSIMD( Mat* r, const Mat& ma, const Mat& mb )
	{
		RPML_ASSERT( ma.col() == mb.row() );

		( *r ).setShape( ma.row(), mb.col() );

		int ix = 0;
		while( ix < ( *r ).col() )
		{
			int iy = 0;
			while( iy < ( *r ).row() )
			{
// Loop tiling - https://stackoverflow.com/questions/59009628/tiled-matrix-multiplication-using-avx
#if 1
				if( iy + 32 <= ( *r ).row() )
				{
					__m256 v0 = _mm256_setzero_ps();
					__m256 v1 = _mm256_setzero_ps();
					__m256 v2 = _mm256_setzero_ps();
					__m256 v3 = _mm256_setzero_ps();

					const int n = ma.col();
					for( int i = 0; i < n; i++ )
					{
						__m256 lhs0 = _mm256_loadu_ps( &ma( i, iy ) );
						__m256 lhs1 = _mm256_loadu_ps( &ma( i, iy + 8 ) );
						__m256 lhs2 = _mm256_loadu_ps( &ma( i, iy + 16 ) );
						__m256 lhs3 = _mm256_loadu_ps( &ma( i, iy + 24 ) );

						__m256 rhs = _mm256_set1_ps( mb( ix, i ) );

						v0 = _mm256_fmadd_ps( lhs0, rhs, v0 );
						v1 = _mm256_fmadd_ps( lhs1, rhs, v1 );
						v2 = _mm256_fmadd_ps( lhs2, rhs, v2 );
						v3 = _mm256_fmadd_ps( lhs3, rhs, v3 );
					}

					_mm256_storeu_ps( &( ( *r )( ix, iy ) ), v0 );
					_mm256_storeu_ps( &( ( *r )( ix, iy + 8 ) ), v1 );
					_mm256_storeu_ps( &( ( *r )( ix, iy + 16 ) ), v2 );
					_mm256_storeu_ps( &( ( *r )( ix, iy + 24 ) ), v3 );

					iy += 32;
				}
				else
#endif
				{ // naiive simd
					__m256 v = _mm256_setzero_ps();

					const int n = ma.col();
					for( int i = 0; i < n; i++ )
					{
						__m256 lhs = _mm256_loadu_ps( &ma( i, iy ) );
						__m256 rhs = _mm256_set1_ps( mb( ix, i ) );
						v = _mm256_fmadd_ps( lhs, rhs, v );
					}

					_mm256_storeu_ps( &( ( *r )( ix, iy ) ), v );
					iy += 8;
				}
			}
			ix++;
		}
	}
#endif
	inline void mul( Mat* r, const Mat& ma, const Mat& mb )
	{
#if ENABLE_SIMD
		mulSIMD( r, ma, mb );
#else 
		mulNaive( r, ma, mb );
#endif
	}

	inline void add( Mat* r, const Mat& x )
	{
		RPML_ASSERT( (*r).col() == x.col() );
		RPML_ASSERT( (*r).row() == x.row() );

		FOR_EACH_ELEMENT( *r, ix, iy )
		{
			( *r )( ix, iy ) += x( ix, iy );
		}
	}
	inline void sub( Mat* r, const Mat& x )
	{
		RPML_ASSERT( ( *r ).col() == x.col() );
		RPML_ASSERT( ( *r ).row() == x.row() );

		FOR_EACH_ELEMENT( *r, ix, iy )
		{
			( *r )( ix, iy ) -= x( ix, iy );
		}
	}

	inline void addVectorForEachRow( Mat *r, const Mat& v )
	{
		RPML_ASSERT( ( *r ).col() == v.col() );
		FOR_EACH_ELEMENT( *r, ix, iy )
	    {
			( *r )( ix, iy ) += v( ix, 0 );
	    }
	}

	inline void vertialSum( Mat *r, const Mat& m )
	{
		( *r ).setShape( 1, m.col() );
		for( int ix = 0; ix < m.col(); ix++ )
		{
			float s = 0.0f;
			for( int iy = 0; iy < m.row(); iy++ )
			{
				s += m( ix, iy );
			}
			( *r )( ix, 0 ) = s;
		}
	}
	inline void sliceH( Mat *r, const Mat& x, int beg, int end )
	{
		RPML_ASSERT( 0 <= beg );
		RPML_ASSERT( end <= x.row() );
		RPML_ASSERT( beg <= end );

		int localN = end - beg;
		( *r ).setShape( localN, x.col() );
		for( int i = 0; i < localN; i++ )
		{
			for( int j = 0; j < x.col(); ++j )
			{
				( *r )( j, i ) = x( j, beg + i );
			}
		}
	}
	inline void concatH( Mat* r, const Mat& x, int beg, int end )
	{
		RPML_ASSERT( 0 <= beg );
		RPML_ASSERT( end <= ( *r ).row() );
		RPML_ASSERT( beg <= end );
		RPML_ASSERT( x.row() == end - beg );

		int localN = end - beg;
		for( int i = 0; i < localN; i++ )
		{
			for( int j = 0; j < x.col(); ++j )
			{
				( *r )( j, beg + i ) = x( j, i );
			}
		}
	}

	inline float minss( float a, float b )
	{
		return ( a < b ) ? a : b;
	}
	inline float maxss( float a, float b )
	{
		return ( b < a ) ? a : b;
	}
	inline float clampss( float x, float a, float b )
	{
		return minss( maxss( x, a ), b );
	}

	// atomic
	inline int32_t as_int32( float f )
	{
		return *reinterpret_cast<int32_t*>( &f );
	}
	inline float as_float( int32_t u )
	{
		return *reinterpret_cast<float*>( &u );
	}

	inline void atomAdd( std::atomic<float>* p, float v )
	{
		float curVal = p->load();
		float newVal;
		do
		{
			newVal = curVal + v;
		}
		while( p->compare_exchange_weak( curVal, newVal ) == false );
	}
	inline void atomAdd( float* p, float v )
	{
		// https://docs.microsoft.com/en-us/windows/win32/winprog/windows-data-types
		// LONG: A 32 - bit signed integer.The range is - 2147483648 through 2147483647 decimal.

		float curVal = ( *p );
		LONG orig;
		for( ;; )
		{
			float newVal = curVal + v;
			orig = InterlockedCompareExchange( (LONG*)p, as_int32( newVal ), as_int32( curVal ) );
			if( orig == as_int32( curVal ) )
			{
				break;
			}
			curVal = as_float( orig );
		};
	}
	inline float atomicExchange( float* p, float v )
	{
		return as_float( InterlockedExchange( (LONG*)p, v ) );
	}

	class Optimizer
	{
	public:
	    virtual ~Optimizer() {}

		virtual void initialize( int row, int col ) = 0;
		virtual void optimize( Mat* parameter, const Mat& gradient, int nElement ) = 0;
	};

	class OptimizerSGD : public Optimizer
	{
	public:
		OptimizerSGD( float alpha ) : m_alpha( alpha )
		{
		}
		virtual void initialize( int row, int col )
		{
		}
		virtual void optimize( Mat* parameter, const Mat& gradient, int nElement )
		{
			float s = m_alpha / nElement;
			FOR_EACH_ELEMENT( ( *parameter ), ix, iy )
			{
				( *parameter )( ix, iy ) -= gradient( ix, iy ) * s;
			}
		}
	private:
		float m_alpha;
	};
	class OptimizerAdam : public Optimizer
	{
	public:
		OptimizerAdam( float alpha, float beta1 = 0.9f, float beta2 = 0.999f, float e = 10.0e-15f ) 
			: m_alpha( alpha ), m_beta1( beta1 ), m_beta2( beta2 ), m_beta1t( 1.0f ), m_beta2t( 1.0f ), m_e( e )
		{
		}
		virtual void initialize( int row, int col )
		{
			m_m.setShape( row, col );
			m_v.setShape( row, col );
			m_m.fill( 0.0f );
			m_v.fill( 0.0f );
		}
		virtual void optimize( Mat* parameter, const Mat& gradient, int nElement )
		{
			m_beta1t *= m_beta1;
			m_beta2t *= m_beta2;

			float s = m_alpha / nElement;
			FOR_EACH_ELEMENT( ( *parameter ), ix, iy )
			{
				float g = gradient( ix, iy );
				float m = m_m( ix, iy ) = m_beta1 * m_m( ix, iy ) + ( 1.0f - m_beta1 ) * g;
				float v = m_v( ix, iy ) = m_beta2 * m_v( ix, iy ) + ( 1.0f - m_beta2 ) * g * g;
				float m_hat = m / ( 1.0f - m_beta1t );
				float v_hat = v / ( 1.0f - m_beta2t );
				( *parameter )( ix, iy ) = ( *parameter )( ix, iy ) - s * m_hat / ( std::sqrt( v_hat ) + m_e );
			}
		}

		// must call this before optimizeHashGridRow
		void incrementParametereHashGridRow()
		{
			m_beta1t *= m_beta1;
			m_beta2t *= m_beta2;
		}
		// gradient will be cleared.
		void atomicOptimizeHashGridRow( Mat* parameter, Mat* gradient, int nElement, int iy )
		{
			float s = m_alpha / nElement;
			for( int ix = 0; ix < ( *parameter ).col(); ix++ )
			{
				float g = atomicExchange( &( *gradient )( ix, iy ), 0.0f );
				if( g == 0.0f )
				{
					continue;
				}
				float m = m_m( ix, iy ) = m_beta1 * m_m( ix, iy ) + ( 1.0f - m_beta1 ) * g;
				float v = m_v( ix, iy ) = m_beta2 * m_v( ix, iy ) + ( 1.0f - m_beta2 ) * g * g;
				float m_hat = m / ( 1.0f - m_beta1t );
				float v_hat = v / ( 1.0f - m_beta2t );
				( *parameter )( ix, iy ) = ( *parameter )( ix, iy ) - s * m_hat / ( std::sqrt( v_hat ) + m_e );
			}
		}
	private:
		float m_alpha;
		float m_beta1;
		float m_beta2;
		float m_beta1t;
		float m_beta2t;
		float m_e;
		Mat m_m;
		Mat m_v;
	};

	inline Optimizer* newOptimizer( OptimizerType optimizerType, float learningRate )
	{
		Optimizer* o = nullptr;
		if( optimizerType == OptimizerType::SGD )
		{
			o = new OptimizerSGD( learningRate );
		}
		else if( optimizerType == OptimizerType::Adam )
		{
			o = new OptimizerAdam( learningRate );
		}
		return o;
	}

	// inline void xavier(Mat)
	// PCG 
	namespace pcg
	{
		struct pcg_state_setseq_64 {    // Internals are *Private*.
			uint64_t state;             // RNG state.  All values are possible.
			uint64_t inc;               // Controls which RNG sequence (stream) is
										// selected. Must *always* be odd.
		};
		typedef struct pcg_state_setseq_64 pcg32_random_t;

		inline uint32_t pcg32_random_r(pcg32_random_t* rng)
		{
			uint64_t oldstate = rng->state;
			rng->state = oldstate * 6364136223846793005ULL + rng->inc;
			uint32_t xorshifted = ((oldstate >> 18u) ^ oldstate) >> 27u;
			uint32_t rot = oldstate >> 59u;
			return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
		}
		inline void pcg32_srandom_r(pcg32_random_t* rng, uint64_t initstate, uint64_t initseq)
		{
			rng->state = 0U;
			rng->inc = (initseq << 1u) | 1u;
			pcg32_random_r(rng);
			rng->state += initstate;
			pcg32_random_r(rng);
		}
	}

	class Rng
	{
	public:
		virtual ~Rng() {}
		float draw()
		{
			uint32_t x = drawUInt();
			uint32_t bits = ( x >> 9 ) | 0x3f800000;
			float value = *reinterpret_cast<float*>( &bits ) - 1.0f;
			return value;
		}
		virtual uint32_t drawUInt() = 0;
	};
	class StandardRng : public Rng
	{
	public:
		StandardRng( ) 
		{
			pcg::pcg32_srandom_r( &m_rng, 543, 0 );
		}

		virtual uint32_t drawUInt()
		{
			return pcg32_random_r( &m_rng );
		}
	private:
		pcg::pcg_state_setseq_64 m_rng;
	};

	class BoxMular
	{
	public:
		BoxMular(Rng* rng):m_hasZ1(false),m_Z1(0.0f),m_rng(rng)
		{
		}
		float draw()
		{
			if( m_hasZ1 )
			{
				m_hasZ1 = false;
				return m_Z1;
			}
			
			float x = m_rng->draw();
			float y = m_rng->draw();
			float k = std::sqrt( -2.0f * std::log( x ) );
			float theta = 2.0f * pi * y;
			float z0 = k * std::cos( theta );

			m_hasZ1 = true;
			m_Z1 = k * std::sin( theta );
			return z0;
		}
	private:
		bool m_hasZ1;
		float m_Z1;
		Rng* m_rng;
	};
	inline float drawRange( float a, float b, Rng* rng )
	{
		return a + ( b - a ) * rng->draw();
	}

	class MatContext
	{
	public:
		Mat& var( const char *name )
		{
			return m_variables[name];
		}
		const Mat& var( const char* name ) const
		{
			return m_variables.find( name )->second;
		}
	private:
		std::map<std::string, Mat> m_variables;
	};

	class Layer
	{
	public:
		Layer( int inputDimensions, int outputDimensions ) : m_inputDimensions( inputDimensions ), m_outputDimensions( outputDimensions ) {}
		virtual ~Layer() {}
		virtual void initialize( InitializationType initType, Rng* rng ) = 0;
		virtual void forward( Mat* r, const Mat& value, MatContext* context /* optional */ ) = 0;
		virtual void backward( Mat* r, const Mat& gradient, MatContext* context ) = 0;
		virtual void optimize( int nElement, pr::ThreadPool *pool ) = 0;

		int inputDimensions() const { return m_inputDimensions; }
		int outputDimensions() const { return m_outputDimensions; }
	private:
		int m_inputDimensions;
		int m_outputDimensions;
	};

#define ENABLE_BIAS 1

	class AffineLayer : public Layer
	{
	public:
		AffineLayer( int i, int o, OptimizerType optimizerType, float learningRate ) 
			: Layer( i, o ) 
			, m_W( i, o ) 
			, m_dW( i, o )
#if ENABLE_BIAS
			, m_b( 1, o ) 
			, m_db( 1, o )
#endif
		{
			m_oW = std::unique_ptr<Optimizer>( newOptimizer( optimizerType, learningRate ) );
			m_oW->initialize( m_W.row(), m_W.col() );
			m_dW.fill( 0.0f );
#if ENABLE_BIAS
			m_ob = std::unique_ptr<Optimizer>( newOptimizer( optimizerType, learningRate ) );
			m_ob->initialize( m_b.row(), m_b.col() );
			m_db.fill( 0.0f );
#endif
		}
		virtual void initialize( InitializationType initType , Rng* rng )
		{
			// xavier
			if( initType == InitializationType::Xavier )
			{
				float k = std::sqrt( 6.0f ) / std::sqrt( (float)inputDimensions() + (float)outputDimensions() );
				FOR_EACH_ELEMENT( m_W, ix, iy )
				{
					m_W( ix, iy ) = drawRange( -k, k, rng );
				}
#if ENABLE_BIAS
				FOR_EACH_ELEMENT( m_b, ix, iy )
				{
					m_b( ix, iy ) = drawRange( -k, k, rng );
				}
#endif
			} 
			else if( initType == InitializationType::He )
			{
				float s = std::sqrt( 2.0f / (float)inputDimensions() );
				BoxMular m( rng );
				FOR_EACH_ELEMENT( m_W, ix, iy )
				{
					m_W( ix, iy ) = m.draw() * s;
				}
#if ENABLE_BIAS
				FOR_EACH_ELEMENT( m_b, ix, iy )
				{
					m_b( ix, iy ) = m.draw() * s;
				}
#endif
			}
			else
			{
				RPML_ASSERT( 0 );
			}
		}
		virtual void forward( Mat* r, const Mat& value, MatContext* context /* optional */ )
		{
			if( context )
			{
				context->var( "x" ) = value; // input ( N, input )
			}
			mul( r, value, m_W );

#if ENABLE_BIAS
			addVectorForEachRow( r, m_b );
#endif
		}
		virtual void backward( Mat* r, const Mat& gradient, MatContext* context )
		{
			const Mat& x = context->var( "x" );
			Mat& dW = context->var( "dW" );
			Mat& trx = context->var( "trx" );
			transpose( &trx, x );
			mul( &dW, trx, gradient );

#if ENABLE_BIAS
			Mat& db = context->var( "db" );
			vertialSum( &db, gradient );
#endif
			{
				std::lock_guard<std::mutex> lc( m_dmutex );
				add( &m_dW, dW );
#if ENABLE_BIAS
				add( &m_db, db );
#endif
			}

			Mat& trW = context->var( "trW" );
			transpose( &trW, m_W );
			mul( r, gradient, trW );
		}
		virtual void optimize( int nElement, pr::ThreadPool* pool ) 
		{
			m_oW->optimize( &m_W, m_dW, nElement );
			m_dW.fill( 0.0f ); 
#if ENABLE_BIAS
			m_ob->optimize( &m_b, m_db, nElement );
			m_db.fill( 0.0f );
#endif
		}
		Mat m_W;  // ( input, output )
		Mat m_dW; // Å›L/Å›W = ( input, output )
		Mat m_b;  // ( 1, output )
		Mat m_db; // Å›L/Å›b = ( 1, output )
		std::unique_ptr<Optimizer> m_oW;
		std::unique_ptr<Optimizer> m_ob;

		std::mutex m_dmutex;
	};

	class LeakyReLULayer : public Layer
	{
	public:
		LeakyReLULayer( int i, int o, float slope ) : Layer( i, o ), m_slope( slope ) {}

		virtual void forward( Mat* r, const Mat& value, MatContext* context /* optional */ )
		{
			if( context )
			{
				context->var( "x" ) = value;
			}

			( *r ).setShape( value.row(), value.col() );
			FOR_EACH_ELEMENT( *r, ix, iy )
			{
				float x = value( ix, iy );
				( *r )( ix, iy ) = 0.0f < x ? x : x * m_slope;
			}
		}
		virtual void backward( Mat* r, const Mat& gradient, MatContext* context )
		{
			const Mat& x = context->var( "x" );

			( *r ).setShape( gradient.row(), gradient.col() );
			FOR_EACH_ELEMENT( gradient, ix, iy )
			{
				float d = 0.0f < x( ix, iy ) ? 1.0f : m_slope;
				( *r )( ix, iy ) = d * gradient( ix, iy );
			}
		}
		virtual void initialize( InitializationType initType, Rng* rng ) {}
		virtual void optimize( int nElement, pr::ThreadPool* pool ) {}

	private:
		float m_slope;
	};
	class ReLULayer : public LeakyReLULayer
	{
	public:
		ReLULayer( int i, int o ) : LeakyReLULayer( i, o, 0.0f ) {}
	};

	class SigmoidLayer : public Layer
	{
	public:
		SigmoidLayer( int i, int o ) : Layer( i, o ) {}
		virtual void forward( Mat* r, const Mat& value, MatContext* context /* optional */ ) 
		{
			(*r).setShape( value.row(), value.col() );
			FOR_EACH_ELEMENT( *r, ix, iy )
			{
				float x = value( ix, iy );
				( *r )( ix, iy ) = 1.0f / ( 1.0f + std::exp( -x ) );
			}
			if( context )
			{
				context->var( "y" ) = ( *r );
			}
		}
		virtual void backward( Mat* r, const Mat& gradient, MatContext* context )
		{
			const Mat& y = context->var( "y" );

			( *r ).setShape( gradient.row(), gradient.col() );
			FOR_EACH_ELEMENT( *r, ix, iy )
			{
				float d = y( ix, iy ) * ( 1.0f - y( ix, iy ) );
				( *r )( ix, iy ) = d * gradient( ix, iy );
			}
		}
		virtual void initialize( InitializationType initType, Rng* rng ) {}
		virtual void optimize( int nElement, pr::ThreadPool* pool ) {}
	};
	class TanhLayer : public Layer
	{
	public:
		TanhLayer( int i, int o ) : Layer( i, o ) {}
		virtual void forward( Mat* r, const Mat& value, MatContext* context /* optional */ )
		{
			( *r ).setShape( value.row(), value.col() );
			FOR_EACH_ELEMENT( *r, ix, iy )
			{
				float x = value( ix, iy );
				( *r )( ix, iy ) = std::tanh( x );
			}
			if( context )
			{
				context->var( "y" ) = ( *r );
			}
		}
		virtual void backward( Mat* r, const Mat& gradient, MatContext* context )
		{
			const Mat& y = context->var( "y" );

			( *r ).setShape( gradient.row(), gradient.col() );
			FOR_EACH_ELEMENT( *r, ix, iy )
			{
				float d = 1.0f - y( ix, iy ) * y( ix, iy );
				( *r )( ix, iy ) = d * gradient( ix, iy );
			}
		}
		virtual void initialize( InitializationType initType, Rng* rng ) {}
		virtual void optimize( int nElement, pr::ThreadPool* pool ) {}
	};

	inline float mse( const Mat& x, const Mat& ref )
	{
		RPML_ASSERT( x.row() == ref.row() );
		RPML_ASSERT( x.col() == ref.col() );

		float e = 0.0f;
		FOR_EACH_ELEMENT( x, ix, iy )
		{
			float d = ref( ix, iy ) - x( ix, iy );
			e += d * d;
		}
		return e;
	}

	class FrequencyEncoder : public Layer
	{
	public:
		struct Config
		{
			int N = 12;
		};
		static int output( int input, const Config& config )
		{
			return input * config.N * 2;
		}
		FrequencyEncoder( int i, int o, const Config& config ) : Layer( i, o ), m_config( config ) {}

		virtual void forward( Mat* r, const Mat& value, MatContext* context )
		{
			( *r ).setShape( value.row(), outputDimensions() );
			FOR_EACH_ELEMENT( value, ix, iy )
			{
				float x = value( ix, iy );

				for( int i = 0; i < m_config.N ; i++ )
				{
					float k = 2.0f * pi * std::pow( 2.0f, i );
					float a = std::sin( k * x );
					float b = std::cos( k * x );
					( *r )( ix * m_config.N * 2 + i * 2 + 0, iy ) = a;
					( *r )( ix * m_config.N * 2 + i * 2 + 1, iy ) = b;
				}
			}
		}
		virtual void backward( Mat* r, const Mat& gradient, MatContext* context ) { }
		virtual void initialize( InitializationType initType, Rng* rng ) {}
		virtual void optimize( int nElement, pr::ThreadPool* pool ) {}

		Config m_config;
	};

	class HashGridEvaluator
	{
	public:
		HashGridEvaluator( int dim ) : m_dim( dim ), m_bits( 0xFFFFFFFF )
		{
		}
		bool moveNext()
		{
			m_bits++;
			return m_bits < ( 1 << m_dim );
		}
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

				RPML_ASSERT( 0.0f <= u && u <= 1.0f );

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
	class HashGridVisitor
	{
	public:
		HashGridVisitor( int dim ) : m_dim( dim ), m_bits( 0xFFFFFFFF )
		{
		}
		bool moveNext()
		{
			m_bits++;
			return m_bits < ( 1 << m_dim );
		}
		uint32_t evaluate( int resolution, float* input )
		{
			DimensionHasher hasher;
			for( int d = 0; d < m_dim; ++d )
			{
				float x_in = input[d];
				float xf = x_in * resolution;
				uint32_t xi = xf;
				if( m_bits & ( 1 << d ) )
				{
					hasher.add( xi + 1, d );
				}
				else
				{
					hasher.add( xi, d );
				}
			}
			return hasher.value();
		}
	private:
		int m_dim;
		uint32_t m_bits;
	};

	class MultiResolutionHashEncoder : public Layer
	{
	public:
		struct Config
		{
			int L = 16;
			int T = std::pow( 2, 19 );
			int F = 2;
			int Nmin = 16;
			float b = 1.38191f;
		};

		static int output( int input, const Config& config )
		{
			return config.L * config.F;
		}
		MultiResolutionHashEncoder( int i, int o, const Config& config, float learningRate ) : Layer( i, o ), m_config( config )
		{
			m_features.resize( m_config.L );
			m_dfeatures.resize( m_config.L );
			m_optimizers.resize( m_config.L );
			for( int i = 0; i < m_config.L ; i++ )
			{
				m_features[i].setShape( m_config.T, m_config.F );
				m_dfeatures[i].setShape( m_config.T, m_config.F );
				m_dfeatures[i].fill( 0.0f );
				m_optimizers[i] = std::unique_ptr<OptimizerAdam>( new OptimizerAdam( learningRate ) );
				m_optimizers[i]->initialize( m_config.T, m_config.F );
			}
		}
		virtual void initialize( InitializationType initType, Rng* rng ) 
		{
			for( int i = 0; i < m_features.size() ; i++ )
			{
				FOR_EACH_ELEMENT( m_features[i], ix, iy )
				{
					m_features[i]( ix, iy ) = -1e-4f + 2.0f * 1e-4f * rng->draw();
				}
			}
		}

		// assume input values are 0 to 1
		virtual void forward( Mat* r, const Mat& value, MatContext* context )
		{
			RPML_ASSERT( value.col() == inputDimensions() );

			( *r ).setShape( value.row(), outputDimensions() );

			if( context )
			{
				context->var( "value" ) = value;
			}

			const int dim = inputDimensions();
			std::vector<float> featureVector( m_config.F );
			std::vector<float> hashInput( dim );

			for( int l = 0 ; l < m_config.L ; l++ )
			{
				float res = floor( m_config.Nmin * std::pow( m_config.b, l ) );
				
				for( int row = 0; row < value.row() ; row++ )
				{
#if !defined( RPML_DISABLE_ASSERT )
					float sw = 0.0f;
#endif
					std::fill( featureVector.begin(), featureVector.end(), 0.0f );
					for( int d = 0; d < dim; d++ )
					{
						hashInput[d] = value( d, row );
					}
					HashGridEvaluator evaluator( dim );
					while( evaluator.moveNext() )
					{
						float w;
						uint32_t h;
						evaluator.evaluate( &w, &h, res, hashInput.data() );

						uint32_t index = h % m_config.T;
						for( int fdim = 0; fdim < m_config.F; fdim++ )
						{
							featureVector[fdim] += w * m_features[l]( fdim, index );
						}

#if !defined( RPML_DISABLE_ASSERT )
						sw += w;
#endif
					}
					RPML_ASSERT( fabs( sw - 1.0f ) < 0.001f );

					// store feature vector
					for( int fdim = 0; fdim < m_config.F; fdim++ )
					{
						( *r )( m_config.F * l + fdim, row ) = featureVector[fdim];
					}
				}
			}
		}

		virtual void backward( Mat* r, const Mat& gradient, MatContext* context ) 
		{
			const Mat& value = context->var( "value" );

			const int dim = inputDimensions();
			const float b = m_config.b;

			std::vector<float> hashInput( dim );
			std::vector<float> dfeatureVector( dim );

			for( int l = 0 ; l < m_config.L ; l++ )
			{
				float res = floor( m_config.Nmin * std::pow( m_config.b, l ) );

				for( int row = 0; row < value.row(); row++ )
				{
					for( int d = 0; d < dim; d++ )
					{
						hashInput[d] = value( d, row );
					}
					Mat& dfeature = m_dfeatures[l];
					HashGridEvaluator evaluator( dim );
					while( evaluator.moveNext() )
					{
						float w;
						uint32_t h;
						evaluator.evaluate( &w, &h, res, hashInput.data() );

						uint32_t index = h % m_config.T;

						//bool touch = false;
						for( int fdim = 0; fdim < m_config.F; fdim++ )
						{
							float g = gradient( m_config.F * l + fdim, row ) * w;
							if( g != 0.0f )
							{
								atomAdd( &dfeature( fdim, index ), g );
							}
						}
					}
				}
			}

			std::lock_guard<std::mutex> lc( m_inputMutex );
			m_inputs.reserve( m_inputs.size() + value.row() * value.col() );
			for( int iy = 0; iy < value.row(); iy++ )
			{
				for( int ix = 0; ix < value.col(); ix++ )
				{
					m_inputs.push_back( value( ix, iy ) );
				}
			}
		}
		
		virtual void optimize( int nElement, pr::ThreadPool* pool ) 
		{
			for( int l = 0; l < m_config.L; l++ )
			{
				m_optimizers[l]->incrementParametereHashGridRow();
			}
			
			int dim = m_inputs.size() / nElement;
			pr::TaskGroup g;
			g.addElements( nElement );
			pool->enqueueFor( nElement, 2, [&g, dim, nElement, this]( uint64_t beg, uint64_t end ) 
			{
#if ENABLE_TRACE
				pr::ChromeTraceTimer tr( pr::ChromeTraceTimer::AddMode::Auto );
				tr.label( "optimize task" );
#endif
				std::vector<float> hashInput( dim );
				for( int l = 0; l < m_config.L; l++ )
				{
					float res = floor( m_config.Nmin * std::pow( m_config.b, l ) );

					for( int iy = beg; iy < end; iy++ )
					{
						for( int d = 0; d < dim; d++ )
						{
							hashInput[d] = m_inputs[ iy * dim + d ];
						}
						HashGridVisitor visitor( dim );
						while( visitor.moveNext() )
						{
							uint32_t h = visitor.evaluate( res, hashInput.data() );
							uint32_t index = h % m_config.T;
							m_optimizers[l]->atomicOptimizeHashGridRow( &m_features[l], &m_dfeatures[l], nElement, index );
						}
					}
				}
				g.doneElements( end - beg );
			} );

			while( !g.isFinished() )
			{
				pool->processTask();
			}
			m_inputs.clear();
		}

		Config m_config;

		std::vector<Mat> m_features;
		std::vector<Mat> m_dfeatures;
		std::vector<std::unique_ptr<OptimizerAdam>> m_optimizers;

		std::mutex m_inputMutex;
		std::vector<float> m_inputs; // row major
	};

	class MLPConfig
	{
	public:
		float m_learningRate = 0.01f;
		std::vector<int> m_shape;
		ActivationType m_activationType = ActivationType::ReLU;
		InitializationType m_initType = InitializationType::He;
		OptimizerType m_optimType = OptimizerType::Adam;
		EncoderType m_encoderType = EncoderType::None;
		FrequencyEncoder::Config m_frequencyEncoderConfig;

#define PROP( name ) \
		MLPConfig& name( const decltype( m_##name )& name ) \
		{ \
			m_##name = name; \
			return *this; \
		}

		PROP( learningRate );
		PROP( shape );
		PROP( activationType );
		PROP( optimType );
		PROP( initType );
		PROP( encoderType );
		PROP( frequencyEncoderConfig );
#undef PROP
	};

	struct LocalStorage
	{
		MatContext context;
		std::vector<MatContext> layerContexts;
	};

	class MLP
	{
	public:
		MLP( const MLPConfig& config ) : m_pool( std::thread::hardware_concurrency() )
		{
			std::unique_ptr<Rng> rng = std::unique_ptr<Rng>( new StandardRng() );

			for( int i = 0; i < config.m_shape.size() - 1; i++ )
			{
				int input = config.m_shape[i];
				int output = config.m_shape[i + 1];

				if( i == 0 )
				{
					if( config.m_encoderType == EncoderType::Frequency )
					{
						int encoderOutput = FrequencyEncoder::output( input, config.m_frequencyEncoderConfig );
						std::unique_ptr<Layer> encoder = std::unique_ptr<Layer>( new FrequencyEncoder( input, encoderOutput, config.m_frequencyEncoderConfig ) );
						encoder->initialize( config.m_initType, rng.get() );
						m_layers.emplace_back( std::move( encoder ) );

						input = encoderOutput;
					}
					else if( config.m_encoderType == EncoderType::MultiResolutionHash )
					{
						int encoderOutput = MultiResolutionHashEncoder::output( input, MultiResolutionHashEncoder::Config() );
						std::unique_ptr<Layer> encoder = std::unique_ptr<Layer>( new MultiResolutionHashEncoder( input, encoderOutput, MultiResolutionHashEncoder::Config(), config.m_learningRate ) );
						encoder->initialize( config.m_initType, rng.get() );
						m_layers.emplace_back( std::move( encoder ) );

						input = encoderOutput;
					}
				}

				std::unique_ptr<Layer> layer( new AffineLayer( input, output, config.m_optimType, config.m_learningRate ) );
				layer->initialize( config.m_initType, rng.get() );
				m_layers.emplace_back( std::move( layer ) );

				bool isLast = i + 1 == config.m_shape.size() - 1;
				if( !isLast )
				{
					
					std::unique_ptr<Layer> activation;

					switch( config.m_activationType )
					{
					case ActivationType::ReLU:
						activation = std::unique_ptr<Layer>( new ReLULayer( output, output ) );
						break;
					case ActivationType::LeakyReLU:
						activation = std::unique_ptr<Layer>( new LeakyReLULayer( output, output, 0.05f ) );
						break;
					case ActivationType::Tanh:
						activation = std::unique_ptr<Layer>( new TanhLayer( output, output ) );
						break;
					case ActivationType::Sigmoid:
						activation = std::unique_ptr<Layer>( new SigmoidLayer( output, output ) );
						break;
					default:
						RPML_ASSERT( 0 );
					}

					activation->initialize( config.m_initType, rng.get() );
					m_layers.emplace_back( std::move( activation ) );
				}
			}
		}
		float train( const Mat& x, const Mat& y, int taskGranularity = 2 )
		{
			RPML_ASSERT( x.row() == y.row() );
#if ENABLE_TRACE
			pr::ChromeTraceTimer tr( pr::ChromeTraceTimer::AddMode::Auto );
			tr.label( "train" );
#endif

			int nElement = x.row();

			float loss = 0.0f;

			std::vector<std::atomic<int>> layerTasks( m_layers.size() );
			std::fill( layerTasks.begin(), layerTasks.end(), 0 );

			pr::TaskGroup g;
			g.addElements( nElement );
			m_pool.enqueueFor( nElement, taskGranularity, [&]( int64_t beg, int64_t end ) 
			{
#if ENABLE_TRACE
				pr::ChromeTraceTimer tr( pr::ChromeTraceTimer::AddMode::Auto );
				tr.label( "train task" );
#endif
				// unsigned int fp_control_state = _controlfp( _EM_INEXACT, _MCW_EM );

				std::shared_ptr<LocalStorage> localStorage = acquireLocalStorage();
				localStorage->layerContexts.resize( m_layers.size() );

				Mat& inputMat = localStorage->context.var( "inputMat" );
				Mat& outputMat = localStorage->context.var( "outputMat" );
				Mat& slicedY = localStorage->context.var( "slicedY" );

				sliceH( &inputMat, x, beg, end );
				sliceH( &slicedY, y, beg, end );
				
				for( int i = 0; i < m_layers.size(); i++ )
				{
					RPML_ASSERT( inputMat.col() == m_layers[i]->inputDimensions() );
					m_layers[i]->forward( &outputMat, inputMat, &localStorage->layerContexts[i] );
					RPML_ASSERT( outputMat.col() == m_layers[i]->outputDimensions() );
					inputMat.swap( outputMat );
				}

				// m: estimated result
				float L = mse( inputMat, slicedY );
				atomAdd( &loss, L );

				// MSE backward is "x - y"
				sub( &inputMat, slicedY );

				for( int i = (int)m_layers.size() - 1; 0 <= i; i-- )
				{
					RPML_ASSERT( inputMat.col() == m_layers[i]->outputDimensions() );
					m_layers[i]->backward( &outputMat, inputMat, &localStorage->layerContexts[i] );

					int nTasks = end - beg;
					if( layerTasks[i].fetch_add( nTasks ) + nTasks == nElement )
					{
						m_layers[i]->optimize( nElement, &m_pool );
					}

					if( i != 0 )
					{
						RPML_ASSERT( outputMat.col() == m_layers[i]->inputDimensions() );
					}
					inputMat.swap( outputMat );
				}
				g.doneElements( end - beg );
			} );
			while( !g.isFinished() )
			{
				m_pool.processTask();
			}

			return loss;
		}
		std::shared_ptr<LocalStorage> acquireLocalStorage()
		{
			std::shared_ptr<LocalStorage> localStorage;
			{
				std::lock_guard<std::mutex> lc( m_localMutex );
				auto tid = std::this_thread::get_id();
				if( m_localStorages.count( tid ) == 0 )
				{
					m_localStorages[tid] = std::shared_ptr<LocalStorage>( new LocalStorage() );
				}
				localStorage = m_localStorages[tid];
			}
			return localStorage;
		}

		void forward( Mat *r, const Mat& x )
		{
			std::shared_ptr<LocalStorage> localStorage = acquireLocalStorage();

			Mat& inputMat = *r;
			Mat& outputMat = localStorage->context.var( "outputMat" );
			inputMat = x;

			for( int i = 0; i < m_layers.size(); i++ )
			{
				RPML_ASSERT( inputMat.col() == m_layers[i]->inputDimensions() );
				m_layers[i]->forward( &outputMat, inputMat, nullptr );
				RPML_ASSERT( outputMat.col() == m_layers[i]->outputDimensions() );
				inputMat.swap( outputMat );
			}
		}
		void forwardMT( Mat* r, const Mat& x, int taskGranularity = 8 /* hyper parameter */ )
		{
#if ENABLE_TRACE
			pr::ChromeTraceTimer tr( pr::ChromeTraceTimer::AddMode::Auto );
			tr.label( "forwardMT" );
#endif

			int outputDim = m_layers[m_layers.size() - 1]->outputDimensions();
			( *r ).setShape( x.row(), outputDim );

			int nElement = x.row();
			std::mutex lmutex;

			pr::TaskGroup g;
			g.addElements( nElement );
			m_pool.enqueueFor( nElement, taskGranularity, [&]( int64_t beg, int64_t end )
			{
#if ENABLE_TRACE
				pr::ChromeTraceTimer tr( pr::ChromeTraceTimer::AddMode::Auto );
				tr.label( "forward task" );
#endif
				std::shared_ptr<LocalStorage> localStorage = acquireLocalStorage();
				localStorage->layerContexts.resize( m_layers.size() );

				Mat& inputMat = localStorage->context.var( "inputMat" );
				Mat& outputMat = localStorage->context.var( "outputMat" );

				sliceH( &inputMat, x, beg, end );
				
				for( int i = 0; i < m_layers.size(); i++ )
				{
					RPML_ASSERT( inputMat.col() == m_layers[i]->inputDimensions() );
					m_layers[i]->forward( &outputMat, inputMat, nullptr );
					RPML_ASSERT( outputMat.col() == m_layers[i]->outputDimensions() );
					inputMat.swap( outputMat );
				}

				concatH( r, inputMat, beg, end );

				g.doneElements( end - beg ); 
			} );
			while( !g.isFinished() )
			{
				m_pool.processTask();
			}
		}

		std::vector<std::unique_ptr<Layer>> m_layers;
		pr::ThreadPool m_pool;
		std::map<std::thread::id, std::shared_ptr<LocalStorage>> m_localStorages;
		std::mutex m_localMutex;
	};

	struct NeRFInput
	{
		float ro[3];
		float rd[3];
	};
	struct NeRFOutput
	{
		float color[3];
	};
	struct NeRFMarching
	{
		int beg;
		int end;
	};
	static const float Teps = 0.0001f;

	static const int OC_BASE_SIZE = 128;
	static const float OC_MIN_A = 0.008f;
	static const float OC_MIN_DENSITY = -OC_BASE_SIZE / std::sqrt( 3.0f ) * std::log( 1.0f - OC_MIN_A );

	class OccupancyGrid
	{
	public:
		OccupancyGrid() : m_grid( OC_BASE_SIZE * OC_BASE_SIZE * OC_BASE_SIZE )
		{
		}
		void decay( float s = 0.95f )
		{
			for( int i = 0; i < m_grid .size(); ++i)
			{
				m_grid[i] *= s;
			}
		}
		void update( float density, int x, int y, int z )
		{
			int index = x + OC_BASE_SIZE * y + OC_BASE_SIZE * OC_BASE_SIZE * z;
			m_grid[index] = maxss( m_grid[index], density );
		}
		bool occupied( float x, float y, float z )
		{
			int xi = clampss( x, 0.0f, 0.999999f ) * OC_BASE_SIZE;
			int yi = clampss( y, 0.0f, 0.999999f ) * OC_BASE_SIZE;
			int zi = clampss( z, 0.0f, 0.999999f ) * OC_BASE_SIZE;
			int index = xi + OC_BASE_SIZE * yi + OC_BASE_SIZE * OC_BASE_SIZE * zi;

			return OC_MIN_DENSITY < m_grid[index];
		}
		std::vector<float> m_grid;
	};

	class NeRF
	{
	public:
		enum
		{
			MLP_DENSITY_OUT = 16,
			MLP_WIDTH = 64,
			//MLP_STEP = 64,
			MLP_STEP = 1024,
		};
		NeRF( ) : m_pool( std::thread::hardware_concurrency() )
		{
			std::unique_ptr<Rng> rng = std::unique_ptr<Rng>( new StandardRng() );
			
			float learningRate = 256;
			InitializationType initializerType = InitializationType::He;
			int input = 3; /* xyz */
			int output = 0;

			auto createActivation = []( int output )
			{
				return std::unique_ptr<Layer>( new ReLULayer( output, output ) );
				// return std::unique_ptr<Layer>( new LeakyReLULayer( output, output, 0.05f ) );
			};

			// density network

			{   // encoding
				int encoderOutput = MultiResolutionHashEncoder::output( 3, MultiResolutionHashEncoder::Config() );
				std::unique_ptr<Layer> encoder = std::unique_ptr<Layer>( new MultiResolutionHashEncoder( input, encoderOutput, MultiResolutionHashEncoder::Config(), learningRate ) );
				encoder->initialize( initializerType, rng.get() );
				m_densityLayers.emplace_back( std::move( encoder ) );
				input = encoderOutput;
			}

			{   // hidden
				output = MLP_WIDTH;

				std::unique_ptr<Layer> layer( new AffineLayer( input, output, OptimizerType::Adam, learningRate ) );
				layer->initialize( initializerType, rng.get() );
				m_densityLayers.emplace_back( std::move( layer ) );

				std::unique_ptr<Layer> activation = createActivation( output );
				activation->initialize( initializerType, rng.get() );
				m_densityLayers.emplace_back( std::move( activation ) );
				input = output;
			}
			{ // out
				output = MLP_DENSITY_OUT;

				std::unique_ptr<Layer> layer( new AffineLayer( input, output, OptimizerType::Adam, learningRate ) );
				layer->initialize( initializerType, rng.get() );
				m_densityLayers.emplace_back( std::move( layer ) );

				//std::unique_ptr<Layer> activation = createActivation( output );
				//activation->initialize( initializerType, rng.get() );
				//m_densityLayers.emplace_back( std::move( activation ) );
				input = output;
			}

			// color network
			input += 16; // dir SH encoding

			std::vector<int> colorShape = { input, MLP_WIDTH, MLP_WIDTH, 3 };
			// std::vector<int> colorShape = { input, MLP_WIDTH, 3 };
			for( int i = 0; i < colorShape.size() - 1; i++ )
			{
				int input = colorShape[i];
				int output = colorShape[i + 1];

				std::unique_ptr<Layer> layer( new AffineLayer( input, output, OptimizerType::Adam, learningRate ) );
				layer->initialize( initializerType, rng.get() );
				m_colorLayers.emplace_back( std::move( layer ) );

				bool isLast = i + 1 == colorShape.size() - 1;
				if( !isLast )
				{
					std::unique_ptr<Layer> activation = createActivation( output );
					activation->initialize( initializerType, rng.get() );
					m_colorLayers.emplace_back( std::move( activation ) );
				}
			}
		}

		inline float densityActivation(float x)
		{
			return std::exp( x );
		}
		inline float densityActivationDrivative( float x )
		{
			return std::exp( clampss( x, -15.0f, 15.0f) );
		}
		inline float rgbActivation( float x )
		{
			return 1.0f / ( 1.0f + std::exp( -x ) );
		}
		inline float rgbActivationDerivative( float x ) 
		{
			float y = rgbActivation( x );
			return y * ( 1 - y );
		}
		void updateOccupancyGrid()
		{
			{
#if ENABLE_TRACE
				pr::ChromeTraceTimer tr( pr::ChromeTraceTimer::AddMode::Auto );
				tr.label( "m_occupancyGrid.decay()" );
#endif
				m_occupancyGrid.decay();
			}
			pr::TaskGroup g;
			g.addElements( OC_BASE_SIZE );
			m_pool.enqueueFor( OC_BASE_SIZE, 8, [&]( int64_t beg, int64_t end ) {
#if ENABLE_TRACE
				pr::ChromeTraceTimer tr( pr::ChromeTraceTimer::AddMode::Auto );
				tr.label( "update cells" );
#endif
				static thread_local StandardRng rng;

				Mat inputMat( ( end - beg ) * OC_BASE_SIZE * OC_BASE_SIZE, 3 );
				Mat outputMat;

				int i = 0;
				for( int64_t zi = beg ; zi < end ; zi++ )
				for( int yi = 0; yi < OC_BASE_SIZE; yi++ )
				for( int xi = 0; xi < OC_BASE_SIZE; xi++ )
				{
					float x = ( xi + rng.draw() ) / OC_BASE_SIZE;
					float y = ( yi + rng.draw() ) / OC_BASE_SIZE;
					float z = ( zi + rng.draw() ) / OC_BASE_SIZE;
					inputMat( 0, i ) = x;
					inputMat( 1, i ) = y;
					inputMat( 2, i ) = z;

					i++;
				}

				for( int i = 0; i < m_densityLayers.size(); i++ )
				{
					RPML_ASSERT( inputMat.col() == m_densityLayers[i]->inputDimensions() );
					m_densityLayers[i]->forward( &outputMat, inputMat, nullptr );
					RPML_ASSERT( outputMat.col() == m_densityLayers[i]->outputDimensions() );
					inputMat.swap( outputMat );
				}

				i = 0;
				for( int64_t zi = beg; zi < end; zi++ )
				for( int yi = 0; yi < OC_BASE_SIZE; yi++ )
				for( int xi = 0; xi < OC_BASE_SIZE; xi++ )
				{
					float density = densityActivation( inputMat( 0, i ) );
					m_occupancyGrid.update( density, xi, yi, zi );
					i++;
				}
				g.doneElements( end - beg );
			} );
			while( !g.isFinished() )
			{
				m_pool.processTask();
			}
		}
		float train( const NeRFInput* inputs, const NeRFOutput* outputs, int nElement )
		{
			// unsigned int fp_control_state = _controlfp( _EM_INEXACT, _MCW_EM );

			static int oI = 0;
			if( ++oI % 16 == 0 )
			{
				updateOccupancyGrid();
				m_hasOc = true;
			}

			float loss = 0.0f;

			std::vector<std::atomic<int>> densityLayerTasks( m_densityLayers.size() );
			std::vector<std::atomic<int>> colorLayerTasks( m_colorLayers.size() );
			std::fill( densityLayerTasks.begin(), densityLayerTasks.end(), 0 );
			std::fill( colorLayerTasks.begin(), colorLayerTasks.end(), 0 );

			std::vector<NeRFMarching>& marchings = m_marchings;
			std::vector<float>& points = m_points;
			marchings.clear();
			points.clear();

			int nSkipEval = 0;

			const float dt = std::sqrt( 3.0f ) / MLP_STEP;
			for( int64_t i = 0; i < nElement; i++ )
			{
				NeRFInput input = inputs[i];

				NeRFMarching m;
				m.beg = points.size() / 3;

				int nSteps = 0;
				for( ; ; )
				{
					static StandardRng rng;
					float t = dt * ( nSteps + rng.draw() );
					nSteps++;
					float x = input.ro[0] + input.rd[0] * t;
					float y = input.ro[1] + input.rd[1] * t;
					float z = input.ro[2] + input.rd[2] * t;

					
					// todo: adjust range
					const float eps = 0.000001f;
					if( x < -eps || 1.0f + eps < x || y < -eps || 1.0f + eps < y || z < -eps || 1.0f + eps < z )
					{
						break;
					}

					x = clampss( x, 0.0f, 1.0f );
					y = clampss( y, 0.0f, 1.0f );
					z = clampss( z, 0.0f, 1.0f );

					// skip
					if( m_hasOc && m_occupancyGrid.occupied( x, y, z ) == false )
					{
						nSkipEval++;
						continue;
					}

					points.push_back( x );
					points.push_back( y );
					points.push_back( z );
				}

				m.end = points.size() / 3;
				marchings.push_back( m );
			}

			printf( "nSkipEval %d\n", nSkipEval );

			pr::TaskGroup g;
			g.addElements( nElement );
			m_pool.enqueueFor( nElement, 8, [&]( int64_t beg, int64_t end ) {
				int mBegin = marchings[beg].beg;
				int mEnd = marchings[end-1].end;
				int nTasks = end - beg;
				int nLocalEvaluation = mEnd - mBegin;
				int nEvaluation = points.size() / 3;

				if( nEvaluation == 0 )
				{
					g.doneElements( end - beg );
					return;
				}

				Mat inputMat( nLocalEvaluation, 3 );
				Mat outputMat;

				for( int i = mBegin; i < mEnd; i++ )
				{
					int localI = i - mBegin;
					inputMat( 0, localI ) = points[i * 3];
					inputMat( 1, localI ) = points[i * 3 + 1];
					inputMat( 2, localI ) = points[i * 3 + 2];
				}

				LocalStorage densityStorage;
				LocalStorage colorStorage;
				densityStorage.layerContexts.resize( m_densityLayers.size() );
				colorStorage.layerContexts.resize( m_colorLayers.size() );

				for( int i = 0; i < m_densityLayers.size(); i++ )
				{
					RPML_ASSERT( inputMat.col() == m_densityLayers[i]->inputDimensions() );
					m_densityLayers[i]->forward( &outputMat, inputMat, &densityStorage.layerContexts[i] );
					RPML_ASSERT( outputMat.col() == m_densityLayers[i]->outputDimensions() );
					inputMat.swap( outputMat );
				}

				Mat densityMat = inputMat;

				// combine dir
				Mat combinedMat( densityMat.row(), densityMat.col() + 16 );
				for( int i = beg; i < end; i++ )
				{
					NeRFInput input = inputs[i];
					float sh_encode[16] = {};
					sh_L4( sh_encode, input.rd[0], input.rd[1], input.rd[2] );

					for( int j = marchings[i].beg; j < marchings[i].end ; j++ )
					{
						int localJ = j - mBegin;

						// original 
						for( int k = 0; k < MLP_DENSITY_OUT; ++k )
						{
							combinedMat( k, localJ ) = densityMat( k, localJ );
						}
						// dir
						for( int k = 0 ; k < 16 ; ++k )
						{
							combinedMat( MLP_DENSITY_OUT + k, localJ ) = sh_encode[k];
						}
					}
				}
				inputMat.swap( combinedMat );

				for( int i = 0; i < m_colorLayers.size(); i++ )
				{
					RPML_ASSERT( inputMat.col() == m_colorLayers[i]->inputDimensions() );
					m_colorLayers[i]->forward( &outputMat, inputMat, &colorStorage.layerContexts[i] );
					RPML_ASSERT( outputMat.col() == m_colorLayers[i]->outputDimensions() );
					inputMat.swap( outputMat );
				}
			
				Mat dL_dC( inputMat.row(), inputMat.col() );
				Mat dL_dSigma( inputMat.row(), 1 );
				for( int i = beg; i < end; i++ )
				{
					float oColor[3] = {};
					float T = 1.0f;
				
					// printf( "[%d] pts %d\n", i, marchings[i].end - marchings[i].beg );
					for( int j = marchings[i].beg; j < marchings[i].end; j++ )
					{
						int localJ = j - mBegin;
						float sigma = densityActivation( densityMat( 0, localJ ) );

						//if( m_hasOc)
						//printf( "%f\n", sigma );

						float c[3];
						for(int k = 0 ; k < 3 ; k++ )
						{
							c[k] = rgbActivation( inputMat( k, localJ ) );
						}
						float a = 1.0f - std::exp( -sigma * dt );

						for( int k = 0; k < 3; k++ )
						{
							oColor[k] += T * a * c[k];
						}
						// printf( "%d %.5f %.5f %.5f\n", j - marchings[i].beg, oColor[0], oColor[1], oColor[2] );

						T *= ( 1.0f - a );

						if( T < Teps )
							break;
					}
					// printf( " %.5f %.5f %.5f\n", oColor[0], oColor[1], oColor[2] );

					float dColor[3];
					for( int k = 0; k < 3; k++ )
					{
						dColor[k] = oColor[k] - outputs[i].color[k];
						// dColor[k] = outputs[i].color[k] - oColor[k];

						atomAdd( &loss, dColor[k] * dColor[k] );
						// printf( " %.5f %.5f\n", oColor[k], outputs[i].color[k] );
					}

					T = 1.0f; // important!!!! 
				
					float oColor2[3] = {};
					for( int j = marchings[i].beg; j < marchings[i].end; j++ )
					{
						int localJ = j - mBegin;
						float sigma = densityActivation( densityMat( 0, localJ ) );
						float c[3];
						for( int k = 0; k < 3; k++ )
						{
							c[k] = rgbActivation( inputMat( k, localJ ) );
						}
						float a = 1.0f - std::exp( -sigma * dt );
						for( int k = 0; k < 3; k++ )
						{
							float coef = T * a;
							oColor2[k] += coef * c[k];
							dL_dC( k, localJ ) = rgbActivationDerivative( inputMat( k, localJ ) ) * coef * dColor[k];
							// printf( "d = %f, a = %f, {%f %f %f} \n", sigma, a, coef * dColor[0], coef * dColor[1], coef * dColor[2] );
						}
						T *= ( 1.0f - a );

						float S[3]; 
						for( int k = 0; k < 3; k++ )
						{
							S[k] = oColor[k] - oColor2[k];
						}

						float dSigma = 0.0f;
						for( int k = 0; k < 3; k++ )
						{
							dSigma += ( T * dt * c[k] - dt * S[k] ) * dColor[k];
						}
						dL_dSigma( 0, localJ ) = densityActivationDrivative( densityMat( 0, localJ ) ) * dSigma;

						if( T < Teps )
							break;

						//printf( "dSigma  %f, T %f, c %f %f %f, s %f %f %f \n", dL_dSigma( 0, j ), T, c[0], c[1], c[2], S[0], S[1], S[2] );
					}
				}

				// backward
				inputMat = dL_dC;
				for( int i = (int)m_colorLayers.size() - 1; 0 <= i; i-- )
				{
					RPML_ASSERT( inputMat.col() == m_colorLayers[i]->outputDimensions() );
					m_colorLayers[i]->backward( &outputMat, inputMat, &colorStorage.layerContexts[i] );

					if( colorLayerTasks[i].fetch_add( nTasks ) + nTasks == nElement )
					{
						m_colorLayers[i]->optimize( nEvaluation, &m_pool );
					}

					if( i != 0 )
					{
						RPML_ASSERT( outputMat.col() == m_colorLayers[i]->inputDimensions() );
					}
					inputMat.swap( outputMat );
				}

				Mat densityBackwardInput( densityMat.row(), densityMat.col() );
				for( int iy = 0; iy < densityMat.row(); iy++ )
				{
					for( int ix = 0; ix < densityMat.col(); ix++ )
					{
						densityBackwardInput( ix, iy ) = inputMat( ix, iy );
					}
					densityBackwardInput( 0, iy ) += dL_dSigma( 0, iy );
				}

				inputMat = densityBackwardInput;

				for( int i = (int)m_densityLayers.size() - 1; 0 <= i; i-- )
				{
					RPML_ASSERT( inputMat.col() == m_densityLayers[i]->outputDimensions() );
					m_densityLayers[i]->backward( &outputMat, inputMat, &densityStorage.layerContexts[i] );

					if( densityLayerTasks[i].fetch_add( nTasks ) + nTasks == nElement )
					{
						m_densityLayers[i]->optimize( nEvaluation, &m_pool );
					}

					if( i != 0 )
					{
						RPML_ASSERT( outputMat.col() == m_densityLayers[i]->inputDimensions() );
					}
					inputMat.swap( outputMat );
				}

				g.doneElements( end - beg );
			} );
			while( !g.isFinished() )
			{
				m_pool.processTask();
			}
			return loss;
		}

		void forward( const NeRFInput* inputs, NeRFOutput* outputs, int nElement )
		{
			std::vector<NeRFMarching>& marchings = m_marchings;
			std::vector<float>& points = m_points;
			marchings.clear();
			points.clear();

			const float dt = std::sqrt( 3.0f ) / MLP_STEP;
			for( int64_t i = 0; i < nElement; i++ )
			{
				NeRFInput input = inputs[i];

				NeRFMarching m;
				m.beg = points.size() / 3;

				int nSteps = 0;
				for( ;; )
				{
					static StandardRng rng;
					float t = dt * ( nSteps + rng.draw() );
					nSteps++;
					float x = input.ro[0] + input.rd[0] * t;
					float y = input.ro[1] + input.rd[1] * t;
					float z = input.ro[2] + input.rd[2] * t;

					// todo: adjust range
					const float eps = 0.000001f;
					if( x < -eps || 1.0f + eps < x || y < -eps || 1.0f + eps < y || z < -eps || 1.0f + eps < z )
					{
						break;
					}

					// skip
					if( m_hasOc && m_occupancyGrid.occupied( x, y, z ) == false )
					{
						continue;
					}

					x = clampss( x, 0.0f, 1.0f );
					y = clampss( y, 0.0f, 1.0f );
					z = clampss( z, 0.0f, 1.0f );

					points.push_back( x );
					points.push_back( y );
					points.push_back( z );
				}

				m.end = points.size() / 3;

				marchings.push_back( m );
			}

			pr::TaskGroup g;
			g.addElements( nElement );
			m_pool.enqueueFor( nElement, 8, [&]( int64_t beg, int64_t end ) {
				int mBegin = marchings[beg].beg;
				int mEnd = marchings[end - 1].end;
				int nTasks = end - beg;
				int nLocalEvaluation = mEnd - mBegin;
				int nEvaluation = points.size() / 3;

				Mat inputMat( nLocalEvaluation, 3 );
				Mat outputMat;

				for( int i = mBegin; i < mEnd; i++ )
				{
					int localI = i - mBegin;
					inputMat( 0, localI ) = points[i * 3];
					inputMat( 1, localI ) = points[i * 3 + 1];
					inputMat( 2, localI ) = points[i * 3 + 2];
				}

				for( int i = 0; i < m_densityLayers.size(); i++ )
				{
					RPML_ASSERT( inputMat.col() == m_densityLayers[i]->inputDimensions() );
					m_densityLayers[i]->forward( &outputMat, inputMat, nullptr );
					RPML_ASSERT( outputMat.col() == m_densityLayers[i]->outputDimensions() );
					inputMat.swap( outputMat );
				}

				Mat densityMat = inputMat;

				// combine dir
				Mat combinedMat( densityMat.row(), densityMat.col() + 16 );
				for( int i = beg; i < end; i++ )
				{
					NeRFInput input = inputs[i];
					float sh_encode[16] = {};
					sh_L4( sh_encode, input.rd[0], input.rd[1], input.rd[2] );

					for( int j = marchings[i].beg; j < marchings[i].end; j++ )
					{
						int localJ = j - mBegin;

						// original
						for( int k = 0; k < MLP_DENSITY_OUT; ++k )
						{
							combinedMat( k, localJ ) = densityMat( k, localJ );
						}
						// dir
						for( int k = 0; k < 16; ++k )
						{
							combinedMat( MLP_DENSITY_OUT + k, localJ ) = sh_encode[k];
						}
					}
				}
				inputMat.swap( combinedMat );

				for( int i = 0; i < m_colorLayers.size(); i++ )
				{
					RPML_ASSERT( inputMat.col() == m_colorLayers[i]->inputDimensions() );
					m_colorLayers[i]->forward( &outputMat, inputMat, nullptr );
					RPML_ASSERT( outputMat.col() == m_colorLayers[i]->outputDimensions() );
					inputMat.swap( outputMat );
				}

				Mat dL_dC( inputMat.row(), inputMat.col() );
				Mat dL_dSigma( inputMat.row(), 1 );
				for( int i = beg; i < end; i++ )
				{
					float oColor[3] = {};
					float T = 1.0f;

					// printf( "[%d] pts %d\n", i, marchings[i].end - marchings[i].beg );
					for( int j = marchings[i].beg; j < marchings[i].end; j++ )
					{
						int localJ = j - mBegin;
						float sigma = densityActivation( densityMat( 0, localJ ) );

						float c[3];
						for( int k = 0; k < 3; k++ )
						{
							c[k] = rgbActivation( inputMat( k, localJ ) );
						}
						float a = 1.0f - std::exp( -sigma * dt );

						for( int k = 0; k < 3; k++ )
						{
							oColor[k] += T * a * c[k];
						}

						T *= ( 1.0f - a );

						if( T < Teps )
							break;
					}

					for( int k = 0; k < 3; ++k )
					{
						outputs[i].color[k] = oColor[k];
					}
				}
				g.doneElements( end - beg );
			} );
			while( !g.isFinished() )
			{
				m_pool.processTask();
			}
		}
		std::vector<NeRFMarching> m_marchings;
		std::vector<float> m_points;

		std::vector<std::unique_ptr<Layer>> m_densityLayers;
		std::vector<std::unique_ptr<Layer>> m_colorLayers;
		bool m_hasOc = false;
		OccupancyGrid m_occupancyGrid;
		pr::ThreadPool m_pool;
	};

} // namespace rpml