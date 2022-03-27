#pragma once

#include <memory>
#include <string>
#include <random>
#include <mutex>
#include <algorithm>

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

namespace rpml
{
	const float pi = 3.14159265358979323846f;

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


	inline int div_round_up( int val, int divisor )
	{
		return ( val + divisor - 1 ) / divisor;
	}
	inline int next_multiple( int val, int divisor )
	{
		return div_round_up( val, divisor ) * divisor;
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
			for( int i = 0; i < m_data.size(); i++ )
			{
				m_data[i] = x ;
			}
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
		int col() const { return m_col; }

		void swap( Mat& rhs )
		{
			std::swap( m_data, rhs.m_data );
			std::swap( m_row, rhs.m_row );
			std::swap( m_paddedRow, rhs.m_paddedRow );
			std::swap( m_col, rhs.m_col );
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
		OptimizerAdam( float alpha, float beta1 = 0.9f, float beta2 = 0.999f, float e = 0.00000001f ) 
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

			FOR_EACH_ELEMENT( (*parameter), ix, iy )
			{
				float g = gradient( ix, iy );
				float m = m_m( ix, iy ) = m_beta1 * m_m( ix, iy ) + ( 1.0f - m_beta1 ) * g;
				float v = m_v( ix, iy ) = m_beta2 * m_v( ix, iy ) + ( 1.0f - m_beta2 ) * g * g;
				float adam_m_hat = m / ( 1.0f - m_beta1t );
				float adam_v_hat = v / ( 1.0f - m_beta2t );
				( *parameter )( ix, iy ) = ( *parameter )( ix, iy ) - m_alpha * adam_m_hat / ( std::sqrt( adam_v_hat ) + m_e );
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
		virtual float draw() = 0;
	};
	class StandardRng : public Rng
	{
	public:
		StandardRng( ) 
		{
			pcg::pcg32_srandom_r( &m_rng, 541, 0 );
		}
		virtual float draw()
		{
			uint32_t x = pcg32_random_r( &m_rng );
			uint32_t bits = (x >> 9) | 0x3f800000;
			float value = *reinterpret_cast<float *>(&bits) - 1.0f;
			return value;
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
		virtual void setupPropagation() = 0;
		virtual void forward( Mat* r, const Mat& value, MatContext* context /* optional */ ) = 0;
		virtual void backward( Mat* r, const Mat& gradient, MatContext* context ) = 0;
		virtual void optimize( int nElement ) = 0;

		int inputDimensions() const { return m_inputDimensions; }
		int outputDimensions() const { return m_outputDimensions; }
	private:
		int m_inputDimensions;
		int m_outputDimensions;
	};

	class AffineLayer : public Layer
	{
	public:
		AffineLayer( int i, int o, OptimizerType optimizerType, float learningRate ) : Layer( i, o ), m_W( i, o ), m_b( 1, o )
		{
			m_oW = std::unique_ptr<Optimizer>( newOptimizer( optimizerType, learningRate ) );
			m_ob = std::unique_ptr<Optimizer>( newOptimizer( optimizerType, learningRate ) );
			m_oW->initialize( m_W.row(), m_W.col() );
			m_ob->initialize( m_b.row(), m_b.col() );

			setupPropagation();
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
				FOR_EACH_ELEMENT( m_b, ix, iy )
				{
					m_b( ix, iy ) = drawRange( -k, k, rng );
				}
			} 
			else if( initType == InitializationType::He )
			{
				float s = std::sqrt( 2.0f / (float)inputDimensions() );
				BoxMular m( rng );
				FOR_EACH_ELEMENT( m_W, ix, iy )
				{
					m_W( ix, iy ) = m.draw() * s;
				}
				FOR_EACH_ELEMENT( m_b, ix, iy )
				{
					m_b( ix, iy ) = m.draw() * s;
				}
			}
			else
			{
				RPML_ASSERT( 0 );
			}
		}
		virtual void setupPropagation()
		{
			m_dW.setShape( m_W.row(), m_W.col() );
			m_db.setShape( m_b.row(), m_b.col() );
			m_dW.fill( 0.0f );
			m_db.fill( 0.0f );
		}
		virtual void forward( Mat* r, const Mat& value, MatContext* context /* optional */ )
		{
			if( context )
			{
				context->var( "x" ) = value; // input ( N, input )
			}
			mul( r, value, m_W );
			addVectorForEachRow( r, m_b );
		}
		virtual void backward( Mat* r, const Mat& gradient, MatContext* context )
		{
			const Mat& x = context->var( "x" );
			Mat& dW = context->var( "dW" );
			Mat& trx = context->var( "trx" );
			transpose( &trx, x );
			mul( &dW, trx, gradient );

			Mat& db = context->var( "db" );
			vertialSum( &db, gradient );

			{
				std::lock_guard<std::mutex> lc( m_dmutex );
				add( &m_dW, dW );
				add( &m_db, db );
			}

			Mat& trW = context->var( "trW" );
			transpose( &trW, m_W );
			mul( r, gradient, trW );
		}
		virtual void optimize( int nElement ) 
		{
			m_oW->optimize( &m_W, m_dW, nElement );
			m_ob->optimize( &m_b, m_db, nElement );
		}
		Mat m_W;  // ( input, output )
		Mat m_b;  // ( 1, output )
		Mat m_dW; // Å›L/Å›W = ( input, output )
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
		virtual void setupPropagation() {}
		virtual void optimize( int nElement ) {}

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
		virtual void setupPropagation() {}
		virtual void initialize( InitializationType initType, Rng* rng ) {}
		virtual void optimize( int nElement ) {}
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
		virtual void setupPropagation() {}
		virtual void initialize( InitializationType initType, Rng* rng ) {}
		virtual void optimize( int nElement ) {}
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
		virtual void setupPropagation() {}
		virtual void optimize( int nElement ) {}

		Config m_config;
	};

	class MultiResolutionHashEncoder : public Layer
	{
	public:
		struct Config
		{
			int L = 16;
			int T = std::pow( 2, 16 );
			int F = 2;
			int Nmin = 16;
			int Nmax = 10000;

			float b() const
			{
				return std::exp( std::log( (float)Nmax / (float)Nmin ) / ( L - 1 ) );
			}
		};
		static int output( int input, const Config& config )
		{
			return config.L * config.F + input;
		}
		uint32_t hash_nd( const uint32_t* xis, int d )
		{
			RPML_ASSERT( d <= 7 );
			const uint32_t primes[7] = { 9973, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737 };
			uint32_t h = 0;
			for( uint32_t i = 0; i < d; ++i )
			{
				h ^= xis[i] * primes[i];
			}
			return h;
		}
		MultiResolutionHashEncoder( int i, int o, const Config& config ) : Layer( i, o ), m_config( config ) 
		{
			m_features.resize( m_config.L );
			m_dfeatures.resize( m_config.L );
			m_optimizers.resize( m_config.L );
			for( int i = 0; i < m_config.L ; i++ )
			{
				m_features[i].setShape( m_config.T, m_config.F );
				m_dfeatures[i].setShape( m_config.T, m_config.F );
				m_dfeatures[i].fill( 0.0f );
				m_optimizers[i] = std::unique_ptr<Optimizer>( newOptimizer( OptimizerType::Adam, 0.005f ) );
				m_optimizers[i]->initialize( m_config.T, m_config.F );
			}
		}
		virtual void initialize( InitializationType initType, Rng* rng ) 
		{
			for( int i = 0; i < m_features.size() ; i++ )
			{
				FOR_EACH_ELEMENT( m_features[i], ix, iy )
				{
					m_features[i]( ix, iy ) = -1.0f + 2.0f * rng->draw();
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

			// copy original 
			for( int row = 0; row < value.row(); row++ )
			{
				for( int col = 0; col < value.col(); col++ )
				{
					( *r )( m_config.F * m_config.L + col, row ) = value( col, row );
				}
			}

			const int dim = inputDimensions();
			const float b = m_config.b();

			std::vector<uint32_t> hash_coordinate( 1 << dim );
			std::vector<float> weights( 1 << dim );

			std::vector<uint32_t> hash_inputs( dim );
			std::vector<float> featureVector( dim );

			for( int row = 0; row < value.row() ; row++ )
			{
				float resolution = m_config.Nmin;
				for( int l = 0 ; l < m_config.L ; l++ )
				{
					float res = floor( resolution );

					// get hash_inputs and its hash_coordinate
					for( uint32_t bits = 0; bits < ( 1 << dim ); bits++ )
					{
						float w = 1.0f;
						for( int d = 0; d < dim; ++d )
						{
							float x_in = value( d, row );

							float xf = x_in * res;
							uint32_t xi = xf;
							float u = xf - xi;

							if( bits & ( 1 << d ) )
							{
								w *= u;
								hash_inputs[d] = xi;
							}
							else
							{
								w *= 1.0f - u;
								hash_inputs[d] = xi + 1;
							}
						}
						weights[bits] = w;
						hash_coordinate[bits] = hash_nd( hash_inputs.data(), dim );
					}

#if !defined( RPML_DISABLE_ASSERT )
					float sw = 0.0f;
					for( auto w : weights )
					{
						sw += w;
					}
					RPML_ASSERT( fabs( sw - 1.0f ) < 0.001f );
#endif
					// linear interpolate
					std::fill( featureVector.begin(), featureVector.end(), 0.0f );
					for( uint32_t bits = 0; bits < ( 1 << dim ); bits++ )
					{
						float w = weights[bits];
						uint32_t index = hash_coordinate[bits] % m_config.T;
						for( int fdim = 0 ; fdim < m_config.F ; fdim++ )
						{
							featureVector[fdim] += w * m_features[l]( fdim, index );
						}
					}

					// store feature vector
					for( int fdim = 0; fdim < m_config.F; fdim++ )
					{
						( *r )( m_config.F * l + fdim, row ) = featureVector[fdim];
					}
					
					// go to next resolution
					resolution *= b;
				}
			}
		}

		// can remove?
		virtual void setupPropagation()
		{
			for( int i = 0; i < m_dfeatures.size(); i++ )
			{
				m_dfeatures[i].fill( 0.0f );
			}
		}
		virtual void backward( Mat* r, const Mat& gradient, MatContext* context ) 
		{
			const Mat& value = context->var( "value" );

			const int dim = inputDimensions();
			const float b = m_config.b();

			std::vector<uint32_t> hash_coordinate( 1 << dim );
			std::vector<float> weights( 1 << dim );

			std::vector<uint32_t> hash_inputs( dim );
			std::vector<float> dfeatureVector( dim );

			std::lock_guard<std::mutex> lc( m_dmutex ); // optimize later

			for( int row = 0; row < value.row(); row++ )
			{
				float resolution = m_config.Nmin;
				for( int l = 0; l < m_config.L; l++ )
				{
					float res = floor( resolution );

					// get hash_inputs and its hash_coordinate
					for( uint32_t bits = 0; bits < ( 1 << dim ); bits++ )
					{
						float w = 1.0f;
						for( int d = 0; d < dim; ++d )
						{
							float x_in = value( d, row );

							float xf = x_in * res;
							uint32_t xi = xf;
							float u = xf - xi;

							if( bits & ( 1 << d ) )
							{
								w *= u;
								hash_inputs[d] = xi;
							}
							else
							{
								w *= 1.0f - u;
								hash_inputs[d] = xi + 1;
							}
						}
						weights[bits] = w;
						hash_coordinate[bits] = hash_nd( hash_inputs.data(), dim );
					}

#if !defined( RPML_DISABLE_ASSERT )
					float sw = 0.0f;
					for( auto w : weights )
					{
						sw += w;
					}
					RPML_ASSERT( fabs( sw - 1.0f ) < 0.001f );
#endif
					// linear interpolate
					for( uint32_t bits = 0; bits < ( 1 << dim ); bits++ )
					{
						float w = weights[bits];
						uint32_t index = hash_coordinate[bits] % m_config.T;
						for( int fdim = 0; fdim < m_config.F; fdim++ )
						{
							float g = gradient( m_config.F * l + fdim, row );
							m_dfeatures[l]( fdim, index ) += g * w;
						}
					}

					// go to next resolution
					resolution *= b;
				}
			}
		}
		
		
		virtual void optimize( int nElement ) 
		{
			for( int i = 0; i < m_config.L ; i++ )
			{
				m_optimizers[i]->optimize( &m_features[i], m_dfeatures[i], nElement );
			}
		}

		Config m_config;

		std::vector<Mat> m_features;
		std::vector<Mat> m_dfeatures;
		std::vector<std::unique_ptr<Optimizer>> m_optimizers;

		std::mutex m_dmutex;
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
			m_rng = std::unique_ptr<Rng>( new StandardRng() );

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
						encoder->initialize( config.m_initType, m_rng.get() );
						m_layers.emplace_back( std::move( encoder ) );

						input = encoderOutput;
					}
					else if( config.m_encoderType == EncoderType::MultiResolutionHash )
					{
						int encoderOutput = MultiResolutionHashEncoder::output( input, MultiResolutionHashEncoder::Config() );
						std::unique_ptr<Layer> encoder = std::unique_ptr<Layer>( new MultiResolutionHashEncoder( input, encoderOutput, MultiResolutionHashEncoder::Config() ) );
						encoder->initialize( config.m_initType, m_rng.get() );
						m_layers.emplace_back( std::move( encoder ) );

						input = encoderOutput;
					}
				}

				std::unique_ptr<Layer> layer( new AffineLayer( input, output, config.m_optimType, config.m_learningRate ) );
				layer->initialize( config.m_initType, m_rng.get() );
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

					activation->initialize( config.m_initType, m_rng.get() );
					m_layers.emplace_back( std::move( activation ) );
				}
			}
		}
		float train( const Mat& x, const Mat& y )
		{
			RPML_ASSERT( x.row() == y.row() );

			int nElement = x.row();

			for( int i = 0; i < m_layers.size(); i++ )
			{
				m_layers[i]->setupPropagation();
			}

			float loss = 0.0f;
			std::mutex lmutex;

			pr::TaskGroup g;
			g.addElements( nElement );
			m_pool.enqueueFor( nElement, 2, [&]( int64_t beg, int64_t end ) 
			{
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
				{
					std::lock_guard<std::mutex> lc( lmutex );
					loss += L;
				}

				// MSE backward is "x - y"
				sub( &inputMat, slicedY );

				for( int i = (int)m_layers.size() - 1; 0 <= i; i-- )
				{
					RPML_ASSERT( inputMat.col() == m_layers[i]->outputDimensions() );
					m_layers[i]->backward( &outputMat, inputMat, &localStorage->layerContexts[i] );

					if( i != 0 )
					{
						RPML_ASSERT( outputMat.col() == m_layers[i]->inputDimensions() );
					}
					inputMat.swap( outputMat );
				}
				g.doneElements( end - beg );
			} );
			g.waitForAllElementsToFinish();

			for( int i = 0; i < m_layers.size(); i++ )
			{
				m_layers[i]->optimize( nElement );
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
		void forwardMT( Mat* r, const Mat& x )
		{
			int outputDim = m_layers[m_layers.size() - 1]->outputDimensions();
			( *r ).setShape( x.row(), outputDim );

			int nElement = x.row();
			std::mutex lmutex;

			pr::TaskGroup g;
			g.addElements( nElement );
			m_pool.enqueueFor( nElement, 2, [&]( int64_t beg, int64_t end )
			{
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
			g.waitForAllElementsToFinish();

		}

		std::vector<std::unique_ptr<Layer>> m_layers;
		std::unique_ptr<Rng> m_rng;
		pr::ThreadPool m_pool;
		std::map<std::thread::id, std::shared_ptr<LocalStorage>> m_localStorages;
		std::mutex m_localMutex;
	};
} // namespace rpml