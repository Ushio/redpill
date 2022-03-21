#pragma once

#include <memory>
#include <string>
#include <random>
#include <mutex>
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
	for( int ix = 0; ix < m.col(); ix++ ) \
		for( int iy = 0; iy < m.row(); iy++ )

namespace rpml
{
	struct Mat
	{
		Mat() {}
		Mat( int row, int col ) : m_row( row ), m_col( col ), m_data( col * row ){};
		Mat( int row, int col, const std::vector<float> data ) : m_row( row ), m_col( col ), m_data( data ) { RPML_ASSERT( m_row * m_col == m_data.size() ); };

		void reinit( int row, int col )
		{
			m_row = row;
			m_col = col;
			m_data.clear();
			m_data.resize( row * col );
			RPML_ASSERT( m_row * m_col == m_data.size() );
		}
		float& operator()( int x, int y )
		{
			RPML_ASSERT( 0 <= y && y < m_row );
			RPML_ASSERT( 0 <= x && x < m_col );
			return m_data[m_row * x + y];
		}
		float operator()( int x, int y ) const
		{
			RPML_ASSERT( 0 <= y && y < m_row );
			RPML_ASSERT( 0 <= x && x < m_col );
			return m_data[m_row * x + y];
		}
		int row() const { return m_row; }
		int col() const { return m_col; }

	private:
		std::vector<float> m_data;
		int m_row = 0;
		int m_col = 0;
	};

	inline Mat transpose( const Mat& m )
	{
		Mat r( m.col(), m.row() );
		FOR_EACH_ELEMENT( r, ix, iy )
		{
			r( ix, iy ) = m( iy, ix );
		}
		return r;
	}

	inline Mat fromRowMajor( int row, int col, std::initializer_list<float> init )
	{
		return transpose( Mat( col, row, init ) );
	}
	inline Mat fromColMajor( int row, int col, std::initializer_list<float> init )
	{
		return Mat( row, col, init );
	}

	inline Mat operator*( const Mat& ma, const Mat& mb )
	{
		RPML_ASSERT( ma.col() == mb.row() );
		Mat r( ma.row(), mb.col() );

		FOR_EACH_ELEMENT( r, ix, iy )
		{
			const int n = ma.col();
			float v = 0.0f;
			for( int i = 0; i < n; i++ )
			{
				v += ma( i, iy ) * mb( ix, i );
			}
			r( ix, iy ) = v;
		}
		return r;
	}

	inline Mat operator+( const Mat& ma, const Mat& mb )
	{
		RPML_ASSERT( ma.col() == mb.col() );
		RPML_ASSERT( ma.row() == mb.row() );

		Mat r( ma.row(), ma.col() );
		FOR_EACH_ELEMENT( r, ix, iy )
		{
			r( ix, iy ) = ma( ix, iy ) + mb( ix, iy );
		}
		return r;
	}
	inline Mat operator-( const Mat& ma, const Mat& mb )
	{
		RPML_ASSERT( ma.col() == mb.col() );
		RPML_ASSERT( ma.row() == mb.row() );

		Mat r( ma.row(), ma.col() );
		FOR_EACH_ELEMENT( r, ix, iy )
		{
			r( ix, iy ) = ma( ix, iy ) - mb( ix, iy );
		}
		return r;
	}
	inline Mat multiplyScalar( const Mat& m, float s )
	{
		Mat r( m.row(), m.col() );
		FOR_EACH_ELEMENT( r, ix, iy )
		{
			r( ix, iy ) = m( ix, iy ) * s;
		} 
		return r;
	}
	inline Mat addVectorForEachRow( const Mat& m, const Mat& v )
	{
	    Mat r(m.row(), m.col());
		FOR_EACH_ELEMENT( r, ix, iy )
	    {
			r( ix, iy ) = m( ix, iy ) + v( ix, 0 );
	    }
		return r;
	}

	inline Mat vertialSum( const Mat& m )
	{
		Mat r( 1, m.col() );
		for( int ix = 0; ix < m.col(); ix++ )
		{
			float s = 0.0f;
			for( int iy = 0; iy < m.row(); iy++ )
			{
				s += m( ix, iy );
			}
			r( ix, 0 ) = s;
		}
		return r;
	}
	inline Mat sliceH( const Mat& x, int beg, int end )
	{
		RPML_ASSERT( 0 <= beg );
		RPML_ASSERT( end <= x.row() );
		RPML_ASSERT( beg <= end );

		int localN = end - beg;
		Mat r( localN, x.col() );
		for( int i = 0; i < localN; i++ )
		{
			for( int j = 0; j < x.col(); ++j )
			{
				r( j, i ) = x( j, beg + i );
			}
		}
		return r;
	}

	template <class F>
	Mat apply( const Mat& m, F f )
	{
		Mat r( m.row(), m.col() );
		FOR_EACH_ELEMENT( r, ix, iy )
		{
			r( ix, iy ) = f( m( ix, iy ) );
		}
		return r;
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
			*parameter = *parameter - multiplyScalar( gradient, m_alpha / nElement );
		}
	private:
		float m_alpha;
	};
	class OptimizerAdam : public Optimizer
	{
	public:
		OptimizerAdam( float alpha, float beta1 = 0.9f, float beta2 = 0.999f, float e = 0.00000001f ) 
			: m_alpha( alpha ), m_beta1( beta1 ), m_beta2( beta2 ), m_e(e)
		{
			m_c = m_alpha * std::sqrt( 1.0f - m_beta2 ) / ( 1.0f - m_beta1 );
		}
		virtual void initialize( int row, int col )
		{
			m_m.reinit( row, col );
			m_v.reinit( row, col );
		}
		virtual void optimize( Mat* parameter, const Mat& gradient, int nElement )
		{
			FOR_EACH_ELEMENT( (*parameter), ix, iy )
			{
				float g = gradient( ix, iy );
				float m = m_m( ix, iy ) = m_beta1 * m_m( ix, iy ) + ( 1.0f - m_beta1 ) * g;
				float v = m_v( ix, iy ) = m_beta2 * m_v( ix, iy ) + ( 1.0f - m_beta2 ) * g * g;
				( *parameter )( ix, iy ) = ( *parameter )( ix, iy ) - m * m_c / ( std::sqrt( v ) + m_e );
			}
		}

	private:
		float m_alpha;
		float m_beta1;
		float m_beta2;
		float m_e;
		float m_c;
		Mat m_m;
		Mat m_v;
	};

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
			const float pi = 3.14159265358979323846f;
			
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
		Tanh,
		Sigmoid
	};

	class LayerContext
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
		virtual Mat forward( const Mat& value, LayerContext* context ) = 0;
		virtual Mat backward( const Mat& gradient, LayerContext* context ) = 0;
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
			if( optimizerType == OptimizerType::SGD )
			{
				m_oW = std::unique_ptr<Optimizer>( new OptimizerSGD( learningRate ) );
				m_ob = std::unique_ptr<Optimizer>( new OptimizerSGD( learningRate ) );
			}
			else if( optimizerType == OptimizerType::Adam)
			{
				m_oW = std::unique_ptr<Optimizer>( new OptimizerAdam( learningRate ) );
				m_ob = std::unique_ptr<Optimizer>( new OptimizerAdam( learningRate ) );
			}
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
					m_b( ix, iy ) = 0.0f;
				}
			}
			else
			{
				RPML_ASSERT( 0 );
			}
		}
		virtual void setupPropagation()
		{
			m_dW.reinit( m_W.row(), m_W.col() );
			m_db.reinit( m_b.row(), m_b.col() );
		}
		virtual Mat forward( const Mat& value, LayerContext* context )
		{
			context->var( "x" ) = value; // input ( N, input )
			return addVectorForEachRow( value * m_W, m_b );
		}
		virtual Mat backward( const Mat& gradient, LayerContext* context )
		{
			const Mat& x = context->var( "x" );
			Mat dW = transpose( x ) * gradient;
			Mat db = vertialSum( gradient );

			{
				std::lock_guard<std::mutex> lc( m_dmutex );
				m_dW = m_dW + dW;
				m_db = m_db + db;
			}

			return gradient * transpose( m_W );
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

	class ReLULayer : public Layer
	{
	public:
		ReLULayer( int i, int o ) : Layer( i, o ) {}

		virtual Mat forward( const Mat& value, LayerContext* context )
		{
			context->var( "x" ) = value;

			Mat r( value.row(), value.col() );
			FOR_EACH_ELEMENT( r, ix, iy )
			{
				float x = value( ix, iy );
				r( ix, iy ) = maxss( x, 0.0f );
			}
			return r;
		}
		virtual Mat backward( const Mat& gradient, LayerContext* context )
		{
			const Mat& x = context->var( "x" );

			Mat r( gradient.row(), gradient.col() );
			FOR_EACH_ELEMENT( gradient, ix, iy )
			{
				float d = 0.0f < x( ix, iy ) ? 1.0f : 0.0f;
				r( ix, iy ) = d * gradient( ix, iy );
			}
			return r;
		}
		virtual void initialize( InitializationType initType, Rng* rng ) {}
		virtual void setupPropagation() {}
		virtual void optimize( int nElement ) {}
	};
	class SigmoidLayer : public Layer
	{
	public:
		SigmoidLayer( int i, int o ) : Layer( i, o ) {}
		virtual Mat forward( const Mat& value, LayerContext* context )
		{
			Mat r( value.row(), value.col() );
			FOR_EACH_ELEMENT( r, ix, iy )
			{
				float x = value( ix, iy );
				r( ix, iy ) = 1.0f / ( 1.0f + std::exp( -x ) );
			}
			context->var( "y" ) = r;
			return r;
		}
		virtual Mat backward( const Mat& gradient, LayerContext* context )
		{
			const Mat& y = context->var( "y" );

			Mat r( gradient.row(), gradient.col() );
			FOR_EACH_ELEMENT( gradient, ix, iy )
			{
				float d = y( ix, iy ) * ( 1.0f - y( ix, iy ) );
				r( ix, iy ) = d * gradient( ix, iy );
			}
			return r;
		}
		virtual void setupPropagation() {}
		virtual void initialize( InitializationType initType, Rng* rng ) {}
		virtual void optimize( int nElement ) {}
	};
	class TanhLayer : public Layer
	{
	public:
		TanhLayer( int i, int o ) : Layer( i, o ) {}
		virtual Mat forward( const Mat& value, LayerContext* context )
		{
			Mat r( value.row(), value.col() );
			FOR_EACH_ELEMENT( r, ix, iy )
			{
				float x = value( ix, iy );
				r( ix, iy ) = std::tanh( x );
			}
			context->var( "y" ) = r;
			return r;
		}
		virtual Mat backward( const Mat& gradient, LayerContext* context )
		{
			const Mat& y = context->var( "y" );

			Mat r( gradient.row(), gradient.col() );
			FOR_EACH_ELEMENT( gradient, ix, iy )
			{
				float d = 1.0f - y( ix, iy ) * y( ix, iy );
				r( ix, iy ) = d * gradient( ix, iy );
			}
			return r;
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
	inline Mat mse_backward( const Mat& output, const Mat& ref )
	{
		return output - ref;
	}

	class MLPConfig
	{
	public:
		float m_learningRate = 0.01f;
		std::vector<int> m_shape;
		ActivationType m_activationType = ActivationType::ReLU;
		InitializationType m_initType = InitializationType::He;
		OptimizerType m_optimType = OptimizerType::Adam;
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
#undef PROP
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
			RPML_ASSERT( x.col() == y.col() );

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
				std::vector<LayerContext> layerContexts( m_layers.size() );

				Mat m = sliceH( x, beg, end );
				Mat slicedY = sliceH( y, beg, end );
				
				for( int i = 0; i < m_layers.size(); i++ )
				{
					RPML_ASSERT( m.col() == m_layers[i]->inputDimensions() );
					m = m_layers[i]->forward( m, &layerContexts[i] );
					RPML_ASSERT( m.col() == m_layers[i]->outputDimensions() );
				}

				// m: estimated result
				float L = mse( m, slicedY );
				{
					std::lock_guard<std::mutex> lc( lmutex );
					loss += L;
				}

				m = mse_backward( m, slicedY );
				for( int i = (int)m_layers.size() - 1; 0 <= i; i-- )
				{
					RPML_ASSERT( m.col() == m_layers[i]->outputDimensions() );
					m = m_layers[i]->backward( m, &layerContexts[i] );
					RPML_ASSERT( m.col() == m_layers[i]->inputDimensions() );
				}
				g.doneElements( end - beg );
			} );
			g.waitForAllElementsToFinish();

			for( int i = 0; i < m_layers.size(); i++ )
			{
				m_layers[i]->optimize( nElement );
			}

			//std::vector<LayerContext> layerContexts( m_layers.size() );
			//Mat m = x;
			//for( int i = 0; i < m_layers.size(); i++ )
			//{
			//	RPML_ASSERT( m.col() == m_layers[i]->inputDimensions() );
			//	m = m_layers[i]->forward( m, &layerContexts[i] );
			//	RPML_ASSERT( m.col() == m_layers[i]->outputDimensions() );
			//}

			//// m: estimated result
			//float loss = mse( m, y );

			//m = mse_backward( m, y );
			//for( int i = (int)m_layers.size() - 1; 0 <= i; i-- )
			//{
			//	RPML_ASSERT( m.col() == m_layers[i]->outputDimensions() );
			//	m = m_layers[i]->backward( m, &layerContexts[i] );
			//	RPML_ASSERT( m.col() == m_layers[i]->inputDimensions() );
			//}
			//for( int i = 0; i < m_layers.size(); i++ )
			//{
			//	m_layers[i]->optimize( nElement );
			//}

			return loss;
		}
		Mat forward( const Mat& x )
		{
			std::vector<LayerContext> layerContexts( m_layers.size() );
			Mat m = x;
			for( int i = 0; i < m_layers.size(); i++ )
			{
				RPML_ASSERT( m.col() == m_layers[i]->inputDimensions() );
				m = m_layers[i]->forward( m, &layerContexts[i] );
				RPML_ASSERT( m.col() == m_layers[i]->outputDimensions() );
			}
			return m;
		}

		std::vector<std::unique_ptr<Layer>> m_layers;
		std::unique_ptr<Rng> m_rng;
		pr::ThreadPool m_pool;
	};
} // namespace rpml