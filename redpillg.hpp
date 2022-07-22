#pragma once

#include "redpill.hpp"
#include <Orochi/Orochi.h>
#include <queue>
#include <memory>
#include <intrin.h>
#define RPMLX_ASSERT( ExpectTrue ) \
	if( ( ExpectTrue ) == 0 )     \
	{                             \
		__debugbreak();           \
	}

namespace rpml
{
	class Buffer
	{
	public:
		Buffer( const Buffer& ) = delete;
		void operator=( const Buffer& ) = delete;

		Buffer( int64_t bytes )
			: m_bytes( std::max( bytes, 1LL ) )
		{
			oroMalloc( &m_ptr, m_bytes );
		}
		~Buffer()
		{
			oroFree( m_ptr );
		}
		int64_t bytes() const
		{
			return m_bytes;
		}
		char* data()
		{
			return (char*)m_ptr;
		}
	private:
		int64_t m_bytes;
		oroDeviceptr m_ptr;
	};

	enum class CompileMode
	{
		Release,
		RelwithDebInfo
	};
	inline void loadAsVector( std::vector<char>* buffer, const char* fllePath )
	{
		FILE* fp = fopen( fllePath, "rb" );
		if( fp == nullptr )
		{
			return;
		}

		fseek( fp, 0, SEEK_END );

		buffer->resize( ftell( fp ) );

		fseek( fp, 0, SEEK_SET );

		size_t s = fread( buffer->data(), 1, buffer->size(), fp );
		if( s != buffer->size() )
		{
			buffer->clear();
			return;
		}
		fclose( fp );
		fp = nullptr;
	}

	struct ShaderArgument
	{
		template <class T>
		void add( T p )
		{
			int bytes = sizeof( p );
			int location = m_buffer.size();
			m_buffer.resize( m_buffer.size() + bytes );
			memcpy( m_buffer.data() + location, &p, bytes );
			m_locations.push_back( location );
		}
		void clear()
		{
			m_buffer.clear();
			m_locations.clear();
		}

		std::vector<void*> kernelParams() const
		{
			std::vector<void*> ps;
			for( int i = 0; i < m_locations.size(); i++ )
			{
				ps.push_back( (void*)( m_buffer.data() + m_locations[i] ) );
			}
			return ps;
		}

	private:
		std::vector<char> m_buffer;
		std::vector<int> m_locations;
	};

	class Shader
	{
	public:
		Shader( const char* filename, const char* kernelLabel, const std::vector<std::string>& includeDirs, const std::vector<std::string>& extraArgs, CompileMode compileMode )
		{
			std::vector<char> src;
			loadAsVector( &src, filename );
			PR_ASSERT( 0 < src.size() );
			src.push_back( '\0' );

			orortcProgram program = 0;
			orortcCreateProgram( &program, src.data(), kernelLabel, 0, 0, 0 );
			std::vector<std::string> options;
			options.push_back( "-std=c++11");

			for( int i = 0; i < includeDirs.size(); ++i )
			{
				options.push_back( "-I " + includeDirs[i] );
			}

			if( compileMode == CompileMode::RelwithDebInfo )
			{
				options.push_back( "-G" );
			}

			for( int i = 0; i < extraArgs.size(); ++i )
			{
				options.push_back( extraArgs[i] );
			}

			std::vector<const char*> optionChars;
			for( int i = 0; i < options.size(); ++i )
			{
				optionChars.push_back( options[i].c_str() );
			}

			orortcResult compileResult = orortcCompileProgram( program, optionChars.size(), optionChars.data() );

			size_t logSize = 0;
			orortcGetProgramLogSize( program, &logSize );
			if( 1 < logSize )
			{
				std::vector<char> compileLog( logSize );
				orortcGetProgramLog( program, compileLog.data() );
				printf( "%s", compileLog.data() );
			}
			PR_ASSERT( compileResult == ORORTC_SUCCESS );

			size_t codeSize = 0;
			orortcGetCodeSize( program, &codeSize );

			std::vector<char> codec( codeSize );
			orortcGetCode( program, codec.data() );

			orortcDestroyProgram( &program );

			orortcResult re;
			oroError e = oroModuleLoadData( &m_module, codec.data() );
			PR_ASSERT( e == oroSuccess );
		}
		~Shader()
		{
			oroModuleUnload( m_module );
		}
		void launch( const char* name,
					 const ShaderArgument& arguments,
					 unsigned int gridDimX, unsigned int gridDimY, unsigned int gridDimZ,
					 unsigned int blockDimX, unsigned int blockDimY, unsigned int blockDimZ,
					 oroStream hStream )
		{
			if( m_functions.count( name ) == 0 )
			{
				oroFunction f = 0;
				oroError e = oroModuleGetFunction( &f, m_module, name );
				PR_ASSERT( e == oroSuccess );
				m_functions[name] = f;
			}

			auto params = arguments.kernelParams();
			oroFunction f = m_functions[name];
			oroError e = oroModuleLaunchKernel( f, gridDimX, gridDimY, gridDimZ, blockDimX, blockDimY, blockDimZ, 0, hStream, params.data(), 0 );
			PR_ASSERT( e == oroSuccess );
		}

	private:
		oroModule m_module = 0;
		std::map<std::string, oroFunction> m_functions;
	};

	class MLPg
	{
	public:
		struct LearningTask 
		{
			oroEvent learnEvent = 0;
			Mat input;
			Mat refs;
			void wait()
			{
				if( learnEvent )
				{
					oroEventSynchronize( learnEvent );
					oroEventDestroy( learnEvent );
					learnEvent = 0;
				}
			}
		};
		MLPg( const MLPConfig& config, std::string kernels ) : m_config( config )
		{
			std::vector<std::string> macros;

			if( config.m_encoderType == EncoderType::Frequency )
			{
				macros.push_back( "-DFREQ_N=" + std::to_string( config.m_frequencyEncoderConfig.N ) );
			}
			else if( config.m_encoderType == EncoderType::MultiResolutionHash )
			{
				auto cfg = m_config.m_multiResolutionHashConfig;
				macros.push_back( "-DGRID_INPUT_DIM=" + std::to_string( config.m_shape[0] ) );
				macros.push_back( "-DGRID_L=" + std::to_string( cfg.L ) );
				macros.push_back( "-DGRID_T=" + std::to_string( cfg.T ) );
				macros.push_back( "-DGRID_F=" + std::to_string( cfg.F ) );
				macros.push_back( "-DGRID_NMIN=(float)(" + std::to_string( cfg.Nmin ) + ")" );
				macros.push_back( "-DGRID_B=(float)(" + std::to_string( cfg.b ) + ")" );
			}

			int location = 0;
			for( int i = 0; i < config.m_shape.size() - 1; i++ )
			{
				int input = config.m_shape[i];
				int output = config.m_shape[i + 1];
				if( i == 0 )
				{
					if( config.m_encoderType == EncoderType::Frequency )
					{
						input = frequencyOutputDim( input, config.m_frequencyEncoderConfig.N );
					}
					else if( config.m_encoderType ==EncoderType::MultiResolutionHash )
					{
						input = multiResolutionHashOutputDim( m_config.m_multiResolutionHashConfig.L, m_config.m_multiResolutionHashConfig.F );
						m_gridFeatureLocation = location;
						auto cfg = m_config.m_multiResolutionHashConfig;
						int elementsPerTable = cfg.T * cfg.F;
						location += elementsPerTable * cfg.L;
					}
				}

				GPUMat w = allocateGPUMat( &location, input, output );
				GPUMat b = allocateGPUMat( &location, 1, output );

				m_Ws.push_back( w );
				m_Bs.push_back( b );
			}

			m_matBuffer = std::unique_ptr<Buffer>( new Buffer( location * sizeof( float ) ) );
			m_dmatBuffer = std::unique_ptr<Buffer>( new Buffer( location * sizeof( float ) ) );
			m_adamBuffer = std::unique_ptr<Buffer>( new Buffer( location * sizeof( Adam ) ) );
			oroMemsetD32( (oroDeviceptr)m_dmatBuffer->data(), 0, m_dmatBuffer->bytes() / sizeof(float) );
			oroMemsetD32( (oroDeviceptr)m_adamBuffer->data(), 0, m_adamBuffer->bytes() / sizeof(float) );

			// initialization
			StandardRng rng;
			std::vector<float> matBuffer( location );
			for( int i = 0; i < m_Ws.size(); i++ )
			{
				int input = m_Ws[i].m_row;

				float s = std::sqrt( 2.0f / (float)input );
				BoxMular m( &rng );

				GPUMat w = m_Ws[i];
				GPUMat b = m_Bs[i];
				
				for( int ix = 0; ix < w.m_col ; ix++ )
				for( int iy = 0; iy < w.m_row; iy++ )
				{
					matBuffer[elem( ix, iy, w )] = m.draw() * s;
				}

				for( int ix = 0; ix < b.m_col; ix++ )
				for( int iy = 0; iy < b.m_row; iy++ )
				{
					matBuffer[elem( ix, iy, b )] = m.draw() * s;
				}
			}
			if( config.m_encoderType == EncoderType::MultiResolutionHash )
			{
				auto cfg = m_config.m_multiResolutionHashConfig;
				int elementsPerTable = cfg.T * cfg.F;
				int numberOfElements = elementsPerTable * cfg.L;

				for( int i = 0; i < numberOfElements ; i++ )
				{
					matBuffer[m_gridFeatureLocation + i] = -1e-4f + 2.0f * 1e-4f * rng.draw();
				}
			}

			oroMemcpyHtoD( (oroDeviceptr)m_matBuffer->data(), matBuffer.data(), matBuffer.size() * sizeof( float ) );

			m_forwardShader = std::unique_ptr<Shader>( new Shader( ( kernels + "\\mlpForward.cu" ).c_str(), "mlpForward.cu", { kernels }, macros, CompileMode::Release ) );
		}
		void train( oroStream stream, const Mat& input, const Mat& refs )
		{
			m_Is.clear();

			int location = 0;
			GPUMat inputGPU = allocateGPUMat( &location, input.row(), input.col() );
			switch( m_config.m_encoderType )
			{
			case EncoderType::None:
				m_Is.push_back( inputGPU );
				break;
			case EncoderType::Frequency: 
			{
				int output = frequencyOutputDim( input.col(), m_config.m_frequencyEncoderConfig.N );
				GPUMat encoded = allocateGPUMat( &location, input.row(), output );
				m_Is.push_back( encoded );
				break;
			}
			case EncoderType::MultiResolutionHash:
			{
				int output = multiResolutionHashOutputDim( m_config.m_multiResolutionHashConfig.L, m_config.m_multiResolutionHashConfig.F );
				GPUMat encoded = allocateGPUMat( &location, input.row(), output );
				m_Is.push_back( encoded );
				break;
			}
			}

			for( int i = 0; i < m_Ws.size(); ++i )
			{
				int outputs = m_Ws[i].m_col;
				GPUMat I = allocateGPUMat( &location, input.row(), outputs );
				m_Is.push_back( I );
			}

			GPUMat inputRefGPU = allocateGPUMat( &location, refs.row(), refs.col() );

			if( !m_intermediateBuffer || m_intermediateBuffer->bytes() < location * sizeof( float ) )
			{
				m_intermediateBuffer = std::unique_ptr<Buffer>( new Buffer( location * sizeof( float ) ) );
			}
			 
			while( 64 < m_learningQueue.size() )
			{
				std::shared_ptr<LearningTask> task = m_learningQueue.front();
				m_learningQueue.pop();
				task->wait();
			}

			std::shared_ptr<LearningTask> task = std::shared_ptr<LearningTask>( new LearningTask() );
			oroEventCreateWithFlags( &task->learnEvent, 0 );
			task->input = input;
			task->refs = refs;
			oroMemcpyHtoDAsync( ( oroDeviceptr )( m_intermediateBuffer->data() + inputGPU.m_location * sizeof( float ) ), (void*)task->input.data(), task->input.bytes(), stream );
			oroMemcpyHtoDAsync( ( oroDeviceptr )( m_intermediateBuffer->data() + inputRefGPU.m_location * sizeof( float ) ), (void*)task->refs.data(), task->refs.bytes(), stream );			

			MLPTrainArg arg;
			arg.inputMat = inputGPU;
			arg.inputRefMat = inputRefGPU;
			for( int i = 0; i < m_Ws.size(); ++i )
			{
				arg.m_Ws[i] = m_Ws[i];
				arg.m_Bs[i] = m_Bs[i];
			}
			for( int i = 0; i < m_Ws.size() + 1; ++i )
			{
				arg.m_Is[i] = m_Is[i];
			}
			arg.nLayer = m_Ws.size();
			arg.encoder = m_config.m_encoderType;
			arg.gridFeatureLocation = m_gridFeatureLocation;

			ShaderArgument trainArgs;
			trainArgs.add( m_matBuffer->data() );
			trainArgs.add( m_dmatBuffer->data() );
			trainArgs.add( m_intermediateBuffer->data() );
			trainArgs.add( arg );

			int gridDim = div_round_up( input.row(), SHARED_TENSOR_ROW );
			m_forwardShader->launch( "train", trainArgs, gridDim, 1, 1, 1, 64, 1, stream );
			m_iteration++;
			float beta1t = pow( ADAM_BETA1, m_iteration );
			float beta2t = pow( ADAM_BETA2, m_iteration );
			int nAdam = m_adamBuffer->bytes() / sizeof( Adam );

			ShaderArgument adamArgs;
			adamArgs.add( m_matBuffer->data() );
			adamArgs.add( m_dmatBuffer->data() );
			adamArgs.add( m_adamBuffer->data() );
			adamArgs.add( m_config.m_learningRate / input.row() );
			adamArgs.add( beta1t );
			adamArgs.add( beta2t );
			adamArgs.add( nAdam );
			
			m_forwardShader->launch( "adamOptimize", adamArgs, div_round_up( nAdam, 64 ), 1, 1, 64, 1, 1, stream );

			oroEventRecord( task->learnEvent, stream );
			m_learningQueue.push( task );
		}
		void synchronizeLearning()
		{
			while( 0 < m_learningQueue.size() )
			{
				std::shared_ptr<LearningTask> task = m_learningQueue.front();
				m_learningQueue.pop();
				task->wait();
			}
		}
		void takeReference( const MLP& mlp )
		{
			bool hasEncoder = dynamic_cast<const AffineLayer*>( mlp.m_layers[0].get() ) == 0;

			auto grid = dynamic_cast<const MultiResolutionHashEncoder*>( mlp.m_layers[0].get() );
			if( grid )
			{
				for( int level = 0; level < grid->m_config.L; ++level )
				{
					const Mat& feature = grid->m_features[level];

					int baseLevel = grid->m_config.T * grid->m_config.F * sizeof( float ) * level;
					for( int fi = 0; fi < grid->m_config.F; ++fi )
					{
						int bytesPerFeature = grid->m_config.T * sizeof( float );
						oroMemcpyHtoD( 
							( oroDeviceptr )( m_matBuffer->data() + m_gridFeatureLocation * sizeof( float ) + baseLevel + bytesPerFeature * fi ), 
							(void*)&feature( fi, 0 ), bytesPerFeature );
					}
				}
			}

			std::vector<const AffineLayer*> affineLayers;
			for( int i = ( hasEncoder ? 1 : 0 ); i < mlp.m_layers.size(); i += 2 )
			{
				auto affineLayer = dynamic_cast<const AffineLayer*>( mlp.m_layers[i].get() );
				affineLayers.push_back( affineLayer );
			}
			for( int i = 0; i < affineLayers.size(); ++i )
			{
				oroMemcpyHtoD( ( oroDeviceptr )( m_matBuffer->data() + m_Ws[i].m_location * sizeof( float ) ), (void*)affineLayers[i]->m_W.data(), affineLayers[i]->m_W.bytes() );
				oroMemcpyHtoD( ( oroDeviceptr )( m_matBuffer->data() + m_Bs[i].m_location * sizeof( float ) ), (void*)affineLayers[i]->m_b.data(), affineLayers[i]->m_b.bytes() );
			}
		}
		void foward( oroStream stream, const Mat& input, Mat* output )
		{
			synchronizeLearning();

			int outputDim = m_Ws[m_Ws.size() - 1].m_col;
			int location = 0;
			GPUMat inputGPU = allocateGPUMat( &location, input.row(), input.col() );
			GPUMat outputGPU = allocateGPUMat( &location, input.row(), outputDim );

			if( !m_intermediateBuffer || m_intermediateBuffer->bytes() < location * sizeof( float ) )
			{
				m_intermediateBuffer = std::unique_ptr<Buffer>( new Buffer( location * sizeof( float ) ) );
			}
			oroMemcpyHtoDAsync( ( oroDeviceptr )( m_intermediateBuffer->data() + inputGPU.m_location * sizeof( float ) ), (void*)input.data(), input.bytes(), stream );

			MLPForwardArg arg;
			arg.inputMat = inputGPU;
			arg.outputMat = outputGPU;
			for( int i = 0; i < m_Ws.size(); ++i )
			{
				arg.m_Ws[i] = m_Ws[i];
				arg.m_Bs[i] = m_Bs[i];
			}
			arg.nLayer = m_Ws.size();
			arg.encoder = m_config.m_encoderType;
			arg.gridFeatureLocation = m_gridFeatureLocation;

			ShaderArgument args;
			args.add( m_intermediateBuffer->data() );
			args.add( m_matBuffer->data() );
			args.add( arg );

			int gridDim = div_round_up( input.row(), SHARED_TENSOR_ROW );
			m_forwardShader->launch( "forward", args, gridDim, 1, 1, 1, 64, 1, stream );

			output->setShape( outputGPU.m_row, outputGPU.m_col );

			oroMemcpyDtoHAsync( output->data(), (oroDeviceptr)( m_intermediateBuffer->data() + outputGPU.m_location * sizeof(float) ), output->bytes(), stream );

			oroStreamSynchronize( stream );
		}

		std::unique_ptr<Buffer> m_matBuffer;
		std::vector<GPUMat> m_Ws;
		std::vector<GPUMat> m_Bs;
		int m_gridFeatureLocation = 0;

		// learning
		std::unique_ptr<Buffer> m_intermediateBuffer;
		std::unique_ptr<Buffer> m_dmatBuffer;
		std::unique_ptr<Buffer> m_adamBuffer;
		std::vector<GPUMat> m_Is;
		std::queue<std::shared_ptr<LearningTask>> m_learningQueue;

		std::unique_ptr<Shader> m_forwardShader;

		int m_iteration = 0;
		MLPConfig m_config;
	};

	class NeRFg
	{
	public:
		NeRFg( std::string kernels )
		{
			std::vector<std::string> macros;

			{
				m_hashConfig.L = 16;
				m_hashConfig.T = std::pow( 2, 19 );
				m_hashConfig.F = 2;
				m_hashConfig.Nmin = 16;
				m_hashConfig.b = 1.38191f;
				macros.push_back( "-DGRID_INPUT_DIM=" + std::to_string( 3 ) );
				macros.push_back( "-DGRID_L=" + std::to_string( m_hashConfig.L ) );
				macros.push_back( "-DGRID_T=" + std::to_string( m_hashConfig.T ) );
				macros.push_back( "-DGRID_F=" + std::to_string( m_hashConfig.F ) );
				macros.push_back( "-DGRID_NMIN=(float)(" + std::to_string( m_hashConfig.Nmin ) + ")" );
				macros.push_back( "-DGRID_B=(float)(" + std::to_string( m_hashConfig.b ) + ")" );
			}

			int location = 0;

			// encording
			int encodeOutputDim = multiResolutionHashOutputDim( m_hashConfig.L, m_hashConfig.F );
			m_gridFeatureLocation = location;
			int elementsPerTable = m_hashConfig.T * m_hashConfig.F;
			location += elementsPerTable * m_hashConfig.L;

			// Density Network ( 32 -> 64 -> 16 )
			m_Ws.push_back( allocateGPUMat( &location, encodeOutputDim, 64 ) );
			m_Bs.push_back( allocateGPUMat( &location, 1, 64 ) );
			m_Ws.push_back( allocateGPUMat( &location, 64, 16 ) );
			m_Bs.push_back( allocateGPUMat( &location, 1,  16 ) );

			// Color Network ( 32 -> 64 -> 64 -> 3 )
			m_Ws.push_back( allocateGPUMat( &location, 32, 64 ) );
			m_Bs.push_back( allocateGPUMat( &location, 1, 64 ) );
			m_Ws.push_back( allocateGPUMat( &location, 64, 64 ) );
			m_Bs.push_back( allocateGPUMat( &location, 1, 64 ) );
			m_Ws.push_back( allocateGPUMat( &location, 64, 3 ) );
			m_Bs.push_back( allocateGPUMat( &location, 1, 3 ) );

			m_matBuffer = std::unique_ptr<Buffer>( new Buffer( location * sizeof( float ) ) );
			m_dmatBuffer = std::unique_ptr<Buffer>( new Buffer( location * sizeof( float ) ) );
			m_adamBuffer = std::unique_ptr<Buffer>( new Buffer( location * sizeof( Adam ) ) );
			oroMemsetD32( (oroDeviceptr)m_dmatBuffer->data(), 0, m_dmatBuffer->bytes() / sizeof( float ) );
			oroMemsetD32( (oroDeviceptr)m_adamBuffer->data(), 0, m_adamBuffer->bytes() / sizeof( float ) );

			// initialization
			StandardRng rng;
			std::vector<float> matBuffer( location );
			for( int i = 0; i < m_Ws.size(); i++ )
			{
				int input = m_Ws[i].m_row;

				float s = std::sqrt( 2.0f / (float)input );
				BoxMular m( &rng );

				GPUMat w = m_Ws[i];
				GPUMat b = m_Bs[i];

				for( int ix = 0; ix < w.m_col; ix++ )
				for( int iy = 0; iy < w.m_row; iy++ )
				{
					matBuffer[elem( ix, iy, w )] = m.draw() * s;
				}

				for( int ix = 0; ix < b.m_col; ix++ )
				for( int iy = 0; iy < b.m_row; iy++ )
				{
					matBuffer[elem( ix, iy, b )] = m.draw() * s;
				}
			}
			{
				auto cfg = m_hashConfig;
				int elementsPerTable = cfg.T * cfg.F;
				int numberOfElements = elementsPerTable * cfg.L;

				for( int i = 0; i < numberOfElements; i++ )
				{
					matBuffer[m_gridFeatureLocation + i] = -1e-4f + 2.0f * 1e-4f * rng.draw();
				}
			}

			oroMemcpyHtoD( (oroDeviceptr)m_matBuffer->data(), matBuffer.data(), matBuffer.size() * sizeof( float ) );

			m_nerfSamplesBuffer = std::unique_ptr<Buffer>( new Buffer( sizeof( GPUMat ) ) );

			m_occupancyBuffer = std::unique_ptr<Buffer>( new Buffer( NERF_OCCUPANCY_GRID_T * sizeof( float ) ) );
			oroMemsetD32( (oroDeviceptr)m_occupancyBuffer->data(), as_int32( 65536.0f ), m_occupancyBuffer->bytes() / sizeof( int ) );
			m_occupancyAvgBuffer = std::unique_ptr<Buffer>( new Buffer( sizeof( float ) ) );
			oroMemsetD32( (oroDeviceptr)m_occupancyAvgBuffer->data(), 0, 1 );
			m_forwardShader = std::unique_ptr<Shader>( new Shader( ( kernels + "\\mlpForward.cu" ).c_str(), "mlpForward.cu", { kernels }, macros, CompileMode::Release ) );
		}
		void takeReference( const NeRF& nerf )
		{
			auto grid = dynamic_cast<const MultiResolutionHashEncoder*>( nerf.m_densityLayers[0].get() );
			for( int level = 0; level < grid->m_config.L; ++level )
			{
				const Mat& feature = grid->m_features[level];

				int baseLevel = grid->m_config.T * grid->m_config.F * sizeof( float ) * level;
				for( int fi = 0; fi < grid->m_config.F; ++fi )
				{
					int bytesPerFeature = grid->m_config.T * sizeof( float );
					oroMemcpyHtoD(
						( oroDeviceptr )( m_matBuffer->data() + m_gridFeatureLocation * sizeof( float ) + baseLevel + bytesPerFeature * fi ),
						(void*)&feature( fi, 0 ), bytesPerFeature );
				}
			}

			{
				std::vector<const AffineLayer*> affineLayers;
				for( int i = 1; i < nerf.m_densityLayers.size(); i += 2 )
				{
					auto affineLayer = dynamic_cast<const AffineLayer*>( nerf.m_densityLayers[i].get() );
					affineLayers.push_back( affineLayer );
				}
				for( int i = 0; i < nerf.m_colorLayers.size(); i += 2 )
				{
					auto affineLayer = dynamic_cast<const AffineLayer*>( nerf.m_colorLayers[i].get() );
					affineLayers.push_back( affineLayer );
				}
				PR_ASSERT( affineLayers.size() == NERF_COLOR_LAYER_END );
				for( int i = 0; i < affineLayers.size(); ++i )
				{
					oroMemcpyHtoD( ( oroDeviceptr )( m_matBuffer->data() + m_Ws[i].m_location * sizeof( float ) ), (void*)affineLayers[i]->m_W.data(), affineLayers[i]->m_W.bytes() );
					oroMemcpyHtoD( ( oroDeviceptr )( m_matBuffer->data() + m_Bs[i].m_location * sizeof( float ) ), (void*)affineLayers[i]->m_b.data(), affineLayers[i]->m_b.bytes() );
				}
			}
			//{
			//	std::vector<const AffineLayer*> affineLayers;
			//	for( int i = 0; i < nerf.m_colorLayers.size(); i += 2 )
			//	{
			//		auto affineLayer = dynamic_cast<const AffineLayer*>( nerf.m_colorLayers[i].get() );
			//		affineLayers.push_back( affineLayer );
			//	}
			//	PR_ASSERT( affineLayers.size() == NERF_COLOR_LAYER_END - NERF_COLOR_LAYER_BEG );
			//	for( int i = 0; i < affineLayers.size(); ++i )
			//	{
			//		oroMemcpyHtoD( ( oroDeviceptr )( m_matBuffer->data() + m_Ws[NERF_COLOR_LAYER_BEG + i].m_location * sizeof( float ) ), (void*)affineLayers[i]->m_W.data(), affineLayers[i]->m_W.bytes() );
			//		oroMemcpyHtoD( ( oroDeviceptr )( m_matBuffer->data() + m_Bs[NERF_COLOR_LAYER_BEG + i].m_location * sizeof( float ) ), (void*)affineLayers[i]->m_b.data(), affineLayers[i]->m_b.bytes() );
			//	}
			//}
		}
		float train( const NeRFInput* inputs, const NeRFOutput* outputs, int nElement, oroStream stream )
		{
			int blockSize = 1024;
			int nIter = div_round_up( nElement, blockSize );
			for( int iBlock = 0; iBlock < nIter; iBlock++ )
			{
				int element_beg = iBlock * blockSize;
				int element_end = std::min( ( iBlock + 1 ) * blockSize, nElement );
				int nItems = element_end - element_beg;

				if( !m_nerfInputBuffer || m_nerfInputBuffer->bytes() < nItems * sizeof( NeRFInput ) )
				{
					m_nerfInputBuffer = std::unique_ptr<Buffer>( new Buffer( nItems * sizeof( NeRFInput ) ) );
				}
				oroMemcpyHtoDAsync( (oroDeviceptr)m_nerfInputBuffer->data(), (void*)( inputs + element_beg ), nItems * sizeof( NeRFInput ), stream );
				if( !m_nerfOutputBuffer || m_nerfOutputBuffer->bytes() < nItems * sizeof( NeRFOutput ) )
				{
					m_nerfOutputBuffer = std::unique_ptr<Buffer>( new Buffer( nItems * sizeof( NeRFOutput ) ) );
				}
				oroMemcpyHtoDAsync( (oroDeviceptr)m_nerfOutputBuffer->data(), (void*)( outputs + element_beg ), nItems * sizeof( NeRFOutput ), stream );

				if( !m_rayBuffer || m_rayBuffer->bytes() < nItems * sizeof( NeRFRay ) )
				{
					m_rayBuffer = std::unique_ptr<Buffer>( new Buffer( nItems * sizeof( NeRFRay ) ) );
				}

				int maxRow = nItems * MLP_STEP;

				int location = 0;
				GPUMat inputGPU = allocateGPUMat( &location, maxRow, 3 );
				GPUMat outputGPU = allocateGPUMat( &location, maxRow, 4 ); // RGB + density
				GPUMat dirGPU = allocateGPUMat( &location, maxRow, 3 );

				std::vector<GPUMat> Is;
				{
					int output = multiResolutionHashOutputDim( m_hashConfig.L, m_hashConfig.F );
					GPUMat encoded = allocateGPUMat( &location, maxRow, output );
					Is.push_back( encoded );
				}
				for( int i = 0; i < m_Ws.size(); ++i )
				{
					int outputs = m_Ws[i].m_col;

					if( i + 1 == NERF_DENSITY_LAYER_END )
					{
						// don't forget.
						// concat directional encoding.
						outputs += 16; 
					}
					// printf( "outputs [%d] %d\n", i, outputs );
					GPUMat I = allocateGPUMat( &location, maxRow, outputs );
					Is.push_back( I );
				}

				if( !m_intermediateBuffer || m_intermediateBuffer->bytes() < location * sizeof( float ) )
				{
					m_intermediateBuffer = std::unique_ptr<Buffer>( new Buffer( location * sizeof( float ) ) );
				}

				inputGPU.m_row = 0;
				oroMemcpyHtoDAsync( (oroDeviceptr)m_nerfSamplesBuffer->data(), &inputGPU, sizeof( GPUMat ), stream );

				{
					ShaderArgument args;
					args.add( m_nerfInputBuffer->data() );
					args.add( m_rayBuffer->data() );
					args.add( m_intermediateBuffer->data() );
					args.add( m_nerfSamplesBuffer->data() );
					args.add( dirGPU );
					args.add( nItems );
					args.add( m_scrubmleIndexTrain++ );
					args.add( m_occupancyBuffer->data() );
					args.add( m_occupancyAvgBuffer->data() );

					int gridDim = div_round_up( nItems, 64 );
					m_forwardShader->launch( "nerfRays", args, gridDim, 1, 1, 64, 1, 1, stream );
				}

				oroMemcpyDtoH( &inputGPU, (oroDeviceptr)m_nerfSamplesBuffer->data(), sizeof( GPUMat ) ); // sync
				
				if( inputGPU.m_row == 0 )
				{
					continue;
				}
				// printf( "inputGPU.m_row %d %d\n", inputGPU.m_row, inputGPU.m_paddedRow );

				outputGPU.m_row = inputGPU.m_row;
				dirGPU.m_row = inputGPU.m_row;

				NeRFTrainArg arg;
				arg.inputMat = inputGPU;
				arg.outputMat = outputGPU;
				arg.dirMat = dirGPU;
				for( int i = 0; i < m_Ws.size(); ++i )
				{
					arg.m_Ws[i] = m_Ws[i];
					arg.m_Bs[i] = m_Bs[i];
				}
				for( int i = 0; i < m_Ws.size() + 1; ++i )
				{
					arg.m_Is[i] = Is[i];
				}
				arg.nLayer = m_Ws.size();
				arg.gridFeatureLocation = m_gridFeatureLocation;
				{
					ShaderArgument args;
					args.add( m_intermediateBuffer->data() );
					args.add( m_matBuffer->data() );
					args.add( arg );

					int gridDim = div_round_up( inputGPU.m_row, SHARED_TENSOR_ROW );
					m_forwardShader->launch( "trainNerfForward", args, gridDim, 1, 1, 1, 64, 1, stream );
				}
				{
					ShaderArgument args;
					args.add( m_rayBuffer->data() );
					args.add( m_nerfOutputBuffer->data() );
					args.add( m_intermediateBuffer->data() );
					args.add( outputGPU );
					args.add( nItems );

					int gridDim = div_round_up( nItems, 64 );
					m_forwardShader->launch( "nerfDerivative", args, gridDim, 1, 1, 64, 1, 1, stream );
				}
				{
					ShaderArgument args;
					args.add( m_intermediateBuffer->data() );
					args.add( m_matBuffer->data() );
					args.add( m_dmatBuffer->data() );
					args.add( arg );

					int gridDim = div_round_up( inputGPU.m_row, SHARED_TENSOR_ROW );
					m_forwardShader->launch( "trainNerfBackward", args, gridDim, 1, 1, 1, 64, 1, stream );
				}
			}

			m_iteration++;

			{
				float beta1t = pow( ADAM_BETA1, m_iteration );
				float beta2t = pow( ADAM_BETA2, m_iteration );
				int nAdam = m_adamBuffer->bytes() / sizeof( Adam );
				// printf( "beta1t, beta2t %f, %f\n", beta1t, beta2t );

				ShaderArgument adamArgs;
				adamArgs.add( m_matBuffer->data() );
				adamArgs.add( m_dmatBuffer->data() );
				adamArgs.add( m_adamBuffer->data() );
				adamArgs.add( 16.0f / nElement );
				adamArgs.add( beta1t );
				adamArgs.add( beta2t );
				adamArgs.add( nAdam );

				m_forwardShader->launch( "adamOptimize", adamArgs, div_round_up( nAdam, 64 ), 1, 1, 64, 1, 1, stream );
			}

			if( ( m_iteration % 16 ) == 0 )
			{
				if( m_iteration == 16 )
				{
					oroMemsetD32( (oroDeviceptr)m_occupancyBuffer->data(), as_int32( 0.0f ), m_occupancyBuffer->bytes() / sizeof( int ) );
				}
				else
				{
					ShaderArgument dargs;
					dargs.add( m_occupancyBuffer->data() );
					m_forwardShader->launch( "nerfDecayOccupancy", dargs, div_round_up( NERF_OCCUPANCY_GRID_T, 64 ), 1, 1, 64, 1, 1, stream );
				}
				

				NeRFForwardArg arg;
				for( int i = 0; i < m_Ws.size(); ++i )
				{
					arg.m_Ws[i] = m_Ws[i];
					arg.m_Bs[i] = m_Bs[i];
				}
				arg.gridFeatureLocation = m_gridFeatureLocation;

				{
					ShaderArgument args;
					args.add( m_matBuffer->data() );
					args.add( arg );
					args.add( m_occupancyBuffer->data() );
					args.add( m_iteration / 16 );
					int gridDim = div_round_up( NERF_OCCUPANCY_GRID_MIN_RES * NERF_OCCUPANCY_GRID_MIN_RES * NERF_OCCUPANCY_GRID_MIN_RES, SHARED_TENSOR_ROW );
					m_forwardShader->launch( "nerfUpdateOccupancy", args, gridDim, 1, 1, 1, 64, 1, stream );
				}

				oroMemsetD32( (oroDeviceptr)m_occupancyAvgBuffer->data(), 0, 1 );

				{
					ShaderArgument args;
					args.add( m_occupancyBuffer->data() );
					args.add( m_occupancyAvgBuffer->data() );
					int gridDim = div_round_up( NERF_OCCUPANCY_GRID_MIN_RES * NERF_OCCUPANCY_GRID_MIN_RES * NERF_OCCUPANCY_GRID_MIN_RES, 32 * NERF_AVG_BATCH );
					m_forwardShader->launch( "avg", args, gridDim, 1, 1, 32, 1, 1, stream );
				}

				//float avg = 0.0f;
				//oroMemcpyDtoH( &avg, (oroDeviceptr)m_occupancyAvgBuffer->data(), 4 );
				//printf( " --avg: %f\n", avg );
			}

			oroError e = oroStreamSynchronize( stream );
			return 0.0f;
		}
		void forward( const NeRFInput* inputs, NeRFOutput* outputs, int nElement, oroStream stream )
		{
			int blockSize = 16384;
			int nIter = div_round_up( nElement, blockSize );
			for( int iBlock = 0; iBlock < nIter; iBlock++ )
			{
				int element_beg = iBlock * blockSize;
				int element_end = std::min( ( iBlock + 1 ) * blockSize, nElement );
				int nItems = element_end - element_beg;

				if( !m_nerfInputBuffer || m_nerfInputBuffer->bytes() < nItems * sizeof( NeRFInput ) )
				{
					m_nerfInputBuffer = std::unique_ptr<Buffer>( new Buffer( nItems * sizeof( NeRFInput ) ) );
				}
				oroMemcpyHtoDAsync( (oroDeviceptr)m_nerfInputBuffer->data(), (void*)( inputs + element_beg ), nItems * sizeof( NeRFInput ), stream );
				if( !m_nerfOutputBuffer || m_nerfOutputBuffer->bytes() < nItems * sizeof( NeRFOutput ) )
				{
					m_nerfOutputBuffer = std::unique_ptr<Buffer>( new Buffer( nItems * sizeof( NeRFOutput ) ) );
				}
			
				if( !m_rayBuffer || m_rayBuffer->bytes() < nItems * sizeof( NeRFRay ) )
				{
					m_rayBuffer = std::unique_ptr<Buffer>( new Buffer( nItems * sizeof( NeRFRay ) ) );
				}

				int maxRow = nItems * MLP_STEP;
				int location = 0;
				GPUMat inputGPU  = allocateGPUMat( &location, maxRow, 3 );
				GPUMat outputGPU = allocateGPUMat( &location, maxRow, 4 ); // RGB + density
				GPUMat dirGPU    = allocateGPUMat( &location, maxRow, 3 );
				if( !m_intermediateBuffer || m_intermediateBuffer->bytes() < location * sizeof( float ) )
				{
					m_intermediateBuffer = std::unique_ptr<Buffer>( new Buffer( location * sizeof( float ) ) );
				}
				//printf( "location * sizeof( float ) %d\n", location * sizeof( float ) );

				inputGPU.m_row = 0;
				oroMemcpyHtoDAsync( (oroDeviceptr)m_nerfSamplesBuffer->data(), &inputGPU, sizeof( GPUMat ), stream );			
				{
					ShaderArgument args;
					args.add( m_nerfInputBuffer->data() );
					args.add( m_rayBuffer->data() );
					args.add( m_intermediateBuffer->data() );
					args.add( m_nerfSamplesBuffer->data() );
					args.add( dirGPU );
					args.add( nItems );
					args.add( 0 );
					args.add( m_occupancyBuffer->data() );
					args.add( m_occupancyAvgBuffer->data() );

					int gridDim = div_round_up( nItems, 64 );
					m_forwardShader->launch( "nerfRays", args, gridDim, 1, 1, 64, 1, 1, stream );
				}

				oroMemcpyDtoH( &inputGPU, ( oroDeviceptr ) m_nerfSamplesBuffer->data(), sizeof( GPUMat ) ); // sync
				outputGPU.m_row = inputGPU.m_row;
				dirGPU.m_row = inputGPU.m_row;

				if( inputGPU.m_row == 0 )
				{
					NeRFOutput black = {};
					std::fill( outputs + element_beg, outputs + element_end, black );
					continue;
				}

				{
					NeRFForwardArg arg;
					arg.inputMat = inputGPU;
					arg.outputMat = outputGPU;
					arg.dirMat = dirGPU;
					for( int i = 0; i < m_Ws.size(); ++i )
					{
						arg.m_Ws[i] = m_Ws[i];
						arg.m_Bs[i] = m_Bs[i];
					}
					arg.gridFeatureLocation = m_gridFeatureLocation;

					ShaderArgument args;
					args.add( m_intermediateBuffer->data() );
					args.add( m_matBuffer->data() );
					args.add( arg );

					int gridDim = div_round_up( inputGPU.m_row, SHARED_TENSOR_ROW );
					m_forwardShader->launch( "nerfForward", args, gridDim, 1, 1, 1, 64, 1, stream );
				}

				{
					ShaderArgument args;
					args.add( m_rayBuffer->data() );
					args.add( m_nerfOutputBuffer->data() );
					args.add( m_intermediateBuffer->data() );
					args.add( outputGPU );
					args.add( nItems );
					// printf( "r %d\n", outputGPU.m_row );

					int gridDim = div_round_up( nItems, 64 );
					m_forwardShader->launch( "nerfEval", args, gridDim, 1, 1, 64, 1, 1, stream );
				}

				oroMemcpyDtoHAsync( outputs + element_beg, ( oroDeviceptr )m_nerfOutputBuffer->data(), nItems * sizeof( NeRFOutput ), stream );
			}

			oroStreamSynchronize( stream );

		}
		MultiResolutionHashEncoder::Config m_hashConfig;
		std::unique_ptr<Shader> m_forwardShader;

		//std::unique_ptr<Buffer> m_inputBuffer;
		//std::unique_ptr<Buffer> m_outputBuffer;
		std::unique_ptr<Buffer> m_matBuffer;
		std::vector<GPUMat> m_Ws;
		std::vector<GPUMat> m_Bs;
		int m_gridFeatureLocation = 0;

		std::unique_ptr<Buffer> m_nerfInputBuffer;
		std::unique_ptr<Buffer> m_nerfOutputBuffer;
		std::unique_ptr<Buffer> m_rayBuffer;
		std::unique_ptr<Buffer> m_nerfSamplesBuffer;

		std::unique_ptr<Buffer> m_occupancyBuffer;
		std::unique_ptr<Buffer> m_occupancyAvgBuffer;

		// learning
		std::unique_ptr<Buffer> m_intermediateBuffer;
		std::unique_ptr<Buffer> m_dmatBuffer;
		std::unique_ptr<Buffer> m_adamBuffer;

		int m_iteration = 0;
		int m_scrubmleIndexTrain = 0;
	};
}