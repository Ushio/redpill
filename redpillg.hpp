#pragma once

#include "redpill.hpp"
#include <Orochi/Orochi.h>

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

	// assume Relu
	class MLP_GPU_Forward
	{
	public:
		MLP_GPU_Forward( const MLP& mlp, std::string kernels )
		{
			bool hasEncoder = dynamic_cast<const AffineLayer*>( mlp.m_layers[0].get() ) == 0;
			std::vector<std::string> macros;
			if( hasEncoder )
			{
				auto f = dynamic_cast<const FrequencyEncoder*>( mlp.m_layers[0].get() );
				if( f )
				{
					m_frequency = f;
				}

				auto g = dynamic_cast<const MultiResolutionHashEncoder*>( mlp.m_layers[0].get() );
				if( g )
				{
					m_grid = g;

					int bytesPerTable = g->m_config.T * g->m_config.F * sizeof( float );
					m_gridBuffer = std::unique_ptr<Buffer>( new Buffer( bytesPerTable * g->m_config.L ) );

					macros.push_back( "-DGRID_INPUT_DIM=" + std::to_string( mlp.m_layers[0]->inputDimensions() ) );
					macros.push_back( "-DGRID_L=" + std::to_string( g->m_config.L ) );
					macros.push_back( "-DGRID_T=" + std::to_string( g->m_config.T ) );
					macros.push_back( "-DGRID_F=" + std::to_string( g->m_config.F ) );
					macros.push_back( "-DGRID_NMIN=(float)(" + std::to_string( g->m_config.Nmin ) + ")" );
					macros.push_back( "-DGRID_B=(float)(" + std::to_string( g->m_config.b ) + ")" );
				}
			}

			int maxCol = 0;
			int location = 0;
			for( int i = ( hasEncoder ? 1 : 0 ); i < mlp.m_layers.size() ; i += 2 )
			{
				auto affineLayer = dynamic_cast<const AffineLayer*>( mlp.m_layers[i].get() );

				GPUMat w;
				w.m_location = location;
				w.m_row = affineLayer->m_W.row();
				w.m_paddedRow = affineLayer->m_W.paddedRow();
				w.m_col = affineLayer->m_W.col();
				location += w.m_paddedRow * w.m_col;

				GPUMat b;
				b.m_location = location;
				b.m_row = affineLayer->m_b.row();
				b.m_paddedRow = affineLayer->m_b.paddedRow();
				b.m_col = affineLayer->m_b.col();
				location += b.m_paddedRow * b.m_col;
				
				m_Ws.push_back( w );
				m_Bs.push_back( b );
				m_affineLayers.push_back( affineLayer );

				maxCol = maxss( maxCol, w.m_col );
			}

			m_maxCol = maxCol;

			m_matBuffer = std::unique_ptr<Buffer>( new Buffer( location * sizeof( float ) ) );

			m_forwardShader = std::unique_ptr<Shader>( new Shader( ( kernels + "\\mlpForward.cu" ).c_str(), "mlpForward.cu", { kernels }, macros, CompileMode::Release ) );
		}
		void foward( oroStream stream, const Mat& input, Mat* output )
		{
			for( int i = 0; i < m_affineLayers.size(); ++i )
			{
				oroMemcpyHtoDAsync( ( oroDeviceptr )( m_matBuffer->data() + m_Ws[i].m_location * sizeof( float ) ), (void*)m_affineLayers[i]->m_W.data(), m_affineLayers[i]->m_W.bytes(), stream );
				oroMemcpyHtoDAsync( ( oroDeviceptr )( m_matBuffer->data() + m_Bs[i].m_location * sizeof( float ) ), (void*)m_affineLayers[i]->m_b.data(), m_affineLayers[i]->m_b.bytes(), stream );
			}
			if( m_grid )
			{
				for( int level = 0; level < m_grid->m_config.L; ++level )
				{
					const Mat& feature = m_grid->m_features[level];

					int baseLevel = m_grid->m_config.T * m_grid->m_config.F * sizeof( float ) * level;
					for( int fi = 0; fi < m_grid->m_config.F; ++fi )
					{
						int bytesPerFeature = m_grid->m_config.T * sizeof( float );
						oroMemcpyHtoDAsync( ( oroDeviceptr )( m_gridBuffer->data() + baseLevel + bytesPerFeature * fi ), (void*)&feature( fi, 0 ), bytesPerFeature, stream );
					}
				}
			}

			int row = input.row();
			int paddedRow = input.paddedRow();

			GPUMat inputGPU;
			inputGPU.m_location = 0;
			inputGPU.m_row = row;
			inputGPU.m_paddedRow = paddedRow;
			inputGPU.m_col = input.col();

			if( !m_inputBuffer || m_inputBuffer->bytes() < input.bytes() )
			{
				m_inputBuffer = std::unique_ptr<Buffer>( new Buffer( input.bytes() ) );
			}

			oroMemcpyHtoDAsync( (oroDeviceptr)m_inputBuffer->data(), (void*)input.data(), input.bytes(), stream );

			GPUMat outputGPU;
			outputGPU.m_location = 0;
			outputGPU.m_row = row;
			outputGPU.m_paddedRow = paddedRow;
			outputGPU.m_col = m_affineLayers[m_affineLayers.size() - 1]->m_W.col();
			int64_t outputGPUBytes = (int64_t)outputGPU.m_paddedRow * outputGPU.m_col * sizeof(float);
			if( !m_outputBuffer || m_outputBuffer->bytes() < outputGPUBytes )
			{
				m_outputBuffer = std::unique_ptr<Buffer>( new Buffer( outputGPUBytes ) );
			}

			MLPForwardFusedArg arg;
			arg.inputMat = inputGPU;
			arg.outputMat = outputGPU;
			for( int i = 0; i < m_affineLayers.size(); ++i )
			{
				arg.m_Ws[i] = m_Ws[i];
				arg.m_Bs[i] = m_Bs[i];
			}
			arg.nLayer = m_affineLayers.size();

			MLPEncodingArg encoding = {};
			if( m_frequency )
			{
				encoding.mode = 1;
				encoding.frequency_N = m_frequency->m_config.N;
			}
			if( m_grid )
			{
				encoding.mode = 2;
			}

			ShaderArgument args;
			args.add( m_inputBuffer->data() );
			args.add( m_outputBuffer->data() );
			args.add( m_matBuffer->data() );
			args.add( m_gridBuffer ? m_gridBuffer->data() : nullptr );
			args.add( arg );
			args.add( encoding );

			int numberOfGrid = div_round_up( row, SHARED_TENSOR_ROW );
			m_forwardShader->launch( "forward", args, numberOfGrid, 1, 1, 1, 64, 1, stream );

			output->setShape( outputGPU.m_row, outputGPU.m_col );

			oroMemcpyDtoHAsync( output->data(), (oroDeviceptr)m_outputBuffer->data(), output->bytes(), stream );

			oroStreamSynchronize( stream );
		}

		int m_maxCol = 0;
		std::unique_ptr<Buffer> m_inputBuffer;
		std::unique_ptr<Buffer> m_outputBuffer;
		std::unique_ptr<Buffer> m_matBuffer;
		std::vector<char> m_matBufferSrc;
		std::vector<GPUMat> m_Ws;
		std::vector<GPUMat> m_Bs;
		
		const FrequencyEncoder* m_frequency = 0;
		const MultiResolutionHashEncoder* m_grid = 0;
		std::vector<const AffineLayer*> m_affineLayers;

		std::unique_ptr<Buffer> m_gridBuffer;
		std::unique_ptr<Shader> m_forwardShader;
	};
}