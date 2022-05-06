#pragma once

#include "redpill.hpp"
#include "sdx.hpp"

namespace rpml
{
	struct GPUMat
	{
		int m_row; // = inputs
		int m_paddedRow;
		int m_col; // = outputs
		int m_location; // float location
	};
	struct MLPForwardArg
	{
		GPUMat inputMat;
		GPUMat outputMat;
		GPUMat m_W;
		GPUMat m_B;
		int activation;
		int padd0;
		int padd1;
		int padd2;
	};
	struct MLPForwardFusedArg
	{
		GPUMat inputMat;
		GPUMat outputMat;
		GPUMat m_Ws[16];
		GPUMat m_Bs[16];
		int nLayer;
		int nBlock;
		int padd1;
		int padd2;
	};
	struct MLPEncoding
	{
		int mode;
		int frequency_N;
		int grid_L;
		int grid_T;

		int grid_F;
		int grid_Nmin;
		float grid_b;
		int padd1;
	};
	#define FUSED 1

	// assume Relu
	class MLP_GPU_Forward
	{
	public:
		MLP_GPU_Forward( dx::Device *device, const MLP& mlp, std::string kernels )
		{
			bool hasEncoder = dynamic_cast<const AffineLayer*>( mlp.m_layers[0].get() ) == 0;

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
					m_gridBuffer = std::unique_ptr<dx::Buffer>( new dx::Buffer( device, bytesPerTable * g->m_config.L, "Grid Buffer" ) );
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

			m_matBuffer = std::unique_ptr<dx::Buffer>( new dx::Buffer( device, location * sizeof( float ), "Layer Mat" ) );

			#if FUSED
			m_forwardShader = std::unique_ptr<dx::Shader>( new dx::Shader( device, ( kernels + "\\mlpForwardFused.hlsl" ).c_str(), kernels.c_str(), dx::CompileMode::Release ) );
			#else
			m_forwardShader = std::unique_ptr<dx::Shader>( new dx::Shader( device, ( kernels + "\\mlpForward.hlsl" ).c_str(), kernels.c_str(), dx::CompileMode::Release ) );
			#endif
			m_arg = std::unique_ptr<dx::Shader::Argument>( m_forwardShader->newArgument( device ) );

			m_stopwatch = std::unique_ptr<dx::DeviceStopwatch>( new dx::DeviceStopwatch( device, 1024, true ) );
		}
		void foward( dx::Device* device, const Mat& input, Mat *output )
		{
			for( int i = 0; i < m_affineLayers.size(); ++i )
			{
				device->copyH2D( m_matBuffer.get(), m_affineLayers[i]->m_W.data(), m_Ws[i].m_location * sizeof( float ), m_affineLayers[i]->m_W.bytes(), dx::Device::CopyMode::PrefferedEnqueue );
				device->copyH2D( m_matBuffer.get(), m_affineLayers[i]->m_b.data(), m_Bs[i].m_location * sizeof( float ), m_affineLayers[i]->m_b.bytes(), dx::Device::CopyMode::PrefferedEnqueue );
			}
			if( m_grid )
			{
				for( int li = 0; li < m_grid->m_config.L; ++li )
				{
					const Mat &feature = m_grid->m_features[li];
					int baseLevel = m_grid->m_config.T * m_grid->m_config.F * sizeof( float ) * li;
					for( int fi = 0; fi < m_grid->m_config.F; ++fi )
					{
						int bytesPerFeature = m_grid->m_config.T * sizeof( float );
						device->copyH2D( m_gridBuffer.get(), &feature( fi, 0 ), baseLevel + bytesPerFeature * fi, bytesPerFeature, dx::Device::CopyMode::PrefferedEnqueue );
					}
				}
			}
#if FUSED
			int row = input.row();
			int paddedRow = input.paddedRow();

			GPUMat inputGPU;
			inputGPU.m_location = 0;
			inputGPU.m_row = row;
			inputGPU.m_paddedRow = paddedRow;
			inputGPU.m_col = input.col();

			if( !m_inputBuffer || m_inputBuffer->bytes() < input.bytes() )
			{
				m_inputBuffer = std::unique_ptr<dx::Buffer>( new dx::Buffer( device, input.bytes(), "Input Mat" ) );
			}
			
			device->copyH2D( m_inputBuffer.get(), input.data(), 0, input.bytes(), dx::Device::CopyMode::PrefferedEnqueue );

			GPUMat outputGPU;
			outputGPU.m_location = 0;
			outputGPU.m_row = row;
			outputGPU.m_paddedRow = paddedRow;
			outputGPU.m_col = m_affineLayers[m_affineLayers.size() - 1]->m_W.col();
			int64_t outputGPUBytes = (int64_t)outputGPU.m_paddedRow * outputGPU.m_col * sizeof(float);
			if( !m_outputBuffer || m_outputBuffer->bytes() < outputGPUBytes )
			{
				m_outputBuffer = std::unique_ptr<dx::Buffer>( new dx::Buffer( device, outputGPUBytes, "Output Mat" ) );
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

			// workaround for D3D12_CS_DISPATCH_MAX_THREAD_GROUPS_PER_DIMENSION
#define DISPATCH_CHUNK 8096
			//int numberOfBlock = div_round_up( row, 8 );
			//int numberOfChunk = div_round_up( numberOfBlock, DISPATCH_CHUNK );
			//arg.nBlock = numberOfBlock;

#define TENSOR_ROW 16
			int numberOfBlock = div_round_up( row, TENSOR_ROW );
			int numberOfChunk = div_round_up( numberOfBlock, DISPATCH_CHUNK );
			arg.nBlock = numberOfBlock;

			m_arg->Constant( "mlpForwardFusedArg", arg );
			m_arg->RWStructured( "inputs", m_inputBuffer.get() );
			m_arg->RWStructured( "outputs", m_outputBuffer.get() );
			m_arg->RWStructured( "matBuffer", m_matBuffer.get() );

			MLPEncoding encoding = {};
			if( m_frequency )
			{
				encoding.mode = 1;
				encoding.frequency_N = m_frequency->m_config.N;
			}
			if( m_grid )
			{
				encoding.mode = 2;
				encoding.grid_L = m_grid->m_config.L;
				encoding.grid_T = m_grid->m_config.T;
				encoding.grid_F = m_grid->m_config.F;
				encoding.grid_Nmin = m_grid->m_config.Nmin;
				encoding.grid_b = m_grid->m_config.b;
				m_arg->RWStructured( "gridFeature", m_gridBuffer.get() );
			}
			m_arg->Constant( "mlpEncoding", encoding );

			m_stopwatch->begin( "forwardShader" );

			// m_forwardShader->dispatchAsync( device, m_arg.get(), 1, div_round_up( row, 8 ), 1 );
			
			// m_forwardShader->dispatchAsync( device, m_arg.get(), 1, DISPATCH_CHUNK, numberOfChunk );
			
			// m_forwardShader->dispatchAsync( device, m_arg.get(), 1, DISPATCH_CHUNK, div_round_up( row, DISPATCH_CHUNK ) );

			m_forwardShader->dispatchAsync( device, m_arg.get(), 1, DISPATCH_CHUNK, numberOfChunk );

			m_stopwatch->end();

			output->setShape( outputGPU.m_row, outputGPU.m_col );

			device->copyD2H( output->data(), m_outputBuffer.get(), 0, output->bytes() );

			m_stopwatch->collect();
			// printf( "forwardShader: %.5f ms\n", m_stopwatch->ms( "forwardShader" ) );
#else
			int row = input.row();
			int paddedRow = input.paddedRow();
			int64_t maxMatBytes = (int64_t)paddedRow * m_maxCol * sizeof( float );

			if( !m_inputBuffer || m_inputBuffer->bytes() < maxMatBytes )
			{
				m_inputBuffer = std::unique_ptr<dx::Buffer>( new dx::Buffer( device, maxMatBytes, "Input Mat" ) );
			}
			if( !m_outputBuffer || m_outputBuffer->bytes() < maxMatBytes )
			{
				m_outputBuffer = std::unique_ptr<dx::Buffer>( new dx::Buffer( device, maxMatBytes, "Output Mat" ) );
			}

			GPUMat inputGPU;
			inputGPU.m_location = 0;
			inputGPU.m_row = row;
			inputGPU.m_paddedRow = paddedRow;
			inputGPU.m_col = input.col();
			device->copyH2D( m_inputBuffer.get(), input.data(), 0, input.bytes() );

			GPUMat outputGPU = {};

			for( int i = 0; i < m_affineLayers.size(); ++i )
			{
				outputGPU.m_location = 0;
				outputGPU.m_row = row;
				outputGPU.m_paddedRow = paddedRow;
				outputGPU.m_col = m_affineLayers[i]->m_W.col();

				MLPForwardArg arg;
				arg.inputMat = inputGPU;
				arg.outputMat = outputGPU;
				arg.m_W = m_Ws[i];
				arg.m_B = m_Bs[i];

				bool isLast = i == m_affineLayers.size() - 1;
				arg.activation = isLast ? 0 : 1;

				m_arg->Constant( "mlpForwardArg", arg );
				m_arg->RWStructured( "inputs", m_inputBuffer.get() );
				m_arg->RWStructured( "outputs", m_outputBuffer.get() );
				m_arg->RWStructured( "matBuffer", m_matBuffer.get() );

				m_forwardShader->dispatchAsync( device, m_arg.get(), outputGPU.m_col, div_round_up( row, 64 ), 1 );
				device->wait();

				std::swap( inputGPU, outputGPU );
				std::swap( m_inputBuffer, m_outputBuffer );
			}

			output->setShape( inputGPU.m_row, inputGPU.m_col );
			device->copyD2H( output->data(), m_inputBuffer.get(), 0, output->bytes() );
#endif
			device->present();
		}

		int m_maxCol = 0;
		std::unique_ptr<dx::Buffer> m_inputBuffer;
		std::unique_ptr<dx::Buffer> m_outputBuffer;
		std::unique_ptr<dx::Buffer> m_matBuffer;
		std::vector<char> m_matBufferSrc;
		std::vector<GPUMat> m_Ws;
		std::vector<GPUMat> m_Bs;
		
		const FrequencyEncoder* m_frequency = 0;
		const MultiResolutionHashEncoder* m_grid = 0;
		std::vector<const AffineLayer*> m_affineLayers;

		std::unique_ptr<dx::Buffer> m_gridBuffer;

		std::unique_ptr<dx::Shader> m_forwardShader;
		std::unique_ptr<dx::Shader::Argument> m_arg;

		std::unique_ptr<dx::DeviceStopwatch> m_stopwatch;
	};
}