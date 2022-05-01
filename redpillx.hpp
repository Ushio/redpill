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
	// assume Relu
	class MLP_GPU_Forward
	{
	public:
		MLP_GPU_Forward( dx::Device *device, const MLP& mlp, std::string kernels )
		{
			bool hasEncoder = dynamic_cast<const AffineLayer*>( mlp.m_layers[0].get() ) == 0;

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

			m_forwardShader = std::unique_ptr<dx::Shader>( new dx::Shader( device, ( kernels + "\\mlpForward.hlsl" ).c_str(), kernels.c_str(), dx::CompileMode::Release ) );
			m_arg = std::unique_ptr<dx::Shader::Argument>( m_forwardShader->newArgument( device ) );
		}

		void copyH2D( dx::Device* device )
		{
			// probably copy is wrong
			for( int i = 0; i < m_affineLayers.size() ; ++i)
			{
				device->copyH2D( m_matBuffer.get(), m_affineLayers[i]->m_W.data(), m_Ws[i].m_location * sizeof( float ), m_affineLayers[i]->m_W.bytes() );
				device->copyH2D( m_matBuffer.get(), m_affineLayers[i]->m_b.data(), m_Bs[i].m_location * sizeof( float ), m_affineLayers[i]->m_b.bytes() );
			}
		}

		void foward( dx::Device* device, const Mat& input, Mat *output )
		{
			int row = input.row();
			int paddedRow = input.paddedRow();
			int maxMatBytes = paddedRow * m_maxCol * sizeof( float );

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

			device->present();
		}

		int m_maxCol = 0;
		std::unique_ptr<dx::Buffer> m_inputBuffer;
		std::unique_ptr<dx::Buffer> m_outputBuffer;
		std::unique_ptr<dx::Buffer> m_matBuffer;
		std::vector<GPUMat> m_Ws;
		std::vector<GPUMat> m_Bs;
		std::vector<const AffineLayer*> m_affineLayers;
		std::unique_ptr<dx::Shader> m_forwardShader;
		std::unique_ptr<dx::Shader::Argument> m_arg;
	};
}