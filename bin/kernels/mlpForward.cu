#include "redpill_common.hpp"

#ifndef GRID_L
    #define GRID_INPUT_DIM 1
    #define GRID_L 16
    #define GRID_T (1 << 19)
    #define GRID_F 3
    #define GRID_NMIN 8
    #define GRID_B 2.0f
#endif

#ifndef FREQ_N 
#define FREQ_N 2
#endif

DEVICE
float getTensor( float* tensor, int xi, int yi )
{
    return tensor[xi * SHARED_TENSOR_ROW + yi];
}
DEVICE
void setTensor( float* tensor, int xi, int yi, float value )
{
    tensor[xi * SHARED_TENSOR_ROW + yi] = value;
}

using namespace rpml;

extern "C" __global__ void train( float* matBuffer, float* dMatBuffer, float* intermediates, MLPTrainArg arg ) 
{
    __shared__ float tensor[64 * SHARED_TENSOR_ROW];

    int yi_global_base = blockIdx.x * SHARED_TENSOR_ROW;
    int xi = threadIdx.y;

    float value[SHARED_TENSOR_ROW];
    
    // Load Input

    if( xi < arg.inputMat.m_col )
    {
        float vs[SHARED_TENSOR_ROW];
        for( int yi_local = 0 ; yi_local < SHARED_TENSOR_ROW ; yi_local++ )
        {
            int yi = yi_global_base + yi_local;
            if( yi < arg.inputMat.m_row )
            {
                vs[yi_local] = intermediates[elem( xi, yi, arg.inputMat)];
            }
            else
            {
                vs[yi_local] = 0.0f;
            }
        }
        for( int yi_local = 0 ; yi_local < SHARED_TENSOR_ROW ; yi_local++ )
        {
            setTensor( tensor, xi, yi_local, vs[yi_local] );
        }
    }
    __syncthreads();

    // Encording

    if( arg.encoder == EncoderType::None )
    {
        // This case, arg.m_Is[0] == inputMat
    }
    else if( arg.encoder == EncoderType::Frequency )
    {
        Frequency frequency( arg.inputMat.m_col, xi, FREQ_N );
        
        if( xi < frequency.outputDim() )
        {
            for( int yi_local = 0 ; yi_local < SHARED_TENSOR_ROW ; yi_local++ )
            {
                float v = getTensor( tensor, frequency.dimIwant(), yi_local );
                value[yi_local] = frequency.encode( v );
            }
        }
        __syncthreads();
        if( xi < frequency.outputDim() )
        {
            for( int yi_local = 0 ; yi_local < SHARED_TENSOR_ROW ; yi_local++ )
            {
                setTensor( tensor, xi, yi_local, value[yi_local] );
                int yi = yi_global_base + yi_local;
                if( yi < arg.inputMat.m_row )
                {
                    intermediates[elem( xi, yi, arg.m_Is[0] )] = value[yi_local];
                }
            }
        }
        __syncthreads();
    }
    else if( arg.encoder == EncoderType::MultiResolutionHash )
    {
        int level = xi / GRID_F;
        int fdim  = xi % GRID_F;
        int baseLevel = GRID_T * GRID_F * level;
        float res = floor( GRID_NMIN * INTRIN_POWF( GRID_B, level ) );
        for( int yi_local = 0 ; yi_local < SHARED_TENSOR_ROW ; yi_local++ )
        {
            float input[GRID_INPUT_DIM];
            for( int x = 0 ; x < GRID_INPUT_DIM ; x++ )
            {
                input[x] = getTensor( tensor, x, yi_local );
            }
            __syncthreads();

            if( level < GRID_L )
            {
                HashGridEvaluator evaluator( GRID_INPUT_DIM );
                float feature = 0.0f;
                while( evaluator.moveNext() )
                {
                    float w;
                    uint32_t h;
                    evaluator.evaluate( &w, &h, res, input );
                    uint32_t index = h % GRID_T;
                    int address = baseLevel + GRID_T * fdim + index;
                    float f = matBuffer[arg.gridFeatureLocation + address];
                    feature += w * f;
                }
                setTensor( tensor, xi, yi_local, feature );

                int yi = yi_global_base + yi_local;
                if( yi < arg.inputMat.m_row )
                {
                    intermediates[elem( xi, yi, arg.m_Is[0] )] = value[yi_local];
                }
            }
        }
        __syncthreads();
    }

    // Forward

    for( int i = 0 ; i < arg.nLayer ; i++ )
    {
        int row = arg.m_Ws[i].m_row; // input
        int col = arg.m_Ws[i].m_col; // output

        float bias = xi < col ? matBuffer[ elem( xi, 0, arg.m_Bs[i] ) ] : 0.0f;
        for( int yi_local = 0 ; yi_local < SHARED_TENSOR_ROW ; yi_local++ )
        {
            value[yi_local] = bias;
        }
        
        if( xi < col )
        {
            for( int j = 0 ; j < row ; j++ ) 
            {
                float b = matBuffer[ elem( xi /* output xi */, j, arg.m_Ws[i] ) ];
                for( int yi_local = 0 ; yi_local < SHARED_TENSOR_ROW ; yi_local++ )
                {
                    float a = getTensor( tensor, j, yi_local );
                    value[yi_local] = fma( a, b, value[yi_local] );
                }
            }
            
            float lowerbounds = i + 1 != arg.nLayer ? 0.0f : -3.40282e+38f;
            for( int yi_local = 0 ; yi_local < SHARED_TENSOR_ROW ; yi_local++ )
            {
                value[yi_local] = fmaxf( value[yi_local], lowerbounds );
            }
        }

        __syncthreads();

        if( xi < col )
        {
            for( int yi_local = 0 ; yi_local < SHARED_TENSOR_ROW ; yi_local++ )
            {
                setTensor( tensor, xi, yi_local, value[yi_local] );
                int yi = yi_global_base + yi_local;
                if( yi < arg.inputMat.m_row )
                {
                    intermediates[elem( xi, yi, arg.m_Is[i + 1] )] = value[yi_local];
                }
            }
        }
        __syncthreads();
    }

    // dL/dx
    float refs[SHARED_TENSOR_ROW];
    for( int yi_local = 0 ; yi_local < SHARED_TENSOR_ROW ; yi_local++ )
    {
        int yi = yi_global_base + yi_local;
        if( xi < arg.inputRefMat.m_col && yi < arg.inputRefMat.m_row )
        {
            refs[yi_local] = intermediates[elem( xi, yi, arg.inputRefMat)];
        }
        else
        {
            refs[yi_local] = 0.0f;
        }
    }

    for( int yi_local = 0 ; yi_local < SHARED_TENSOR_ROW ; yi_local++ )
    {
        float x = getTensor( tensor, xi, yi_local );
        float d = x - refs[yi_local];
        int yi = yi_global_base + yi_local;
        if( arg.inputMat.m_row <= yi )
        {
            d = 0.0f;
        }
        setTensor( tensor, xi, yi_local, d );
    }

    __syncthreads();

    // backward
    for( int i = arg.nLayer - 1 ; 0 <= i ; i-- )
    {
        int row = arg.m_Ws[i].m_row; // input
        int col = arg.m_Ws[i].m_col; // output

        if( xi < col )
        {
            float dB = 0.0f;

            for( int yi_local = 0 ; yi_local < SHARED_TENSOR_ROW ; yi_local++ )
            {
                float v = getTensor( tensor, xi, yi_local );

                // ReLU derivative
                if( i != arg.nLayer - 1 )
                {
                    int yi = yi_global_base + yi_local;
                    if( yi < arg.inputMat.m_row )
                    {
                        if( intermediates[elem( xi, yi, arg.m_Is[i+1] )] == 0.0f )
                        {
                            v = 0.0f;
                        }
                    }
                }

                // bias derivative
                dB += v;

                setTensor( tensor, xi, yi_local, v );
            }
            
            if( 0.0f != dB )
            {
                atomicAdd( &dMatBuffer[ elem( xi, 0, arg.m_Bs[i] ) ], dB );
            }
        }
        
        __syncthreads();

        if( xi < col )
        {    

            // Weight derivative
            for( int yi_W = 0 ; yi_W < row ; yi_W++ )
            {
                float dW = 0.0f;
                for( int yi_local = 0 ; yi_local < SHARED_TENSOR_ROW ; yi_local++ )
                {
                    int yi = yi_global_base + yi_local;
                    if( yi < arg.inputMat.m_row )
                    {
                        float X = intermediates[elem( yi_W, yi, arg.m_Is[i] /* input Xs */ )];
                        float Y = getTensor( tensor, xi, yi_local );
                        dW = fma( X, Y, dW );
                    }
                }
                atomicAdd( &dMatBuffer[ elem( xi, yi_W, arg.m_Ws[i] ) ], dW );
            }
        }

        // X derivative
        for( int yi_local = 0 ; yi_local < SHARED_TENSOR_ROW ; yi_local++ )
        {
            value[yi_local] = 0.0f;
        }

        if( xi < row )
        {
            for( int j = 0 ; j < col ; j++ ) 
            {
                float b = matBuffer[ elem( j, xi, arg.m_Ws[i] ) ];
                for( int yi_local = 0 ; yi_local < SHARED_TENSOR_ROW ; yi_local++ )
                {
                    float a = getTensor( tensor, j, yi_local );
                    value[yi_local] = fma( a, b, value[yi_local] );
                }
            }
        }

        __syncthreads();

        if( xi < row )
        {
            for( int yi_local = 0 ; yi_local < SHARED_TENSOR_ROW ; yi_local++ )
            {
                setTensor( tensor, xi, yi_local, value[yi_local] );
            }
        }
        __syncthreads();
    }

    // encode backward
    if( arg.encoder == EncoderType::MultiResolutionHash )
    {
        int level = xi / GRID_F;
        int fdim  = xi % GRID_F;
        int baseLevel = GRID_T * GRID_F * level;
        float res = floor( GRID_NMIN * INTRIN_POWF( GRID_B, level ) );
        for( int yi_local = 0 ; yi_local < SHARED_TENSOR_ROW ; yi_local++ )
        {
            int yi = yi_global_base + yi_local;
            if( arg.inputMat.m_row <= yi )
            {
                break;
            }

            float input[GRID_INPUT_DIM];
            for( int x = 0 ; x < GRID_INPUT_DIM ; x++ )
            {
                input[x] = intermediates[elem( x, yi, arg.inputMat)];
            }

            if( level < GRID_L )
            {
                float derivative = getTensor( tensor, xi, yi_local );

                HashGridEvaluator evaluator( GRID_INPUT_DIM );
                while( evaluator.moveNext() )
                {
                    float w;
                    uint32_t h;
                    evaluator.evaluate( &w, &h, res, input );
                    uint32_t index = h % GRID_T;
                    int address = baseLevel + GRID_T * fdim + index;
                    atomicAdd( &dMatBuffer[arg.gridFeatureLocation + address], w * derivative );
                }
            }
        }
    }
}

extern "C" __global__ void adamOptimize( float* matBuffer, float* dMatBuffer, Adam* adamBuffer, float alpha, float beta1t, float beta2t, int nAdam ) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if( x < nAdam )
    {
        if( dMatBuffer[x] != 0.0f )
        {
            matBuffer[x] = adamBuffer[x].optimize( matBuffer[x], dMatBuffer[x], alpha, beta1t, beta2t );
            dMatBuffer[x] = 0.0f;
        }
    }
}

extern "C" __global__ void forward( float* inputs, float* output, float* matBuffer, MLPForwardArg arg ) 
{
    __shared__ float tensor[64 * SHARED_TENSOR_ROW];

    int yi_global_base = blockIdx.x * SHARED_TENSOR_ROW;
    int xi = threadIdx.y;

    float value[SHARED_TENSOR_ROW];
    
    if( xi < arg.inputMat.m_col )
    {
        float vs[SHARED_TENSOR_ROW];
        for( int yi_local = 0 ; yi_local < SHARED_TENSOR_ROW ; yi_local++ )
        {
            int yi = yi_global_base + yi_local;
            if( yi < arg.inputMat.m_row )
            {
                vs[yi_local] = inputs[elem( xi, yi, arg.inputMat)];
            }
            else
            {
                vs[yi_local] = 0.0f;
            }
        }
        for( int yi_local = 0 ; yi_local < SHARED_TENSOR_ROW ; yi_local++ )
        {
            setTensor( tensor, xi, yi_local, vs[yi_local] );
        }
    }
    __syncthreads();

    if( arg.encoder == EncoderType::Frequency )
    {
        Frequency frequency( arg.inputMat.m_col, xi, FREQ_N );
        
        if( xi < frequency.outputDim() )
        {
            for( int yi_local = 0 ; yi_local < SHARED_TENSOR_ROW ; yi_local++ )
            {
                float v = getTensor( tensor, frequency.dimIwant(), yi_local );
                value[yi_local] = frequency.encode( v );
            }
        }
        __syncthreads();
        if( xi < frequency.outputDim() )
        {
            for( int yi_local = 0 ; yi_local < SHARED_TENSOR_ROW ; yi_local++ )
            {
                setTensor( tensor, xi, yi_local, value[yi_local] );
            }
        }
        __syncthreads();
    }
    else if( arg.encoder == EncoderType::MultiResolutionHash )
    {
        int level = xi / GRID_F;
        int fdim  = xi % GRID_F;
        int baseLevel = GRID_T * GRID_F * level;
        float res = floor( GRID_NMIN * INTRIN_POWF( GRID_B, level ) );
        for( int yi_local = 0 ; yi_local < SHARED_TENSOR_ROW ; yi_local++ )
        {
            float input[GRID_INPUT_DIM];
            for( int x = 0 ; x < GRID_INPUT_DIM ; x++ )
            {
                input[x] = getTensor( tensor, x, yi_local );
            }
            __syncthreads();

            if( level < GRID_L )
            {
                HashGridEvaluator evaluator( GRID_INPUT_DIM );
                float feature = 0.0f;
                while( evaluator.moveNext() )
                {
                    float w;
                    uint32_t h;
                    evaluator.evaluate( &w, &h, res, input );
                    uint32_t index = h % GRID_T;
                    int address = baseLevel + GRID_T * fdim + index;
                    float f = matBuffer[arg.gridFeatureLocation + address];
                    feature += w * f;
                }
                setTensor( tensor, xi, yi_local, feature );
            }
        }
        __syncthreads();
    }
    
    for( int i = 0 ; i < arg.nLayer ; i++ )
    {
        int row = arg.m_Ws[i].m_row; // input
        int col = arg.m_Ws[i].m_col; // output

        float bias = xi < col ? matBuffer[ elem( xi, 0, arg.m_Bs[i] ) ] : 0.0f;
        for( int yi_local = 0 ; yi_local < SHARED_TENSOR_ROW ; yi_local++ )
        {
            value[yi_local] = bias;
        }
        
        if( xi < col )
        {
            for( int j = 0 ; j < row ; j++ ) 
            {
                float b = matBuffer[ elem( xi /* output xi */, j, arg.m_Ws[i] ) ];
                for( int yi_local = 0 ; yi_local < SHARED_TENSOR_ROW ; yi_local++ )
                {
                    float a = getTensor( tensor, j, yi_local );
                    value[yi_local] = fma( a, b, value[yi_local] );
                }
            }
            
            float lowerbounds = i + 1 != arg.nLayer ? 0.0f : -3.40282e+38f;
            for( int yi_local = 0 ; yi_local < SHARED_TENSOR_ROW ; yi_local++ )
            {
                value[yi_local] = fmaxf( value[yi_local], lowerbounds );
            }
        }

        __syncthreads();

        if( xi < col )
        {
            for( int yi_local = 0 ; yi_local < SHARED_TENSOR_ROW ; yi_local++ )
            {
                setTensor( tensor, xi, yi_local, value[yi_local] );
            }
        }
        __syncthreads();
    }

    if( xi < arg.outputMat.m_col )
    {
        for( int yi_local = 0 ; yi_local < SHARED_TENSOR_ROW ; yi_local++ )
        {
            value[yi_local] = getTensor( tensor, xi, yi_local );
        }
        for( int yi_local = 0 ; yi_local < SHARED_TENSOR_ROW ; yi_local++ )
        {
            int yi = yi_global_base + yi_local;
            if( yi < arg.outputMat.m_row )
            {
                output[ elem( xi, yi, arg.outputMat )] = value[yi_local];
            }
        }
    }
}

extern "C" __global__ void nerfRays( NeRFInput* inputs, NeRFRay *rays, float* intermediates, GPUMat* nerfSamples, int nElement ) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if( nElement <= x )
    {
        return;
    }

    float3 ro = float3( inputs[x].ro[0], inputs[x].ro[1], inputs[x].ro[2] );
    float3 rd = float3( inputs[x].rd[0], inputs[x].rd[1], inputs[x].rd[2] );
    float3 one_over_rd = safe_inv_rd( rd );
	float2 h = slabs( float3( 0.0f, 0.0f, 0.0f ), float3( 1.0f, 1.0f, 1.0f ), ro, rd );

    GPUMat outputMat = *nerfSamples;

	if( h.x /* min */ < h.y /* max */ )
	{
        int nSteps = 0;
        float dt = sqrt( 3.0f ) / MLP_STEP;
		
        for( int i = 0 ; i < 1024 ; ++i )
        {
            float3 p = ro + rd * ( h.x + dt * i );
            nSteps++;
            
            const float eps = 0.000001f;
            if( p.x < -eps || 1.0f + eps < p.x || p.y < -eps || 1.0f + eps < p.y || p.z < -eps || 1.0f + eps < p.z )
            {
                break;
            }
        }

        int eval_beg = atomicAdd( &nerfSamples->m_row, nSteps );
        int eval_end = eval_beg + nSteps;
        rays[x].eval_beg = eval_beg;
        rays[x].eval_end = eval_end;

        for( int i = 0 ; i < 1024 ; ++i )
        {
            float3 p = ro + rd * ( h.x + dt * i );
            
            const float eps = 0.000001f;
            if( p.x < -eps || 1.0f + eps < p.x || p.y < -eps || 1.0f + eps < p.y || p.z < -eps || 1.0f + eps < p.z )
            {
                break;
            }

            intermediates[elem( 0, eval_beg + i, outputMat )] = p.x;
            intermediates[elem( 1, eval_beg + i, outputMat )] = p.y;
            intermediates[elem( 2, eval_beg + i, outputMat )] = p.z;
        }
    }
    else
    {
        rays[x].eval_beg = 0;
        rays[x].eval_end = 0;
    }
}

extern "C" __global__ void nerfForward( float* intermediates, float* matBuffer, NeRFForwardArg arg ) 
{
    __shared__ float tensor[64 * SHARED_TENSOR_ROW];

    int yi_global_base = blockIdx.x * SHARED_TENSOR_ROW;
    int xi = threadIdx.y;

}