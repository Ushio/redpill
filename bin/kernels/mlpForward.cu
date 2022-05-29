#include "redpill_common.hpp"

#ifndef GRID_L
    #define GRID_INPUT_DIM 1
    #define GRID_L 16
    #define GRID_T (1 << 19)
    #define GRID_F 3
    #define GRID_NMIN 8
    #define GRID_B 2.0f
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

extern "C" __global__ void train_forward( float* intermediates, float* matBuffer, GPUMat inputMat, GPUMat outputMat, GPUMat W, GPUMat B, int relu )
{
    int yi = blockIdx.x;
    int xi = threadIdx.y;

    if( inputMat.m_row <= yi )
    {
        return;
    }

    int row = W.m_row; // input
    int col = W.m_col; // output

    float value = xi < col ? matBuffer[ elem( xi, 0, B ) ] : 0.0f;

    if( xi < col )
    {
        for( int j = 0 ; j < row ; j++ ) 
        {
            float b = matBuffer[ elem( xi /* output xi */, j, W ) ];
            float a = intermediates[elem( j, yi, inputMat )];
            value = fma( a, b, value );
        }
        
        float lowerbounds = relu ? 0.0f : -3.40282e+38f;
        value = fmaxf( value, lowerbounds );

        intermediates[ elem( xi, yi, outputMat ) ] = value;
    }
}
extern "C" __global__ void train( float* inputs, float* output, float* matBuffer, float* intermediates, MLPTrainArg mlpTrainArg ) 
{
    __shared__ float tensor[64 * SHARED_TENSOR_ROW];

    int yi_global_base = blockIdx.x * SHARED_TENSOR_ROW;
    int xi = threadIdx.y;

    float value[SHARED_TENSOR_ROW];
    
    if( xi < mlpTrainArg.inputMat.m_col )
    {
        float vs[SHARED_TENSOR_ROW];
        for( int yi_local = 0 ; yi_local < SHARED_TENSOR_ROW ; yi_local++ )
        {
            int yi = yi_global_base + yi_local;
            if( yi < mlpTrainArg.inputMat.m_row )
            {
                vs[yi_local] = inputs[elem( xi, yi, mlpTrainArg.inputMat)];
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

    for( int i = 0 ; i < mlpTrainArg.nLayer ; i++ )
    {
        int row = mlpTrainArg.m_Ws[i].m_row; // input
        int col = mlpTrainArg.m_Ws[i].m_col; // output

        float bias = xi < col ? matBuffer[ elem( xi, 0, mlpTrainArg.m_Bs[i] ) ] : 0.0f;
        for( int yi_local = 0 ; yi_local < SHARED_TENSOR_ROW ; yi_local++ )
        {
            value[yi_local] = bias;
        }
        
        if( xi < col )
        {
            for( int j = 0 ; j < row ; j++ ) 
            {
                float b = matBuffer[ elem( xi /* output xi */, j, mlpTrainArg.m_Ws[i] ) ];
                for( int yi_local = 0 ; yi_local < SHARED_TENSOR_ROW ; yi_local++ )
                {
                    float a = getTensor( tensor, j, yi_local );
                    value[yi_local] = fma( a, b, value[yi_local] );
                }
            }
            
            float lowerbounds = i + 1 != mlpTrainArg.nLayer ? 0.0f : -3.40282e+38f;
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
                intermediates[elem( xi, yi, mlpTrainArg.m_Is[i] )] = value[yi_local];
            }
        }
        __syncthreads();
    }

    // if( xi < mlpTrainArg.outputMat.m_col )
    // {
    //     for( int yi_local = 0 ; yi_local < SHARED_TENSOR_ROW ; yi_local++ )
    //     {
    //         value[yi_local] = getTensor( tensor, xi, yi_local );
    //     }
    //     for( int yi_local = 0 ; yi_local < SHARED_TENSOR_ROW ; yi_local++ )
    //     {
    //         int yi = yi_global_base + yi_local;
    //         if( yi < mlpTrainArg.outputMat.m_row )
    //         {
    //             output[ elem( xi, yi, mlpTrainArg.outputMat )] = value[yi_local];
    //         }
    //     }
    // }
}

extern "C" __global__ void forward( float* inputs, float* output, float* matBuffer, float* gridFeature, MLPForwardFusedArg mlpForwardFusedArg, MLPEncodingArg mlpEncoding ) 
{
    __shared__ float tensor[64 * SHARED_TENSOR_ROW];

    int yi_global_base = blockIdx.x * SHARED_TENSOR_ROW;
    int xi = threadIdx.y;

    float value[SHARED_TENSOR_ROW];
    
    if( xi < mlpForwardFusedArg.inputMat.m_col )
    {
        float vs[SHARED_TENSOR_ROW];
        for( int yi_local = 0 ; yi_local < SHARED_TENSOR_ROW ; yi_local++ )
        {
            int yi = yi_global_base + yi_local;
            if( yi < mlpForwardFusedArg.inputMat.m_row )
            {
                vs[yi_local] = inputs[elem( xi, yi, mlpForwardFusedArg.inputMat)];
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

    if( mlpEncoding.mode == 1 ) // frequency
    {
        int outputCol = mlpForwardFusedArg.inputMat.m_col * mlpEncoding.frequency_N * 2;
        
        if( xi < outputCol )
        {
            for( int yi_local = 0 ; yi_local < SHARED_TENSOR_ROW ; yi_local++ )
            {
                int xi_src = xi / ( mlpEncoding.frequency_N * 2 );
                int baseEachDim = xi % ( mlpEncoding.frequency_N * 2 );
                int i = baseEachDim / 2;
                int tri_idx = baseEachDim % 2;
                float v = getTensor( tensor, xi_src, yi_local );
                float k = 2.0f * pi * __powf( 2.0f, (float)i );
                v = __sinf( k * v + ( tri_idx ? pi * 0.5f : 0.0f ) );
                value[yi_local] = v;
            }
        }
        __syncthreads();
        if( xi < outputCol )
        {
            for( int yi_local = 0 ; yi_local < SHARED_TENSOR_ROW ; yi_local++ )
            {
                setTensor( tensor, xi, yi_local, value[yi_local] );
            }
        }
        __syncthreads();
    }
    if( mlpEncoding.mode == 2 ) // Multi Resolution Hash
    {
        int level = xi / GRID_F;
        int fdim  = xi % GRID_F;
        int baseLevel = GRID_T * GRID_F * level;
        float res = floor( GRID_NMIN * __powf( GRID_B, level ) );
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
                    float f = gridFeature[address];
                    feature += w * f;
                }
                setTensor( tensor, xi, yi_local, feature );
            }
        }
        __syncthreads();
    }
    
    for( int i = 0 ; i < mlpForwardFusedArg.nLayer ; i++ )
    {
        int row = mlpForwardFusedArg.m_Ws[i].m_row; // input
        int col = mlpForwardFusedArg.m_Ws[i].m_col; // output

        float bias = xi < col ? matBuffer[ elem( xi, 0, mlpForwardFusedArg.m_Bs[i] ) ] : 0.0f;
        for( int yi_local = 0 ; yi_local < SHARED_TENSOR_ROW ; yi_local++ )
        {
            value[yi_local] = bias;
        }
        
        if( xi < col )
        {
            for( int j = 0 ; j < row ; j++ ) 
            {
                float b = matBuffer[ elem( xi /* output xi */, j, mlpForwardFusedArg.m_Ws[i] ) ];
                for( int yi_local = 0 ; yi_local < SHARED_TENSOR_ROW ; yi_local++ )
                {
                    float a = getTensor( tensor, j, yi_local );
                    value[yi_local] = fma( a, b, value[yi_local] );
                }
            }
            
            float lowerbounds = i + 1 != mlpForwardFusedArg.nLayer ? 0.0f : -3.40282e+38f;
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

    if( xi < mlpForwardFusedArg.outputMat.m_col )
    {
        for( int yi_local = 0 ; yi_local < SHARED_TENSOR_ROW ; yi_local++ )
        {
            value[yi_local] = getTensor( tensor, xi, yi_local );
        }
        for( int yi_local = 0 ; yi_local < SHARED_TENSOR_ROW ; yi_local++ )
        {
            int yi = yi_global_base + yi_local;
            if( yi < mlpForwardFusedArg.outputMat.m_row )
            {
                output[ elem( xi, yi, mlpForwardFusedArg.outputMat )] = value[yi_local];
            }
        }
    }
}
