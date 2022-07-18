#include "redpill_common.hpp"
#include "helper_math.h"

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

                // store intermediates
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

                // store intermediates
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
                if( dW != 0.0f )
                {
                    atomicAdd( &dMatBuffer[ elem( xi, yi_W, arg.m_Ws[i] ) ], dW );
                } 
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
                    float wd =  w * derivative;
                    if ( wd != 0.0f )
                    {
                        atomicAdd( &dMatBuffer[arg.gridFeatureLocation + address], wd );
                    }
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

extern "C" __global__ void forward( float* intermediates, float* matBuffer, MLPForwardArg arg ) 
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
                intermediates[ elem( xi, yi, arg.outputMat )] = value[yi_local];
            }
        }
    }
}

extern "C" __global__ void nerfRays( NeRFInput* inputs, NeRFRay *rays, float* intermediates, GPUMat* nerfSamples, GPUMat dirMat, int nElement, uint32_t scrubmleIndex, float* occupancyGrid, float* occupancyAverage ) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if( nElement <= x )
    {
        return;
    }

    float avg = *occupancyAverage;

    float3 ro = make_float3( inputs[x].ro[0], inputs[x].ro[1], inputs[x].ro[2] );
    float3 rd = make_float3( inputs[x].rd[0], inputs[x].rd[1], inputs[x].rd[2] );
    float3 one_over_rd = safe_inv_rd( rd );
	float2 h = slabs( make_float3( 0.0f, 0.0f, 0.0f ), make_float3( 1.0f, 1.0f, 1.0f ), ro, rd );

    GPUMat outputMat = *nerfSamples;

	if( h.x /* min */ < h.y /* max */ )
	{
        const float eps = 0.00001f;

        int nSteps = 0;
        int nEvals = 0;
        float dt = sqrt( 3.0f ) / MLP_STEP;
        // float bias = dt * 0.5f;
        float bias = dt * hash1( scrubmleIndex * 0xFFFF + x );
        for( int i = 0 ; i < MLP_STEP ; ++i )
        {
            float3 p = ro + rd * ( h.x + bias + dt * i );
            
            if( p.x < -eps || 1.0f + eps < p.x || p.y < -eps || 1.0f + eps < p.y || p.z < -eps || 1.0f + eps < p.z )
            {
                break;
            }
            p.x = fclamp( p.x, 0.0f, 1.0f );
            p.y = fclamp( p.y, 0.0f, 1.0f );
            p.z = fclamp( p.z, 0.0f, 1.0f );

            if( isOccupied( p.x, p.y, p.z, dt, occupancyGrid, avg ) )
            {
                nEvals++;
            }
            nSteps++;
        }

        int eval_beg = atomicAdd( &nerfSamples->m_row, nEvals );
        int eval_end = eval_beg + nEvals;
        rays[x].eval_beg = eval_beg;
        rays[x].eval_end = eval_end;

        int i_sample = 0;
        for( int i = 0 ; i < nSteps ; ++i )
        {
            float3 p = ro + rd * ( h.x + bias + dt * i );
            
            if( p.x < -eps || 1.0f + eps < p.x || p.y < -eps || 1.0f + eps < p.y || p.z < -eps || 1.0f + eps < p.z )
            {
                break;
            }
            p.x = fclamp( p.x, 0.0f, 1.0f );
            p.y = fclamp( p.y, 0.0f, 1.0f );
            p.z = fclamp( p.z, 0.0f, 1.0f );
            
            if( isOccupied( p.x, p.y, p.z, dt, occupancyGrid, avg ) )
            {
                intermediates[elem( 0, eval_beg + i_sample, outputMat )] = p.x;
                intermediates[elem( 1, eval_beg + i_sample, outputMat )] = p.y;
                intermediates[elem( 2, eval_beg + i_sample, outputMat )] = p.z;
                intermediates[elem( 0, eval_beg + i_sample, dirMat )] = rd.x;
                intermediates[elem( 1, eval_beg + i_sample, dirMat )] = rd.y;
                intermediates[elem( 2, eval_beg + i_sample, dirMat )] = rd.z;
                i_sample++;
            }
        }
    }
    else
    {
        rays[x].eval_beg = 0;
        rays[x].eval_end = 0;
    }
}

DEVICE_INLINE
void sh_encode( float* tensor, int yi, float x, float y, float z )
{
    constexpr float rpi = 1.7724538509055160273f; // newton_sqrt( pi );
    constexpr float r3 = 1.73205080756887729353f; // newton_sqrt( 3.0f );
    constexpr float r15 = 3.87298334620741688518f; // newton_sqrt( 15.0f );
    constexpr float r5 = 2.23606797749978969641f; // newton_sqrt( 5.0f );
    constexpr float r2 = 1.4142135623730950488f; // newton_sqrt( 2.0f );
    constexpr float r35 = 5.91607978309961604257f; // newton_sqrt( 35.0f );
    constexpr float r105 = 10.24695076595959838322f; // newton_sqrt( 105.0f );
    constexpr float r21 = 4.58257569495584000659f; // newton_sqrt( 21.0f );
    constexpr float r7 = 2.6457513110645905905f; // newton_sqrt( 7 );

    const float xy = x * y;
    const float yz = y * z;
    const float xz = x * z;
    const float xx = x * x;
    const float yy = y * y;
    const float zz = z * z;
    const float xyz = xy * z;

    int xi_base = 16;

    // L=0
    float v0 = +1.0f / ( 2.0f * rpi );

    // L=1
    float v1 = -( r3 / ( 2.0f * rpi ) ) * y;
    float v2 = +( r3 / ( 2.0f * rpi ) ) * z;
    float v3 = -( r3 / ( 2.0f * rpi ) ) * x;

    // L=2
    float v4 = +( r15 / ( 2.0f * rpi ) ) * xy;
    float v5 = -( r15 / ( 2.0f * rpi ) ) * yz;
    float v6 = +( r5 / ( 4.0f * rpi ) ) * ( 3.0f * z * z - 1.0f );
    float v7 = -( r15 / ( 2.0f * rpi ) ) * xz;
    float v8 = +( r15 / ( 4.0f * rpi ) ) * ( xx - yy );

    // L=3
    float v9 = -( r2 * r35 / ( 8.0f * rpi ) ) * y * ( 3.0f * xx - yy );
    float v10 = +( r105 / ( 2.0f * rpi ) ) * xyz;
    float v11 = -( r2 * r21 / ( 8.0f * rpi ) ) * y * ( -1.0f + 5.0f * zz );
    float v12 = +( r7 / ( 4.0f * rpi ) ) * z * ( 5.0f * z * z - 3.0f );
    float v13 = -( r2 * r21 / ( 8.0f * rpi ) ) * x * ( -1.0f + 5.0f * zz );
    float v14 = +( r105 / ( 4.0f * rpi ) ) * ( xx - yy ) * z;
    float v15 = -( r2 * r35 / ( 8.0f * rpi ) ) * x * ( xx - 3.0f * yy );

    setTensor( tensor, xi_base + 0, yi, v0 );
    setTensor( tensor, xi_base + 1, yi, v1 );
    setTensor( tensor, xi_base + 2, yi, v2 );
    setTensor( tensor, xi_base + 3, yi, v3 );
    setTensor( tensor, xi_base + 4, yi, v4 );
    setTensor( tensor, xi_base + 5, yi, v5 );
    setTensor( tensor, xi_base + 6, yi, v6 );
    setTensor( tensor, xi_base + 7, yi, v7 );
    setTensor( tensor, xi_base + 8, yi, v8 );
    setTensor( tensor, xi_base + 9, yi, v9 );
    setTensor( tensor, xi_base + 10, yi, v10 );
    setTensor( tensor, xi_base + 11, yi, v11 );
    setTensor( tensor, xi_base + 12, yi, v12 );
    setTensor( tensor, xi_base + 13, yi, v13 );
    setTensor( tensor, xi_base + 14, yi, v14 );
    setTensor( tensor, xi_base + 15, yi, v15 );
}

extern "C" __global__ void nerfForward( float* intermediates, float* matBuffer, NeRFForwardArg arg ) 
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
    
    for( int i = NERF_DENSITY_LAYER_BEG ; i < NERF_DENSITY_LAYER_END ; i++ )
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
            
            float lowerbounds = i + 1 != NERF_DENSITY_LAYER_END ? 0.0f : -3.40282e+38f;
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

        // density
        if( i + 1 == NERF_DENSITY_LAYER_END && xi == 0 )
        {
            for( int yi_local = 0 ; yi_local < SHARED_TENSOR_ROW ; yi_local++ )
            {
                float density = value[yi_local];
                int yi = yi_global_base + yi_local;
                if( yi < arg.outputMat.m_row )
                {
                    intermediates[ elem( 3, yi, arg.outputMat )] = density;
                }
            }
        }

        __syncthreads();
    }

    // // density
    // if( xi == 0 )
    // {
    //     for( int yi_local = 0 ; yi_local < SHARED_TENSOR_ROW ; yi_local++ )
    //     {
    //         float density = value[yi_local];
    //         int yi = yi_global_base + yi_local;
    //         if( yi < arg.outputMat.m_row )
    //         {
    //             intermediates[ elem( 3, yi, arg.outputMat )] = density;
    //         }
    //     }
    // }

	// directional
	int nItrEncode = div_round_up( SHARED_TENSOR_ROW, 64 );
	for( int i = 0; i < nItrEncode; i++ )
	{
		int yi_local = i * 64 + xi;
		int yi = yi_global_base + yi_local;
		if( yi_local < SHARED_TENSOR_ROW )
		{
			float x = intermediates[elem( 0, yi, arg.dirMat )];
			float y = intermediates[elem( 1, yi, arg.dirMat )];
			float z = intermediates[elem( 2, yi, arg.dirMat )];
			sh_encode( tensor, yi_local, x, y, z );
		}
	}
	__syncthreads(); 

    for( int i = NERF_COLOR_LAYER_BEG ; i < NERF_COLOR_LAYER_END ; i++ )
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
            
            float lowerbounds = i + 1 != NERF_COLOR_LAYER_END ? 0.0f : -3.40282e+38f;
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

    if( xi < arg.outputMat.m_col - 1 /* !important! don't override density */ )
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
                intermediates[ elem( xi, yi, arg.outputMat )] = value[yi_local];
            }
        }
    }
}
extern "C" __global__ void nerfDecayOccupancy( float* occupancyGrid ) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    occupancyGrid[x] *= 0.95f;
}
extern "C" __global__ void nerfUpdateOccupancy( float* matBuffer, NeRFForwardArg arg, float* occupancyGrid, int iteration ) 
{
    __shared__ float tensor[64 * SHARED_TENSOR_ROW];

    int yi_global_base = blockIdx.x * SHARED_TENSOR_ROW;
    int xi = threadIdx.y;
    float value[SHARED_TENSOR_ROW];

    if( xi < 3 )
    {
        float vs[SHARED_TENSOR_ROW];
        for( int yi_local = 0 ; yi_local < SHARED_TENSOR_ROW ; yi_local++ )
        {
            int yi = yi_global_base + yi_local;

            int hashbase = NERF_OCCUPANCY_GRID_MIN_RES * NERF_OCCUPANCY_GRID_MIN_RES * NERF_OCCUPANCY_GRID_MIN_RES * 3 * iteration;
            int index_z = yi / ( NERF_OCCUPANCY_GRID_MIN_RES * NERF_OCCUPANCY_GRID_MIN_RES ); yi = yi % ( NERF_OCCUPANCY_GRID_MIN_RES * NERF_OCCUPANCY_GRID_MIN_RES );
            int index_y = yi / NERF_OCCUPANCY_GRID_MIN_RES; yi = yi % NERF_OCCUPANCY_GRID_MIN_RES;
            int index_x = yi;
            float pz = ( index_z + hash1( hashbase + yi * 3 ) ) * ( 1.0f / (float)NERF_OCCUPANCY_GRID_MIN_RES ) ;
            float py = ( index_y + hash1( hashbase + yi * 3 + 1 ) ) * ( 1.0f / (float)NERF_OCCUPANCY_GRID_MIN_RES );
            float px = ( index_x + hash1( hashbase + yi * 3 + 2 ) ) * ( 1.0f / (float)NERF_OCCUPANCY_GRID_MIN_RES );
            float v;
            if( xi == 0 )
            {
                v = px;
            }
            else if( xi == 1 )
            {
                v = py;
            }
            else
            {
                v = pz;
            }
            vs[yi_local] = v;
        }
        for( int yi_local = 0 ; yi_local < SHARED_TENSOR_ROW ; yi_local++ )
        {
            setTensor( tensor, xi, yi_local, vs[yi_local] );
        }
    }
    __syncthreads();

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
    
    for( int i = NERF_DENSITY_LAYER_BEG ; i < NERF_DENSITY_LAYER_END ; i++ )
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
            
            float lowerbounds = i + 1 != NERF_DENSITY_LAYER_END ? 0.0f : -3.40282e+38f;
            for( int yi_local = 0 ; yi_local < SHARED_TENSOR_ROW ; yi_local++ )
            {
                value[yi_local] = fmaxf( value[yi_local], lowerbounds );
            }
        }

        __syncthreads();

        if( xi < col )
		{
			for( int yi_local = 0; yi_local < SHARED_TENSOR_ROW; yi_local++ )
			{
				setTensor( tensor, xi, yi_local, value[yi_local] );
			}
		}
    }

    if( xi == 0 )
    {
        for( int yi_local = 0 ; yi_local < SHARED_TENSOR_ROW ; yi_local++ )
        {
            float density = nerfDensityActivation( getTensor( tensor, 0, yi_local ) );

            int yi = yi_global_base + yi_local;

            int hashbase = NERF_OCCUPANCY_GRID_MIN_RES * NERF_OCCUPANCY_GRID_MIN_RES * NERF_OCCUPANCY_GRID_MIN_RES * 3 * iteration;
            int index_z = yi / ( NERF_OCCUPANCY_GRID_MIN_RES * NERF_OCCUPANCY_GRID_MIN_RES ); yi = yi % ( NERF_OCCUPANCY_GRID_MIN_RES * NERF_OCCUPANCY_GRID_MIN_RES );
            int index_y = yi / NERF_OCCUPANCY_GRID_MIN_RES; yi = yi % NERF_OCCUPANCY_GRID_MIN_RES;
            int index_x = yi;
            float pz = ( index_z + hash1( hashbase + yi * 3 ) ) * ( 1.0f / (float)NERF_OCCUPANCY_GRID_MIN_RES ) ;
            float py = ( index_y + hash1( hashbase + yi * 3 + 1 ) ) * ( 1.0f / (float)NERF_OCCUPANCY_GRID_MIN_RES );
            float px = ( index_x + hash1( hashbase + yi * 3 + 2 ) ) * ( 1.0f / (float)NERF_OCCUPANCY_GRID_MIN_RES );
            
            int res = NERF_OCCUPANCY_GRID_MIN_RES;
            for( int lv = 0 ; lv < NERF_OCCUPANCY_GRID_L ; lv++ )
            {
                SequencialHasher hasher;
                hasher.add( index_x, NERF_OCCUPANCY_GRID_MIN_RES );
                hasher.add( index_y, NERF_OCCUPANCY_GRID_MIN_RES );
                hasher.add( index_z, NERF_OCCUPANCY_GRID_MIN_RES );
                res *= 2;

                uint32_t index = hasher.value() % NERF_OCCUPANCY_GRID_T;
				atomicMax( (uint32_t*)&occupancyGrid[NERF_OCCUPANCY_GRID_T * lv + index], __float_as_uint( density ) );
            }
        }
    }
}
extern "C" __global__ void avg( float* occupancyGrid, float* occupancyAverage )
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	// float density = occupancyGrid[x] / ( NERF_OCCUPANCY_GRID_MIN_RES * NERF_OCCUPANCY_GRID_MIN_RES * NERF_OCCUPANCY_GRID_MIN_RES );
	float density = occupancyGrid[x];

    __shared__ float localAvg;

    if( threadIdx.x == 0 )
	{
		localAvg = 0.0f;
    }
    
    __syncthreads();

	atomicAdd( &localAvg, density );

    __syncthreads();

    if( threadIdx.x == 0 )
	{
		atomicAdd( occupancyAverage, localAvg / ( NERF_OCCUPANCY_GRID_MIN_RES * NERF_OCCUPANCY_GRID_MIN_RES * NERF_OCCUPANCY_GRID_MIN_RES ) );
		// atomicAdd( occupancyAverage, localAvg );
	}
	// atomicAdd( occupancyAverage, density );
}
extern "C" __global__ void nerfEval( NeRFRay *rays, NeRFOutput* outputs, float* intermediates, GPUMat nerfSamples, int nElement ) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if( nElement <= x )
    {
        return;
    }
    const float dt = sqrt( 3.0f ) / MLP_STEP;

    float3 oColor = make_float3( 0.0f, 0.0f, 0.0f );
    float T = 1.0f;
    const float Teps = 1e-4f;
    
    int eval_beg = rays[x].eval_beg;
    int eval_end = rays[x].eval_end;
    for( int yi = eval_beg; yi < eval_end; yi++ )
    {
        float density = intermediates[elem( 3, yi, nerfSamples )];
        float sigma = nerfDensityActivation( density );

        float3 c = make_float3(
            nerfRgbActivation( intermediates[elem( 0, yi, nerfSamples )] ),
            nerfRgbActivation( intermediates[elem( 1, yi, nerfSamples )] ),
            nerfRgbActivation( intermediates[elem( 2, yi, nerfSamples )] )
        );
        
        float a = 1.0f - INTRIN_EXPF( -sigma * dt );
        oColor += T * a * c;

        T *= ( 1.0f - a );

        if( T < Teps )
            break;
    }

    outputs[x].color[0] = oColor.x;
    outputs[x].color[1] = oColor.y;
    outputs[x].color[2] = oColor.z;
}

extern "C" __global__ void trainNerfForward( float* intermediates, float* matBuffer, NeRFTrainArg arg ) 
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
    
    for( int i = NERF_DENSITY_LAYER_BEG ; i < NERF_DENSITY_LAYER_END ; i++ )
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
            
            float lowerbounds = i + 1 != NERF_DENSITY_LAYER_END ? 0.0f : -3.40282e+38f;
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
                
                // store intermediates
                int yi = yi_global_base + yi_local;
                if( yi < arg.inputMat.m_row )
                {
                    intermediates[elem( xi, yi, arg.m_Is[i + 1] )] = value[yi_local];
                }
            }
        }

        __syncthreads();
    }

    // density
    if( xi == 0 )
    {
        for( int yi_local = 0 ; yi_local < SHARED_TENSOR_ROW ; yi_local++ )
        {
            float density = value[yi_local];
            int yi = yi_global_base + yi_local;
            if( yi < arg.outputMat.m_row )
            {
                intermediates[ elem( 3, yi, arg.outputMat )] = density;
            }
        }
    }

    // directional
    int nItrEncode = div_round_up( SHARED_TENSOR_ROW, 64 );
    for(int i = 0 ; i < nItrEncode ; i++ )
    {
        int yi_local = i * 64 + xi;
		int yi = yi_global_base + yi_local;
		if( yi_local < SHARED_TENSOR_ROW )
        {
            float x = intermediates[ elem( 0, yi, arg.dirMat )];
            float y = intermediates[ elem( 1, yi, arg.dirMat )];
            float z = intermediates[ elem( 2, yi, arg.dirMat )];
			sh_encode( tensor, yi_local, x, y, z );
        }
    }
    __syncthreads(); 

    // sh - store intermediates
	if( 16 <= xi && xi < 32 )
	{
		for( int yi_local = 0; yi_local < SHARED_TENSOR_ROW; yi_local++ )
		{
			int yi = yi_global_base + yi_local;
			if( yi < arg.inputMat.m_row )
			{
				float sh = getTensor( tensor, xi, yi_local );
				// printf( "[%d] sh %f\n", xi, sh );
				intermediates[elem( xi, yi, arg.m_Is[NERF_COLOR_LAYER_BEG] )] = sh;
			}
		}
    }

    for( int i = NERF_COLOR_LAYER_BEG ; i < NERF_COLOR_LAYER_END ; i++ )
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
            
            float lowerbounds = i + 1 != NERF_COLOR_LAYER_END ? 0.0f : -3.40282e+38f;
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
                
                // store intermediates
                int yi = yi_global_base + yi_local;
                if( yi < arg.inputMat.m_row )
                {
                    intermediates[elem( xi, yi, arg.m_Is[i + 1] )] = value[yi_local];
                }
            }
        }

        __syncthreads();
    }

    if( xi < arg.outputMat.m_col - 1 /* !important! don't override density */ )
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
                intermediates[ elem( xi, yi, arg.outputMat )] = value[yi_local];
            }
        }
    }
}

extern "C" __global__ void nerfDerivative( NeRFRay *rays, NeRFOutput* refs, float* intermediates, GPUMat nerfSamples, int nElement ) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    if( nElement <= x )
    {
        return;
    }

    // forward
    const float dt = sqrt( 3.0f ) / MLP_STEP;

    float3 oColor = make_float3( 0.0f, 0.0f, 0.0f );
    float T = 1.0f;
    const float Teps = 1e-4f;
    
    int eval_beg = rays[x].eval_beg;
    int eval_end = rays[x].eval_end;
    for( int yi = eval_beg; yi < eval_end; yi++ )
    {
        float density = intermediates[elem( 3, yi, nerfSamples )];
        float sigma = nerfDensityActivation( density );
        //if( ( x % 117 ) == 0 )
        //{
        //    printf( "density %f\n", sigma );
        //}

        float3 c = make_float3(
            nerfRgbActivation( intermediates[elem( 0, yi, nerfSamples )] ),
            nerfRgbActivation( intermediates[elem( 1, yi, nerfSamples )] ),
            nerfRgbActivation( intermediates[elem( 2, yi, nerfSamples )] )
        );
        
        float a = 1.0f - INTRIN_EXPF( -sigma * dt );
        oColor += T * a * c;

        T *= ( 1.0f - a );

        if( T < Teps )
            break;
    }

    T = 1.0f; // important!!!! 

    // mse derivative
    float3 refColor = make_float3( refs[x].color[0], refs[x].color[1], refs[x].color[2] );
    float3 dColor = oColor - refColor;

    // backward
    float3 oColor2 = make_float3( 0.0f, 0.0f, 0.0f );
    int tail_yi = eval_end;
    for( int yi = eval_beg; yi < eval_end; yi++ )
    {
        float density = intermediates[elem( 3, yi, nerfSamples )];
        float sigma = nerfDensityActivation( density );
        float3 c = make_float3(
            nerfRgbActivation( intermediates[elem( 0, yi, nerfSamples )] ),
            nerfRgbActivation( intermediates[elem( 1, yi, nerfSamples )] ),
            nerfRgbActivation( intermediates[elem( 2, yi, nerfSamples )] )
        );
        float a = 1.0f - INTRIN_EXPF( -sigma * dt );
        float coefficient = T * a;
        oColor2 += coefficient * c;

        float3 dRGB = make_float3( nerfRgbActivationDrivativeY( c.x ), nerfRgbActivationDrivativeY( c.y ), nerfRgbActivationDrivativeY( c.z ) );
        float3 dOutput = dRGB * coefficient * dColor;
        intermediates[elem( 0, yi, nerfSamples )] = dOutput.x;
        intermediates[elem( 1, yi, nerfSamples )] = dOutput.y;
        intermediates[elem( 2, yi, nerfSamples )] = dOutput.z;

        T *= ( 1.0f - a );

        float3 S = oColor - oColor2;
        float dSigma = dt * dot( T * c - S, dColor );
        intermediates[elem( 3, yi, nerfSamples )] = nerfDensityActivationDerivative( density ) * dSigma;

  //      if( ( x % 117 ) == 0 )
		//{
		//	printf( "density %f / %f ~ %f\n", intermediates[elem( 3, yi, nerfSamples )] , sigma, density );
		//}

        if( T < Teps )
        {
            tail_yi = yi + 1;
            break;
        }
    }
	// printf( "nSkip %d / %d\n", eval_end - tail_yi, eval_end - eval_beg );
    for( int yi = tail_yi; yi < eval_end; yi++ )
    {
        for( int i = 0 ; i < 4; i++ )
        {
            intermediates[elem( i, yi, nerfSamples )] = 0.0f;
        }
    }
}

extern "C" __global__ void trainNerfBackward( float* intermediates, float* matBuffer, float* dMatBuffer, NeRFTrainArg arg ) 
{
    __shared__ float tensor[64 * SHARED_TENSOR_ROW];
    
    int yi_global_base = blockIdx.x * SHARED_TENSOR_ROW;
    int xi = threadIdx.y;
    float value[SHARED_TENSOR_ROW];

    if( xi < arg.outputMat.m_col - 1 /* no need to read density derivative here */ )
    {
        float vs[SHARED_TENSOR_ROW];
        for( int yi_local = 0 ; yi_local < SHARED_TENSOR_ROW ; yi_local++ )
        {
            int yi = yi_global_base + yi_local;
            if( yi < arg.outputMat.m_row )
            {
                vs[yi_local] = intermediates[elem( xi, yi, arg.outputMat )];
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

    // backward
    for( int i = arg.nLayer - 1 ; 0 <= i ; i-- )
    {
        int row = arg.m_Ws[i].m_row; // input
        int col = arg.m_Ws[i].m_col; // output

        // load density
        if( i == NERF_DENSITY_LAYER_END - 1 ) 
        {
            if( xi == 0 )
            {
                for( int yi_local = 0 ; yi_local < SHARED_TENSOR_ROW ; yi_local++ )
                {
                    float v = getTensor( tensor, xi, yi_local );
                    int yi = yi_global_base + yi_local;
                    if( yi < arg.outputMat.m_row )
                    {
						//if( ( yi % 117 ) == 0 )
						//{
						//	printf( "concat %f %f\n", v, intermediates[elem( 3, yi, arg.outputMat )] );
						//}
                        v += intermediates[elem( 3, yi, arg.outputMat )];
                    }
                    setTensor( tensor, xi, yi_local, v );
                }
            }
            __syncthreads();
        }

        if( xi < col )
        {
            float dB = 0.0f;

            // ReLU derivative
			bool noActivation = ( i == NERF_COLOR_LAYER_END - 1 ) || ( i == NERF_DENSITY_LAYER_END - 1 );
			bool activation = !noActivation;
            for( int yi_local = 0 ; yi_local < SHARED_TENSOR_ROW ; yi_local++ )
            {
                float v = getTensor( tensor, xi, yi_local );

				if( activation )
                {
                    int yi = yi_global_base + yi_local;
                    if( yi < arg.inputMat.m_row )
                    {
                        if( intermediates[elem( xi, yi, arg.m_Is[i+1] )] == 0.0f )
                        {
                            v = 0.0f;
                        }
                    }
					else
					{
						v = 0.0f;
                    }
                }

                // bias derivative
                dB += v;

                if( activation )
				{
					setTensor( tensor, xi, yi_local, v );
				}
            }
            
            if( 0.0f != dB )
            {
                atomicAdd( &dMatBuffer[ elem( xi, 0, arg.m_Bs[i] ) ], dB );
            }
        }
        
        __syncthreads();

        if( xi < col )
        {    
   //         if( i == 2 )
			//{
			//	printf( "rc %d %d, %d %d\n", row, col, arg.m_Is[i].m_row, arg.m_Is[i].m_col );
   //         }

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
                if( dW != 0.0f )
                {
                    atomicAdd( &dMatBuffer[ elem( xi, yi_W, arg.m_Ws[i] ) ], dW );
                } 
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
                float wd =  w * derivative;
                if ( wd != 0.0f )
                {
                    atomicAdd( &dMatBuffer[arg.gridFeatureLocation + address], wd );
                }
            }
        }
    }
}