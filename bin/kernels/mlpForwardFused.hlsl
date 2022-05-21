// Macros:
//   GRID_INPUT_DIM, GRID_L, GRID_T, GRID_F, GRID_NMIN, GRID_B
#ifndef GRID_INPUT_DIM
    #define GRID_INPUT_DIM 2
    #define GRID_L 16
    #define GRID_T (1 << 19)
    #define GRID_F 3
    #define GRID_NMIN 8
    #define GRID_B 2.0f
#endif

//     int grid_L;
//     int grid_T;

//     int grid_F;
//     int grid_Nmin;
//     float grid_b;

struct GPUMat
{
    int m_row; // = inputs
    int m_paddedRow;
    int m_col; // = outputs
    int m_location;
};
float getElem( int x, int y, RWStructuredBuffer<float> buffer, GPUMat mat )
{
    return buffer[mat.m_location + mat.m_paddedRow * x + y];
}
void setElem( int x, int y, RWStructuredBuffer<float> buffer, GPUMat mat, float v )
{
    buffer[mat.m_location + mat.m_paddedRow * x + y] = v;
}
float getElem( int x, int y, RWByteAddressBuffer buffer, GPUMat mat )
{
    int index = mat.m_location + mat.m_paddedRow * x + y;
    return asfloat( buffer.Load( index * 4 ) );
}
float4 getElem4( int x, int y, RWByteAddressBuffer buffer, GPUMat mat )
{
    int index = mat.m_location + mat.m_paddedRow * x + y;
    return asfloat( buffer.Load4( index * 4 ) );
}
void setElem( int x, int y, RWByteAddressBuffer buffer, GPUMat mat, float v )
{
    int index = mat.m_location + mat.m_paddedRow * x + y;
    buffer.Store( index * 4, asuint( v ) );
}
void setElem4( int x, int y, RWByteAddressBuffer buffer, GPUMat mat, float4 v )
{
    int index = mat.m_location + mat.m_paddedRow * x + y;
    buffer.Store4( index * 4, asuint( v ) );
}

float relu( float x ) 
{
    return max( x, 0.0f );
}
inline int div_round_up( int val, int divisor )
{
    return ( val + divisor - 1 ) / divisor;
}
inline int next_multiple( int val, int divisor )
{
    return div_round_up( val, divisor ) * divisor;
}

RWByteAddressBuffer inputs;
RWByteAddressBuffer outputs;
RWByteAddressBuffer matBuffer;

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
ConstantBuffer<MLPForwardFusedArg> mlpForwardFusedArg;

struct MLPEncoding
{
    int mode;
    int frequency_N;
    int padd1;
    int padd2;
};
ConstantBuffer<MLPEncoding> mlpEncoding;

RWByteAddressBuffer gridFeature;

// #define TENSOR_ROW 8

// groupshared float tensor[2][TENSOR_ROW * 64];

// #define DISPATCH_CHUNK 8096

// [numthreads(8, 8, 1)]
// void main( uint3 threadId : SV_DispatchThreadID, uint3 localId: SV_GroupThreadID, uint3 groupId: SV_GroupID )
// {
//     // block: 8x8
// 	if( mlpForwardFusedArg.nBlock <= groupId.y + groupId.z * DISPATCH_CHUNK )
// 	{
// 		return;
//     }
// 	threadId.y = threadId.y + threadId.z * DISPATCH_CHUNK * 8 /* numthreads(8, x, 1) */;

//     int yi = threadId.y;
//     int yi_local = localId.y;

//     int tensorIn  = 0;
//     {
//         // matrix row is always multiple of 8
//         for( int xi = 0 ; xi < mlpForwardFusedArg.inputMat.m_col ; xi++)
//         {
// 			tensor[tensorIn][xi * TENSOR_ROW + yi_local] = getElem( xi, yi, inputs, mlpForwardFusedArg.inputMat );
//         }
//     }

//     GroupMemoryBarrierWithGroupSync();

//     // 
//     // int tensorOut = 1;
//     [unroll(8)]
//     for( int i = 0 ; i < mlpForwardFusedArg.nLayer ; i++ )
//     {
//         int row = mlpForwardFusedArg.m_Ws[i].m_row; // input
//         int col = mlpForwardFusedArg.m_Ws[i].m_col; // output

//         for( int xi_origin = 0 ; xi_origin < col ; xi_origin += 8 )
//         {
//             int xi = xi_origin + localId.x;
//             if( col <= xi )
//             {
//                 continue;
//             }

//             float s = 0.0f;
//             for( int j = 0 ; j < row ; j++ ) 
//             {
//                 float a = tensor[tensorIn % 2][ j * TENSOR_ROW + yi_local ];
//                 float b = getElem( xi, j, matBuffer, mlpForwardFusedArg.m_Ws[i] );
//                 s += a * b;
//             }

//             float bias = getElem( xi, 0, matBuffer, mlpForwardFusedArg.m_Bs[i] );
//             float value = s + bias;
//             if( i + 1 != mlpForwardFusedArg.nLayer )
//             {
//                 value = relu( value );
//             }
//             tensor[ ( tensorIn + 1 ) % 2 ][ xi * TENSOR_ROW + yi_local ] = value;
//         }
//         tensorIn++;
//         GroupMemoryBarrierWithGroupSync();
//     }

//     {
//         // matrix row is always multiple of 8
//         for( int xi = 0 ; xi < mlpForwardFusedArg.outputMat.m_col ; xi++)
//         {
//             float s = tensor[tensorIn % 2][ xi * TENSOR_ROW + yi_local ];
//             setElem( xi, yi, outputs, mlpForwardFusedArg.outputMat, s );
//         }
//     }
// }

// groupshared float tensor[64];

// #define DISPATCH_CHUNK 8096

// [numthreads(64, 1, 1)]
// void main( uint3 threadId : SV_DispatchThreadID, uint3 localId: SV_GroupThreadID )
// {
// 	threadId.y = threadId.y + threadId.z * DISPATCH_CHUNK;
//     if( mlpForwardFusedArg.inputMat.m_row <= threadId.y )
//     {
//         return;
//     }

//     int xi = localId.x;
//     int yi = threadId.y;
//     if( xi < mlpForwardFusedArg.inputMat.m_col ) 
//     {
//         tensor[xi] = getElem( xi, yi, inputs, mlpForwardFusedArg.inputMat );
//     }

//     GroupMemoryBarrierWithGroupSync();

//     [unroll(4)]
//     for( int i = 0 ; i < mlpForwardFusedArg.nLayer ; i++ )
//     {
//         int row = mlpForwardFusedArg.m_Ws[i].m_row; // input
//         int col = mlpForwardFusedArg.m_Ws[i].m_col; // output

//         float value = 0.0f;
//         if( xi < col )
//         {
//             for( int j = 0 ; j < row ; j++ ) 
//             {
//                 float a = tensor[j];
//                 float b = getElem( xi /* output xi */, j, matBuffer, mlpForwardFusedArg.m_Ws[i] );
//                 value += a * b;
//             }
//             float bias = getElem( xi, 0, matBuffer, mlpForwardFusedArg.m_Bs[i] );
//             value += bias;
//             if( i + 1 != mlpForwardFusedArg.nLayer )
//             {
//                 value = relu( value );
//             }
//         }
//         GroupMemoryBarrierWithGroupSync();
//         if( xi < col )
//         {
//             tensor[xi] = value;
//         }
//         GroupMemoryBarrierWithGroupSync();
//     }

//     if( xi < mlpForwardFusedArg.outputMat.m_col ) 
//     {
//         float s = tensor[xi];
//         setElem( xi, yi, outputs, mlpForwardFusedArg.outputMat, s );
//     }
// }

// ex) 
// [ TENSOR_ROW ] [ TENSOR_ROW ] [ TENSOR_ROW ]
// nBlock: 3

#define DISPATCH_CHUNK 8096
#define TENSOR_ROW 16
#define PI 3.14159265358979323846f

static const uint PRIMES[7] = { 1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737 };

class DimensionHasher
{
    uint m_h;
    void init()
    {
        m_h = 0;
    }
    void add( uint xs, int d )
    {
        m_h ^= xs * PRIMES[ min( d, 6 ) ];
    }
    uint value() { return m_h; }
};

struct HashGridEvaluator
{
    int m_dim;
    uint m_bits;
    void init( int dim )
    {
        m_dim = dim;
        m_bits = 0xFFFFFFFF;
    }
    bool moveNext()
    {
        m_bits++;
        return m_bits < ( 1U << m_dim );
    }
    void evaluate( out float weight, out uint hashValue, int resolution, float input[GRID_INPUT_DIM] )
    {
        DimensionHasher hasher;
        hasher.init();

        float w = 1.0f;
        for( int d = 0; d < m_dim; ++d )
        {
            float x_in = input[ min( d, GRID_INPUT_DIM - 1 ) ];

            float xf = x_in * resolution;
            uint xi = xf;
            float u = xf - xi;

            if( m_bits & ( 1U << d ) )
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
        weight = w;
        hashValue = hasher.value();
    }
};

groupshared float tensor[64 * TENSOR_ROW];

// column major for ds_read_b128
float getTensor( int xi, int yi )
{
    return tensor[xi * TENSOR_ROW + yi];
}
void setTensor( int xi, int yi, float value )
{
    tensor[xi * TENSOR_ROW + yi] = value;
}

[numthreads(64, 1, 1)]
void main( uint3 threadId : SV_DispatchThreadID, uint3 localId: SV_GroupThreadID )
{
    float value[TENSOR_ROW];

	threadId.y = threadId.y + threadId.z * DISPATCH_CHUNK;
    if( mlpForwardFusedArg.nBlock <= threadId.y )
    {
        return;
    }

    int xi = localId.x;
    if( xi < mlpForwardFusedArg.inputMat.m_col )
    {
        int yi_local;
        for( yi_local = 0 ; yi_local < TENSOR_ROW ; yi_local += 4 )
        {
            int yi = threadId.y * TENSOR_ROW + yi_local;
            float4 v = float4( 0.0f, 0.0f, 0.0f, 0.0f );
            if( yi < mlpForwardFusedArg.inputMat.m_row )
            {
                v = getElem4( xi, yi, inputs, mlpForwardFusedArg.inputMat );
            }
            for( int j = 0 ; j < 4 ; j++ )
            {
                setTensor( xi, yi_local + j, v[j] ); // ds_write_B128
            }
        }
    }

    GroupMemoryBarrierWithGroupSync();

    if( mlpEncoding.mode == 1 ) // frequency
    {
        int yi_local;
        int outputCol = mlpForwardFusedArg.inputMat.m_col * mlpEncoding.frequency_N * 2;

        if( xi < outputCol )
        {
            for( yi_local = 0 ; yi_local < TENSOR_ROW ; yi_local++ )
            {
                int xi_src = xi / ( mlpEncoding.frequency_N * 2 );
                int baseEachDim = xi % ( mlpEncoding.frequency_N * 2 );
                int i = baseEachDim / 2;
                int tri_idx = baseEachDim % 2;
                float v = getTensor( xi_src, yi_local );
                float k = 2.0f * PI * pow( 2.0f, (float)i );
                v = sin( k * v + ( tri_idx ? PI * 0.5f : 0.0f ) );
                value[yi_local] = v;
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if( xi < outputCol )
        {
            for( yi_local = 0 ; yi_local < TENSOR_ROW ; yi_local++ )
            {
                setTensor( localId.x, yi_local, value[yi_local] );
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if( mlpEncoding.mode == 2 ) // Multi Resolution Hash
    {
        int level = xi / GRID_F;
        int fdim  = xi % GRID_F;
        float res = floor( GRID_NMIN * pow( GRID_B, level ) );
        for( int yi_local = 0 ; yi_local < TENSOR_ROW ; yi_local++ )
        {
            float input[GRID_INPUT_DIM];
            for( int x = 0 ; x < GRID_INPUT_DIM ; x++ )
            {
                input[x] = getTensor( x, yi_local );
            }
            GroupMemoryBarrierWithGroupSync();

            HashGridEvaluator evaluator;
            evaluator.init( GRID_INPUT_DIM );
            float feature = 0.0f;
            while( evaluator.moveNext() )
            {
                float w;
                uint h;
                evaluator.evaluate( w, h, res, input );
                uint index = h % GRID_T;
                int baseLevel = GRID_T * GRID_F * level;
                int address = baseLevel + GRID_T * fdim + index;
                float f = asfloat( gridFeature.Load( address * 4 ) );
                feature += w * f;
            }

            setTensor( xi, yi_local, feature );
        }
        GroupMemoryBarrierWithGroupSync();
    }
    for( int i = 0 ; i < mlpForwardFusedArg.nLayer ; i++ )
    {
        int row = mlpForwardFusedArg.m_Ws[i].m_row; // input
        int col = mlpForwardFusedArg.m_Ws[i].m_col; // output

        {
            float bias = xi < col ? getElem( xi, 0, matBuffer, mlpForwardFusedArg.m_Bs[i] ) : 0.0f;
            for( int yi_local = 0 ; yi_local < TENSOR_ROW ; yi_local++ )
            {
                value[yi_local] = bias;
            }
        }
        
        if( xi < col )
        {
            float4 bs;
            for( int j = 0 ; j < row ; j++ ) 
            {
                if( ( j % 4 ) == 0 )
                {
                    bs = getElem4( xi /* output xi */, j, matBuffer, mlpForwardFusedArg.m_Ws[i] );
                }
                float b = bs[j % 4];
                for( int yi_local = 0 ; yi_local < TENSOR_ROW ; yi_local++ )
                {
                    float a = getTensor( j, yi_local );
                    value[yi_local] += a * b;
                }
            }
            
            float lowerbounds = i + 1 != mlpForwardFusedArg.nLayer ? 0.0f : -3.40282e+38f;
            for( int yi_local = 0 ; yi_local < TENSOR_ROW ; yi_local++ )
            {
                value[yi_local] = max( value[yi_local], lowerbounds );
            }
        }
        GroupMemoryBarrierWithGroupSync();
        if( xi < col )
        {
            for( int yi_local = 0 ; yi_local < TENSOR_ROW ; yi_local++ )
            {
                setTensor( xi, yi_local, value[yi_local] );
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if( xi < mlpForwardFusedArg.outputMat.m_col )
    {
        int yi_local;
        for( yi_local = 0 ; yi_local < TENSOR_ROW ; yi_local++ )
        {
            value[yi_local] = getTensor( xi, yi_local ); // ds_read_B128
        }

        for( yi_local = 0 ; yi_local < TENSOR_ROW ; yi_local += 4 )
        {
            int yi = threadId.y * TENSOR_ROW + yi_local;
            if( yi < mlpForwardFusedArg.outputMat.m_row )
            {
                float4 v = float4( value[yi_local], value[yi_local + 1], value[yi_local + 2], value[yi_local + 3] );
                setElem4( xi, yi, outputs, mlpForwardFusedArg.outputMat, v );
            }
        }
    }
}

// HALF ver
// #define DISPATCH_CHUNK 8096
// #define TENSOR_ROW 32
// #define PI 3.14159265358979323846f

// #define GRID_MAX_INPUT_DIM 4

// static const uint PRIMES[7] = { 1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737 };

// class DimensionHasher
// {
//     uint m_h;
//     void init()
//     {
//         m_h = 0;
//     }
//     void add( uint xs, int d )
//     {
//         m_h ^= xs * PRIMES[ min( d, 6 ) ];
//     }
//     uint value() { return m_h; }
// };

// struct HashGridEvaluator
// {
//     int m_dim;
//     uint m_bits;
//     void init( int dim )
//     {
//         m_dim = dim;
//         m_bits = 0xFFFFFFFF;
//     }
//     bool moveNext()
//     {
//         m_bits++;
//         return m_bits < ( 1U << m_dim );
//     }
//     void evaluate( out float weight, out uint hashValue, int resolution, float input[GRID_MAX_INPUT_DIM] )
//     {
//         DimensionHasher hasher;
//         hasher.init();

//         float w = 1.0f;
//         for( int d = 0; d < m_dim; ++d )
//         {
//             float x_in = input[ min( d, GRID_MAX_INPUT_DIM - 1 ) ];

//             float xf = x_in * resolution;
//             uint xi = xf;
//             float u = xf - xi;

//             if( m_bits & ( 1U << d ) )
//             {
//                 w *= u;
//                 hasher.add( xi + 1, d );
//             }
//             else
//             {
//                 w *= 1.0f - u;
//                 hasher.add( xi, d );
//             }
//         }
//         weight = w;
//         hashValue = hasher.value();
//     }
// };

// groupshared uint tensor[64 * TENSOR_ROW / 2];

// // column major for ds_read_b128
// min16float getTensor( int xi, int yi )
// {
//     uint v = tensor[xi * TENSOR_ROW / 2 + yi / 2];
//     if( ( yi % 2 ) != 0 )
//     {
//         v = v >> 16;
//     }
//     return f16tof32( v );
// }
// void setTensor( int xi, int yi, min16float value )
// {
//     uint fp16 = f32tof16( value );
//     uint v = tensor[xi * TENSOR_ROW / 2 + yi / 2];
//     if( ( yi % 2 ) != 0 )
//     {
//         tensor[xi * TENSOR_ROW / 2 + yi / 2] = ( v & 0x0000FFFF ) | ( fp16 << 16 );
//     }
//     else
//     {
//         tensor[xi * TENSOR_ROW / 2 + yi / 2] = ( v & 0xFFFF0000 ) | fp16;
//     }
// }

// [numthreads(64, 1, 1)]
// void main( uint3 threadId : SV_DispatchThreadID, uint3 localId: SV_GroupThreadID )
// {
// 	threadId.y = threadId.y + threadId.z * DISPATCH_CHUNK;
//     if( mlpForwardFusedArg.nBlock <= threadId.y )
//     {
//         return;
//     }

//     int xi = localId.x;
//     if( xi < mlpForwardFusedArg.inputMat.m_col )
//     {
//         uint value[TENSOR_ROW / 2];

//         int yi_local;
//         for( yi_local = 0 ; yi_local < TENSOR_ROW ; yi_local += 2 )
//         {
//             int yi_0 = threadId.y * TENSOR_ROW + yi_local;
//             int yi_1 = threadId.y * TENSOR_ROW + yi_local + 1;
//             float v0 = yi_0 < mlpForwardFusedArg.inputMat.m_row ? getElem( xi, yi_0, inputs, mlpForwardFusedArg.inputMat ) : 0.0f;
//             float v1 = yi_1 < mlpForwardFusedArg.inputMat.m_row ? getElem( xi, yi_1, inputs, mlpForwardFusedArg.inputMat ) : 0.0f;
//             value[yi_local / 2] = f32tof16( v0 ) | ( f32tof16( v1 ) << 16 );
//         }
//         for( yi_local = 0 ; yi_local < TENSOR_ROW / 2 ; yi_local++ )
//         {
//             tensor[xi * TENSOR_ROW / 2 + yi_local] = value[yi_local]; // ds_write_B128
//         }
//     }

//     GroupMemoryBarrierWithGroupSync();

//     for( int i = 0 ; i < mlpForwardFusedArg.nLayer ; i++ )
//     {
//         int row = mlpForwardFusedArg.m_Ws[i].m_row; // input
//         int col = mlpForwardFusedArg.m_Ws[i].m_col; // output

//         uint value[TENSOR_ROW/2];
//         {
//             for( int yi_local = 0 ; yi_local < TENSOR_ROW/2 ; yi_local++ )
//             {
//                 value[yi_local] = 0;
//             }
//         }
        
//         if( xi < col )
//         {
//             for( int j = 0 ; j < row ; j++ ) 
//             {
//                 min16float  b = getElem( xi /* output xi */, j, matBuffer, mlpForwardFusedArg.m_Ws[i] );
//                 min16float2 bb = min16float2( b, b );
//                 for( int yi_local = 0 ; yi_local < TENSOR_ROW ; yi_local += 2 )
//                 {
//                     uint a = tensor[j * TENSOR_ROW / 2 + yi_local / 2];
//                     uint c = value[yi_local / 2];
//                     min16float2 cc = min16float2( (min16float)f16tof32( c ), (min16float)f16tof32( c >> 16 ) );
//                     min16float2 aa = min16float2( (min16float)f16tof32( a ), (min16float)f16tof32( a >> 16 ) );
//                     cc += aa * bb;
//                     uint2 ccii = f32tof16( float2( cc.x, cc.y ) );
//                     value[yi_local / 2] = ccii.x | ( ccii.y << 16 );
//                 }
//             }
            
//             min16float bias = getElem( xi, 0, matBuffer, mlpForwardFusedArg.m_Bs[i] );

//             min16float2 relu_lower2 = i + 1 != mlpForwardFusedArg.nLayer ? min16float2( 0.0h, 0.0h ) : min16float2( -65504.f, -65504.f );
//             for( int yi_local = 0 ; yi_local < TENSOR_ROW ; yi_local += 2 )
//             {
//                 uint c = value[yi_local / 2];
//                 min16float2 cc = min16float2( (min16float)f16tof32( c ), (min16float)f16tof32( c >> 16 ) );
//                 cc += min16float2( bias, bias );
//                 cc = max( cc, relu_lower2 );
//                 uint2 ccii = f32tof16( float2( cc.x, cc.y ) );
//                 value[yi_local / 2] = ccii.x | ( ccii.y << 16 );
//             }
//         }
//         GroupMemoryBarrierWithGroupSync();
//         if( xi < col )
//         {
//             for( int yi_local = 0 ; yi_local < TENSOR_ROW / 2 ; yi_local++ )
//             {
//                 tensor[xi * TENSOR_ROW / 2 + yi_local] = value[yi_local];
//             }
//         }
//         GroupMemoryBarrierWithGroupSync();
//     }

//     if( xi < mlpForwardFusedArg.outputMat.m_col )
//     {
//         int yi_local;
//         for( yi_local = 0 ; yi_local < TENSOR_ROW ; yi_local += 2 )
//         {
//             uint v = tensor[xi * TENSOR_ROW / 2 + yi_local / 2];
//             int yi_0 = threadId.y * TENSOR_ROW + yi_local;
//             int yi_1 = threadId.y * TENSOR_ROW + yi_local + 1;
//             float v0 = f16tof32( v );
//             float v1 = f16tof32( v >> 16 );
//             if( yi_0 < mlpForwardFusedArg.outputMat.m_row )
//             {
//                 setElem( xi, yi_0, outputs, mlpForwardFusedArg.outputMat, v0 );
//             }
//             if( yi_1 < mlpForwardFusedArg.outputMat.m_row )
//             {
//                 setElem( xi, yi_1, outputs, mlpForwardFusedArg.outputMat, v1 );
//             }
//         }
//     }
// }