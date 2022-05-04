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

RWStructuredBuffer<float> inputs;
RWStructuredBuffer<float> outputs;
RWStructuredBuffer<float> matBuffer;

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
    int grid_L;
    int grid_T;

    int grid_F;
    int grid_Nmin;
    float grid_b;
    int padd1;
};
ConstantBuffer<MLPEncoding> mlpEncoding;

RWStructuredBuffer<float> gridFeature;

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

#define GRID_MAX_INPUT_DIM 4

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
    void evaluate( out float weight, out uint hashValue, int resolution, float input[GRID_MAX_INPUT_DIM] )
    {
        DimensionHasher hasher;
        hasher.init();

        float w = 1.0f;
        for( int d = 0; d < m_dim; ++d )
        {
            float x_in = input[ min( d, GRID_MAX_INPUT_DIM - 1 ) ];

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

// groupshared float tensor[TENSOR_ROW][64];
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
        for( yi_local = 0 ; yi_local < TENSOR_ROW ; yi_local++ )
        {
            int yi = threadId.y * TENSOR_ROW + yi_local;
            value[yi_local] = yi < mlpForwardFusedArg.inputMat.m_row ? getElem( xi, yi, inputs, mlpForwardFusedArg.inputMat ) : 0.0f;
        }
        for( yi_local = 0 ; yi_local < TENSOR_ROW ; yi_local++ )
        {
            setTensor( xi, yi_local, value[yi_local] ); // ds_write_B128
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
        int nTasks = TENSOR_ROW * mlpEncoding.grid_L;
        int nIter = div_round_up( nTasks, 64 );
        int dim = mlpForwardFusedArg.inputMat.m_col;
        
        for( int iIter = 0 ; iIter < nIter ; iIter++ )
        {
            // each local item is in charge of handle a level of a row
            // (yi=0,lv=0), (yi=0,lv=1).. (yi=0,lv=15), (yi=1,lv=0), (yi=1,lv=1)...
            int i = iIter * 64 + localId.x;
            int yi_local = i / mlpEncoding.grid_L;
            int level    = i % mlpEncoding.grid_L;
            float res = floor( mlpEncoding.grid_Nmin * pow( mlpEncoding.grid_b, level ) );
            int elementsPerTable = mlpEncoding.grid_T * mlpEncoding.grid_F;
            int baseLevel = elementsPerTable * level;
            float input[GRID_MAX_INPUT_DIM];
            for( int xi = 0 ; xi < GRID_MAX_INPUT_DIM ; xi++ )
            {
                input[xi] = getTensor( xi, yi_local );
            }
            GroupMemoryBarrierWithGroupSync();

            // all inputs have been loaded. so clear all target feature vector.
            for( int fdim = 0; fdim < mlpEncoding.grid_F; fdim++ )
            {
                setTensor( level * mlpEncoding.grid_F + fdim, yi_local, 0.0f );
            }
            GroupMemoryBarrierWithGroupSync();

            if( yi_local < TENSOR_ROW )
            {
                HashGridEvaluator evaluator;
                evaluator.init( dim );
                while( evaluator.moveNext() )
                {
                    float w;
                    uint h;
                    evaluator.evaluate( w, h, res, input );
                    uint index = h % mlpEncoding.grid_T;
                    int elementsPerFeature = mlpEncoding.grid_T;
                    for( int fdim = 0; fdim < mlpEncoding.grid_F; fdim++ )
                    {
                        float f = gridFeature[baseLevel + elementsPerFeature * fdim + index];
                        float v = getTensor( level * mlpEncoding.grid_F + fdim, yi_local );
                        setTensor( level * mlpEncoding.grid_F + fdim, yi_local, v + w * f );
                    }
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    for( int i = 0 ; i < mlpForwardFusedArg.nLayer ; i++ )
    {
        int row = mlpForwardFusedArg.m_Ws[i].m_row; // input
        int col = mlpForwardFusedArg.m_Ws[i].m_col; // output

        {
            for( int yi_local = 0 ; yi_local < TENSOR_ROW ; yi_local++ )
            {
                value[yi_local] = 0.0f;
            }
        }
        
        if( xi < col )
        {
            for( int j = 0 ; j < row ; j++ ) 
            {
                float b = getElem( xi /* output xi */, j, matBuffer, mlpForwardFusedArg.m_Ws[i] );
                for( int yi_local = 0 ; yi_local < TENSOR_ROW ; yi_local++ )
                {
                    float a = getTensor( j, yi_local );
                    value[yi_local] += a * b;
                }
            }
            float bias = getElem( xi, 0, matBuffer, mlpForwardFusedArg.m_Bs[i] );

            for( int yi_local = 0 ; yi_local < TENSOR_ROW ; yi_local++ )
            {
                value[yi_local] += bias;
                if( i + 1 != mlpForwardFusedArg.nLayer )
                {
                    value[yi_local] = relu( value[yi_local] );
                }
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

        for( yi_local = 0 ; yi_local < TENSOR_ROW ; yi_local++ )
        {
            int yi = threadId.y * TENSOR_ROW + yi_local;
            if( yi < mlpForwardFusedArg.outputMat.m_row )
            {
                setElem( xi, yi, outputs, mlpForwardFusedArg.outputMat, value[yi_local] );
            }
        }
    }
}