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

// struct MLPEncoding
// {
//     int mode;
//     int frequency_N;
// };
// ConstantBuffer<MLPEncoding> mlpEncoding;

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

groupshared float tensor[TENSOR_ROW][64];

[numthreads(64, 1, 1)]
void main( uint3 threadId : SV_DispatchThreadID, uint3 localId: SV_GroupThreadID )
{
	threadId.y = threadId.y + threadId.z * DISPATCH_CHUNK;
    if( mlpForwardFusedArg.nBlock <= threadId.y )
    {
        return;
    }

    int xi = localId.x;
    {
        for( int yi_local = 0 ; yi_local < TENSOR_ROW ; yi_local++ )
        {
            int yi = threadId.y * TENSOR_ROW + yi_local;
            if( xi < mlpForwardFusedArg.inputMat.m_col )
            {
                tensor[yi_local][xi] = getElem( xi, yi, inputs, mlpForwardFusedArg.inputMat );
            }
        }
    }

    GroupMemoryBarrierWithGroupSync();

    [unroll(4)]
    for( int i = 0 ; i < mlpForwardFusedArg.nLayer ; i++ )
    {
        int row = mlpForwardFusedArg.m_Ws[i].m_row; // input
        int col = mlpForwardFusedArg.m_Ws[i].m_col; // output

        float value[TENSOR_ROW];
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
                    float a = tensor[yi_local][j];
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
                tensor[yi_local][xi] = value[yi_local];
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    for( int yi_local = 0 ; yi_local < TENSOR_ROW ; yi_local++ )
    {
        int yi = threadId.y * TENSOR_ROW + yi_local;
        if( xi < mlpForwardFusedArg.outputMat.m_col )
        {
            setElem( xi, yi, outputs, mlpForwardFusedArg.outputMat, tensor[yi_local][xi] );
        }
    }
}