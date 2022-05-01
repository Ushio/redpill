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
    int padd0;
    int padd1;
    int padd2;
};
ConstantBuffer<MLPForwardFusedArg> mlpForwardFusedArg;


// groupshared float tensor[2][64 * 64];

// [numthreads(1,64,1)]
// void main( uint3 threadId : SV_DispatchThreadID, uint3 localId: SV_GroupThreadID )
// {
//     // int xi = threadId.x;
//     int yi = threadId.y;
//     int yi_local = localId.y;

//     {
//         // todo out of bounds
//         for( int xi = 0 ; xi < mlpForwardFusedArg.inputMat.m_col ; xi++)
//         {
//             tensor[0][ xi * 64 + yi_local ] = getElem( xi, yi, inputs, mlpForwardFusedArg.inputMat );
//         }
//     }

//     GroupMemoryBarrierWithGroupSync();

//     int tensorIn  = 0;
//     int tensorOut = 1;
    
//     for( int i = 0 ; i < mlpForwardFusedArg.nLayer ; i++ )
//     {
//         int row = mlpForwardFusedArg.m_Ws[i].m_row; // input
//         int col = mlpForwardFusedArg.m_Ws[i].m_col; // output

//         for( int xi = 0 ; xi < col ; xi++ )
//         {
//             float s = 0.0f;
//             for( int j = 0 ; j < row ; j++ ) 
//             {
//                 float a = tensor[tensorIn][ j * 64 + yi_local ];
//                 float b = getElem( xi, j, matBuffer, mlpForwardFusedArg.m_Ws[i] );
//                 s += a * b;
//             }
//             float bias = getElem( xi, 0, matBuffer, mlpForwardFusedArg.m_Bs[i] );
//             float value = s + bias;
//             tensor[tensorOut][ xi * 64 + yi_local ] = relu( value );
//         }

//         int tmp = tensorIn;
//         tensorIn = tensorOut;
//         tensorOut = tmp;
//         GroupMemoryBarrierWithGroupSync();
//     }

//     GroupMemoryBarrierWithGroupSync();
    
//     {
//         // todo out of bounds
//         for( int xi = 0 ; xi < mlpForwardFusedArg.outputMat.m_col ; xi++)
//         {
//             float s = tensor[tensorIn][ xi * 64 + yi_local ];
//             setElem( xi, yi, outputs, mlpForwardFusedArg.outputMat, s );
//         }
//     }
// }

#define TENSOR_ROW 8

groupshared float tensor[2][TENSOR_ROW * 64];

[numthreads(8, 8, 1)]
void main( uint3 threadId : SV_DispatchThreadID, uint3 localId: SV_GroupThreadID )
{
    // int xi = threadId.x;
    int yi = threadId.y;
    int yi_local = localId.y;

    {
        // matrix row is always multiple of 8
        for( int xi = 0 ; xi < mlpForwardFusedArg.inputMat.m_col ; xi++)
        {
            if( yi_local < TENSOR_ROW )
            {
                tensor[0][ xi * TENSOR_ROW + yi_local ] = getElem( xi, yi, inputs, mlpForwardFusedArg.inputMat );
            }
        }
    }

    GroupMemoryBarrierWithGroupSync();

    // int tensorIn  = 0;
    // int tensorOut = 1;
    [unroll(8)]
    for( int i = 0 ; i < mlpForwardFusedArg.nLayer ; i++ )
    {
        int row = mlpForwardFusedArg.m_Ws[i].m_row; // input
        int col = mlpForwardFusedArg.m_Ws[i].m_col; // output

        for( int xi_origin = 0 ; xi_origin < col ; xi_origin += 8 )
        {
            int xi = xi_origin + localId.x;
            if( col <= xi )
            {
                continue;
            }

            float s = 0.0f;
            for( int j = 0 ; j < row ; j++ ) 
            {
                float a = tensor[i % 2][ j * TENSOR_ROW + yi_local ];
                float b = getElem( xi, j, matBuffer, mlpForwardFusedArg.m_Ws[i] );
                s += a * b;
            }
            float bias = getElem( xi, 0, matBuffer, mlpForwardFusedArg.m_Bs[i] );
            float value = s + bias;
            if( i + 1 != mlpForwardFusedArg.nLayer )
            {
                value = relu( value );
            }
            tensor[ ( i + 1 ) % 2 ][ xi * TENSOR_ROW + yi_local ] = value;
        }
        GroupMemoryBarrierWithGroupSync();
    }

    int tensorOut = mlpForwardFusedArg.nLayer % 2;
    {
        // matrix row is always multiple of 8
        for( int xi = 0 ; xi < mlpForwardFusedArg.outputMat.m_col ; xi++)
        {
            if( yi_local < TENSOR_ROW )
            {
                float s = tensor[tensorOut][ xi * TENSOR_ROW + yi_local ];
                setElem( xi, yi, outputs, mlpForwardFusedArg.outputMat, s );
            }
        }
    }
}