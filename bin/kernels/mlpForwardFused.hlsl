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

groupshared float tensor[2][64 * 64];

[numthreads(1,64,1)]
void main( uint3 threadId : SV_DispatchThreadID, uint3 localId: SV_GroupThreadID )
{
    // int xi = threadId.x;
    int yi = threadId.y;
    int yi_local = localId.y;

    {
        // todo out of bounds
        for( int xi = 0 ; xi < mlpForwardFusedArg.inputMat.m_col ; xi++)
        {
            tensor[0][ xi * 64 + yi_local ] = getElem( xi, yi, inputs, mlpForwardFusedArg.inputMat );
        }
    }

    GroupMemoryBarrierWithGroupSync();

    int tensorIn  = 0;
    int tensorOut = 1;
    
    for( int i = 0 ; i < mlpForwardFusedArg.nLayer ; i++ )
    {
        int row = mlpForwardFusedArg.m_Ws[i].m_row; // input
        int col = mlpForwardFusedArg.m_Ws[i].m_col; // output

        for( int xi = 0 ; xi < col ; xi++ )
        {
            float s = 0.0f;
            for( int j = 0 ; j < row ; j++ ) 
            {
                float a = tensor[tensorIn][ j * 64 + yi_local ];
                float b = getElem( xi, j, matBuffer, mlpForwardFusedArg.m_Ws[i] );
                s += a * b;
            }
            float bias = getElem( xi, 0, matBuffer, mlpForwardFusedArg.m_Bs[i] );
            float value = s + bias;
            tensor[tensorOut][ xi * 64 + yi_local ] = relu( value );
        }

        int tmp = tensorIn;
        tensorIn = tensorOut;
        tensorOut = tmp;
        GroupMemoryBarrierWithGroupSync();
    }

    GroupMemoryBarrierWithGroupSync();
    
    {
        // todo out of bounds
        for( int xi = 0 ; xi < mlpForwardFusedArg.outputMat.m_col ; xi++)
        {
            float s = tensor[tensorIn][ xi * 64 + yi_local ];
            setElem( xi, yi, outputs, mlpForwardFusedArg.outputMat, s );
        }
    }
}

// dispatch ( 1, N / 8, 1 )
// [numthreads(8,8,1)]
// void main( uint3 threadId : SV_DispatchThreadID, uint3 localID: SV_GroupThreadID )
// {
//     {
//         int row = forwardInput.inputMat.m_row;
//         int col = forwardInput.inputMat.m_col;
//         for( int iCol = 0 ; iCol < col ; iCol += 8 )
//         {
//             int xi = iCol + localID.x;
//             int yi_to   = localID.y;
//             int yi_from = threadId.y;

//             float value;
//             if( xi < col && yi_from < row )
//             {
//                 value = getElem( xi, yi_from, inputs, forwardInput.inputMat );
//             }
//             else
//             {
//                 value = 0.0f;
//             }
//             sm[0][ xi * 8 + yi_from ] = value;
//         }
//     }

//     GroupMemoryBarrierWithGroupSync();

//     int smIn  = 0;
//     int smOut = 1;
//     for( int i = 0 ; i < forwardInput.nLayer ; i++ )
//     {
//         int row = forwardInput.m_Ws[i].m_row; // input
//         int col = forwardInput.m_Ws[i].m_col; // output
//         for( int iCol = 0 ; iCol < col ; iCol += 8 )
//         {
//             int xi = iCol + localID.x;
//             int yi = localID.y;

//             float s = 0.0f;
//             for( int j = 0 ; j < row ; j++ ) 
//             {
//                 float a = sm[smIn][ j * 8 + yi ];
//                 float b = getElem( xi, yi + j, matBuffer, forwardInput.m_Ws[i] );
//                 s += a * b;
//             }
//             float bias = getElem( xi, 0, matBuffer, forwardInput.m_Bs[i] );
//             float value = s + bias;
//             sm[smOut][ xi * 8 + yi ] = relu( value );
//         }

//         int tmp = smIn;
//         smIn = smOut;
//         smOut = tmp;
//         GroupMemoryBarrierWithGroupSync();
//     }

//     // uint index = threadId.x;
//     inputs[0] = 0;
//     outputs[0] = 0;
//     matBuffer[0] = 0;
// }
