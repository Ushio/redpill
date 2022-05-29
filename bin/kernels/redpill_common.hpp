#pragma once

#if defined( __HIPCC__ ) || defined( __CUDACC__ )
#define DEVICE __device__
typedef unsigned int uint32_t;
#else
#define DEVICE
#endif

#define SHARED_TENSOR_ROW 16

namespace rpml
{
	const float pi = 3.14159265358979323846f;

    DEVICE
    inline int div_round_up( int val, int divisor )
    {
        return ( val + divisor - 1 ) / divisor;
    }

    DEVICE
    inline int next_multiple( int val, int divisor )
    {
        return div_round_up( val, divisor ) * divisor;
    }

    struct GPUMat
	{
		int m_row; // = inputs
		int m_paddedRow;
		int m_col; // = outputs
		int m_location;
	};
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
	struct MLPEncodingArg
	{
		int mode;
		int frequency_N;
		int padd1;
		int padd2;
	};

	DEVICE
	int elem( int x, int y, GPUMat mat )
	{
		return mat.m_location + mat.m_paddedRow * x + y;
	}
}