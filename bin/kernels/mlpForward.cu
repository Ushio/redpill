#ifndef GRID_L
    #define GRID_INPUT_DIM 1
    #define GRID_L 16
    #define GRID_T (1 << 19)
    #define GRID_F 3
    #define GRID_NMIN 8
    #define GRID_B 2.0f
#endif
typedef unsigned int uint32_t;

#define DEVICE __device__

#define TENSOR_ROW 16
#define PI 3.14159265358979323846f

struct GPUMat
{
    int m_row; // = inputs
    int m_paddedRow;
    int m_col; // = outputs
    int m_location;
};
DEVICE
float getElem( int x, int y, float* buffer, GPUMat mat )
{
    return buffer[mat.m_location + mat.m_paddedRow * x + y];
}

DEVICE
void setElem( int x, int y, float* buffer, GPUMat mat, float v )
{
    buffer[mat.m_location + mat.m_paddedRow * x + y] = v;
}

DEVICE
float4 getElem4( int x, int y, float* buffer, GPUMat mat )
{
    int index = mat.m_location + mat.m_paddedRow * x + y;
    float4 r;
    r.x = buffer[index];
    r.y = buffer[index + 1];
    r.z = buffer[index + 2];
    r.w = buffer[index + 3];
    return r;
}

DEVICE
void setElem4( int x, int y, float* buffer, GPUMat mat, float4 v )
{
    int index = mat.m_location + mat.m_paddedRow * x + y;
    buffer[index] =  v.x ;
    buffer[index + 1] = v.y;
    buffer[index + 2] = v.z;
    buffer[index + 3] = v.w;
}
DEVICE
inline float maxss( float a, float b )
{
    return ( b < a ) ? a : b;
}

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
struct MLPEncoding
{
    int mode;
    int frequency_N;
    int padd1;
    int padd2;
};

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

DEVICE
float getTensor( float* tensor, int xi, int yi )
{
    return tensor[xi * TENSOR_ROW + yi];
}

DEVICE
void setTensor( float* tensor, int xi, int yi, float value )
{
    tensor[xi * TENSOR_ROW + yi] = value;
}
const uint32_t PRIMES[7] = { 1, 2654435761, 805459861, 3674653429, 2097192037, 1434869437, 2165219737 };

struct DimensionHasher
{
    uint32_t m_h;

    DEVICE
    void init()
    {
        m_h = 0;
    }

    DEVICE
    void add( uint32_t xs, int d )
    {
        m_h ^= xs * PRIMES[ min( d, 6 ) ];
    }

    DEVICE
    uint32_t value() { return m_h; }
};

struct HashGridEvaluator
{
    int m_dim;
    uint32_t m_bits;

    DEVICE
    void init( int dim )
    {
        m_dim = dim;
        m_bits = 0xFFFFFFFF;
    }

    DEVICE
    bool moveNext()
    {
        m_bits++;
        return m_bits < ( 1U << m_dim );
    }

    DEVICE
    void evaluate( float* weight, uint32_t* hashValue, int resolution, float* input )
    {
        DimensionHasher hasher;
        hasher.init();

        float w = 1.0f;
        for( int d = 0; d < m_dim; ++d )
        {
            float x_in = input[ min( d, GRID_INPUT_DIM - 1 ) ];

            float xf = x_in * resolution;
            uint32_t xi = xf;
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
        *weight = w;
        *hashValue = hasher.value();
    }
};

extern "C" __global__ void forward( float* inputs, float* output, float* matBuffer, float* gridFeature, MLPForwardFusedArg mlpForwardFusedArg, MLPEncoding mlpEncoding ) 
{
    __shared__ float tensor[64 * TENSOR_ROW];

    int yi_global_base = blockIdx.x * TENSOR_ROW;
    int xi = threadIdx.y;

    float value[TENSOR_ROW];
    
    if( xi < mlpForwardFusedArg.inputMat.m_col )
    {
        float vs[TENSOR_ROW];
        for( int yi_local = 0 ; yi_local < TENSOR_ROW ; yi_local++ )
        {
            int yi = yi_global_base + yi_local;
            if( yi < mlpForwardFusedArg.inputMat.m_row )
            {
                vs[yi_local] = getElem( xi, yi, inputs, mlpForwardFusedArg.inputMat );
            }
            else
            {
                vs[yi_local] = 0.0f;
            }
        }
        for( int yi_local = 0 ; yi_local < TENSOR_ROW ; yi_local++ )
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
            for( int yi_local = 0 ; yi_local < TENSOR_ROW ; yi_local++ )
            {
                int xi_src = xi / ( mlpEncoding.frequency_N * 2 );
                int baseEachDim = xi % ( mlpEncoding.frequency_N * 2 );
                int i = baseEachDim / 2;
                int tri_idx = baseEachDim % 2;
                float v = getTensor( tensor, xi_src, yi_local );
                float k = 2.0f * PI * __powf( 2.0f, (float)i );
                v = __sinf( k * v + ( tri_idx ? PI * 0.5f : 0.0f ) );
                value[yi_local] = v;
            }
        }
        __syncthreads();
        if( xi < outputCol )
        {
            for( int yi_local = 0 ; yi_local < TENSOR_ROW ; yi_local++ )
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
        for( int yi_local = 0 ; yi_local < TENSOR_ROW ; yi_local++ )
        {
            float input[GRID_INPUT_DIM];
            for( int x = 0 ; x < GRID_INPUT_DIM ; x++ )
            {
                input[x] = getTensor( tensor, x, yi_local );
            }
            __syncthreads();

            if( level < GRID_L )
            {
                HashGridEvaluator evaluator;
                evaluator.init( GRID_INPUT_DIM );
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

        float bias = xi < col ? getElem( xi, 0, matBuffer, mlpForwardFusedArg.m_Bs[i] ) : 0.0f;
        for( int yi_local = 0 ; yi_local < TENSOR_ROW ; yi_local++ )
        {
            value[yi_local] = bias;
        }
        
        if( xi < col )
        {
            for( int j = 0 ; j < row ; j++ ) 
            {
                float b = getElem( xi /* output xi */, j, matBuffer, mlpForwardFusedArg.m_Ws[i] );
                for( int yi_local = 0 ; yi_local < TENSOR_ROW ; yi_local++ )
                {
                    float a = getTensor( tensor, j, yi_local );
                    value[yi_local] = fma( a, b, value[yi_local] );
                }
            }
            
            float lowerbounds = i + 1 != mlpForwardFusedArg.nLayer ? 0.0f : -3.40282e+38f;
            for( int yi_local = 0 ; yi_local < TENSOR_ROW ; yi_local++ )
            {
                value[yi_local] = fmaxf( value[yi_local], lowerbounds );
            }
        }

        __syncthreads();

        if( xi < col )
        {
            for( int yi_local = 0 ; yi_local < TENSOR_ROW ; yi_local++ )
            {
                setTensor( tensor, xi, yi_local, value[yi_local] );
            }
        }
        __syncthreads();
    }

    if( xi < mlpForwardFusedArg.outputMat.m_col )
    {
        for( int yi_local = 0 ; yi_local < TENSOR_ROW ; yi_local++ )
        {
            value[yi_local] = getTensor( tensor, xi, yi_local );
        }
        for( int yi_local = 0 ; yi_local < TENSOR_ROW ; yi_local++ )
        {
            int yi = yi_global_base + yi_local;
            if( yi < mlpForwardFusedArg.outputMat.m_row )
            {
                setElem( xi, yi, output, mlpForwardFusedArg.outputMat, value[yi_local]);
            }
        }
    }
}
