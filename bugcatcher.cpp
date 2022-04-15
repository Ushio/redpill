#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "redpill.hpp"
using namespace rpml;

constexpr float abs_constant( float x )
{
	return x < 0.0f ? -x : x;
}
constexpr float newton_sqrt_r( float xn, float a, int e )
{
	float xnp1 = xn - ( xn * xn - a ) * 0.5f / xn;
	float e0 = abs_constant( xn * xn - a );
	float e1 = abs_constant( xnp1 * xnp1 - a );
	return ( e1 < e0 )
			   ? newton_sqrt_r( xnp1, a, e )
			   : ( e < 4 /* magic number */ ? newton_sqrt_r( xnp1, a, e + 1 ) : xn );
}
constexpr float newton_sqrt( float x )
{
	bool valid =
		0.0f <= x &&
		x < std::numeric_limits<float>::infinity() &&
		x == x; // nan
	return valid
			   ? ( x == 0.0f ? 0.0f : newton_sqrt_r( x, x, 0 ) )
			   : std::numeric_limits<double>::quiet_NaN();
}
TEST_CASE( "misc", "[misc]" )
{
	REQUIRE( next_multiple( 0, 10 ) == 0 );
	REQUIRE( next_multiple( 4, 10 ) == 10 );
	REQUIRE( next_multiple( 10, 10 ) == 10 );
	REQUIRE( next_multiple( 11, 10 ) == 20 );
	REQUIRE( next_multiple( 65, 8 ) == 72 );

	pr::ThreadPool pool( std::thread::hardware_concurrency() );

	for( int i = 0; i < 100; i++ )
	{
		int N = 8388608;
		float v = 0.0f;

		pr::TaskGroup g;
		g.addElements( N );
		pool.enqueueFor( N, 1000, [&v, &g]( int64_t beg, int64_t end )
		{
			atomAdd( &v, end - beg );
			g.doneElements( end - beg );
		} );
		g.waitForAllElementsToFinish();

		REQUIRE( v == N );
	}


	for(int i = 0 ; i < 100 ; i++ ) {
		int N = 8388608;
		std::atomic<float> v = 0.0f;

		pr::TaskGroup g;
		g.addElements( N );
		pool.enqueueFor( N, 1000, [&v, &g]( int64_t beg, int64_t end )
		{
			atomAdd( &v, end - beg );
			g.doneElements( end - beg );
		} );
		g.waitForAllElementsToFinish();

		REQUIRE( v == N );
	}

	static_assert( newton_sqrt( 0.0f ) == 0.0f, "" );
	REQUIRE( std::isnan( newton_sqrt( std::numeric_limits<float>::infinity() ) ) );
	REQUIRE( std::isnan( newton_sqrt( -1.0f ) ) );

	StandardRng rng;
	for( int i = 0; i < 1000000; i++ )
	{
		float x = rng.draw() * 1000.0f;
		REQUIRE( Approx( std::sqrt( x ) ).margin( 0.0000001f ) == newton_sqrt( x ) );
	}
}

TEST_CASE("Mat transpose", "[transpose]") {
    Mat a = fromColMajor(2, 3, { 1, 4, 2, 5, 3, 6 });
    Mat b = fromRowMajor(2, 3, { 1, 2, 3, 4, 5, 6 });

    FOR_EACH_ELEMENT( a, ix, iy )
    {
        REQUIRE( a(ix, iy) == b(ix, iy) );
    }
}

TEST_CASE("Mat multiply", "[multiply]") {
    Mat a = fromRowMajor(2, 3, { 
        1, 3, -2, 
        -1, 5, 4,
    });
    Mat b = fromRowMajor(3, 3, {
        3, -2, 4,
       -4, 1, 2,
       2, 3, -1
    });
    Mat truth = fromRowMajor(2, 3, {
       -13, -5, 12,
       -15, 19, 2
    });

    Mat ab;
	mul( &ab, a, b );

    FOR_EACH_ELEMENT( truth, ix, iy )
    {
		REQUIRE( ab( ix, iy ) == truth( ix, iy ) );
    }
}
TEST_CASE( "Mat add", "[Mat add]" )
{
    Mat a = fromRowMajor( 2, 3, { 
        1, 2, 3, 
        4, 5, 6,
    });
	Mat b = fromRowMajor( 1, 3, { 3, 2, 1 } );
	addVectorForEachRow( &a, b );

    Mat truth = fromRowMajor( 2, 3, { 
        4, 4, 4, 
        7, 7, 7,
    });
	FOR_EACH_ELEMENT( truth, ix, iy )
	{
		REQUIRE( a( ix, iy ) == truth( ix, iy ) );
	}
}
TEST_CASE( "Mat vertialSum", "[Mat vertialSum]" )
{
    Mat a = fromRowMajor( 2, 3, { 
        1, 2, 3, 
        6, 5, 4,
    });
	Mat r;
	vertialSum( &r, a );
	REQUIRE( r.row() == 1 );
	REQUIRE( r.col() == 3 );

	REQUIRE( r( 0, 0 ) == 7.0f );
	REQUIRE( r( 1, 0 ) == 7.0f );
	REQUIRE( r( 2, 0 ) == 7.0f );
}
TEST_CASE( "Mat sliceH", "[Mat sliceH]" )
{
    Mat a = fromRowMajor( 4, 3, { 
        1, 2, 3, 
        6, 5, 4,
        7, 8, 9,
		0, 1, 2
    });
	Mat s1;
	sliceH( &s1, a, 1, 2 );

	Mat s0, s2;
	sliceH( &s0, a, 0, 1 );
	sliceH( &s2, a, 2, 4 );

	REQUIRE( s1.row() == 1 );
	REQUIRE( s1.col() == 3 );
	REQUIRE( s1( 0, 0 ) == 6 );
	REQUIRE( s1( 1, 0 ) == 5 );
	REQUIRE( s1( 2, 0 ) == 4 );

	REQUIRE( s0( 0, 0 ) == 1 );
	REQUIRE( s2( 2, 1 ) == 2 );

	Mat b( 4, 3 );
	concatH( &b, s0, 0, 1 );
	concatH( &b, s1, 1, 2 );
	concatH( &b, s2, 2, 4 );

	FOR_EACH_ELEMENT( a, ix, iy )
	{
		REQUIRE( a( ix, iy ) == b( ix, iy ) );
	}
}
#if ENABLE_SIMD
TEST_CASE( "Mat simd", "[Mat simd]" )
{
	StandardRng rng;
	Mat ma( 129, 321 );
	Mat mb( 321, 129 );

	for( int i = 0; i < 10; ++i )
	{
		FOR_EACH_ELEMENT( ma, ix, iy )
		{
			ma( ix, iy ) = rng.draw();
		}
		FOR_EACH_ELEMENT( mb, ix, iy )
		{
			mb( ix, iy ) = rng.draw();
		}
		Mat m;
		mulNaive( &m, ma, mb );
		Mat m_simd;
		mulSIMD( &m_simd, ma, mb );
		FOR_EACH_ELEMENT( m, ix, iy )
		{
			REQUIRE( m( ix, iy ) == Approx( m_simd( ix, iy ) ).margin( 0.0000001f ) );
		}
	}
}
#endif

TEST_CASE("AffineLayer foward", "[affine foward]") 
{
    AffineLayer layer( 2, 3, OptimizerType::SGD, 0.01f );
    layer.m_W = fromRowMajor( 2, 3, {
        1, 2, 3,
        1, 4, 6
    } );
    layer.m_b = fromRowMajor(1, 3, { 1, 1, 1 });

    Mat x = fromRowMajor(1, 2, { 1, 1 } );
	MatContext context;
	Mat o;
	layer.forward( &o, x, &context );

    REQUIRE( o( 0, 0 ) == 1 * 1 + 1 * 1 + 1 );
    REQUIRE( o( 1, 0 ) == 1 * 2 + 1 * 4 + 1 );
    REQUIRE( o( 2, 0 ) == 1 * 3 + 1 * 6 + 1 );

    Mat xs = fromRowMajor(3, 2, { 
        1, 1, // data 1
        2, 3, // data 2
        4, 5, // data 3
    });
	Mat os;
	layer.forward( &os, xs, &context );

    REQUIRE( os( 0, 0 ) == 1 * 1 + 1 * 1 + 1 );
	REQUIRE( os( 1, 0 ) == 1 * 2 + 1 * 4 + 1 );
	REQUIRE( os( 2, 0 ) == 1 * 3 + 1 * 6 + 1 );
    
    REQUIRE( os( 0, 1 ) == 2 * 1 + 3 * 1 + 1 );
	REQUIRE( os( 1, 1 ) == 2 * 2 + 3 * 4 + 1 );
	REQUIRE( os( 2, 1 ) == 2 * 3 + 3 * 6 + 1 );

    REQUIRE( os( 0, 2 ) == 4 * 1 + 5 * 1 + 1 );
	REQUIRE( os( 1, 2 ) == 4 * 2 + 5 * 4 + 1 );
	REQUIRE( os( 2, 2 ) == 4 * 3 + 5 * 6 + 1 );
}

TEST_CASE( "AffineLayer backward", "[affine backward]" )
{
	AffineLayer layer0( 2, 3, OptimizerType::SGD, 0.01f );
	layer0.m_W = fromRowMajor( 2, 3, { 1, 2, 3, 1, 4, 6 } );
	layer0.m_b = fromRowMajor( 1, 3, { 1, 1, 1 } );

    AffineLayer layer1( 3, 1, OptimizerType::SGD, 0.01f );
	layer1.m_W = fromRowMajor( 3, 1, { 0.1, 0.3, 0.8 } );
	layer1.m_b = fromRowMajor( 1, 1, { 3 } );

	MatContext context0;
	MatContext context1;
	MatContext contextS;

    SigmoidLayer sigmoidLayer( 0, 0 );

	Mat x = fromRowMajor( 1, 2, { 1, 1 } );
	Mat o;
	Mat ref = fromRowMajor( 1, 1, { 10.0f } );

	auto evalMSE = [&]()
	{
		Mat input = x;
		Mat output;
		layer0.forward( &output, input, &context0 ); input.swap( output );
		sigmoidLayer.forward( &output, input, &contextS ); input.swap( output );
		layer1.forward( &output, input, &context1 ); input.swap( output );
		o = input;
		float L = mse( input, ref );
		return L;
	};
	float mse = evalMSE();
	
    // back propagation

	Mat input = o;
	sub( &input, ref ); // mse backward
	Mat output;
	layer1.backward( &output, input, &context1 ); input.swap( output );
	sigmoidLayer.backward( &output, input, &contextS ); input.swap( output );
	layer0.backward( &output, input, &context0 ); input.swap( output );

    // calc numerical derivatives
    Mat dW_numerical( layer0.m_W.row(), layer0.m_W.col() );
	FOR_EACH_ELEMENT( layer0.m_W, ix, iy )
	{
		float h = 0.01f;
        float x = layer0.m_W( ix, iy );
		float x0 = x - h * 0.5f;
		float x1 = x + h * 0.5f;

        layer0.m_W( ix, iy ) = x0;
		float y0 = evalMSE();
		layer0.m_W( ix, iy ) = x1;
		float y1 = evalMSE();
		layer0.m_W( ix, iy ) = x;
		dW_numerical( ix, iy ) = ( y1 - y0 ) / h;
	}

    FOR_EACH_ELEMENT( dW_numerical, ix, iy )
	{
		REQUIRE( layer0.m_dW( ix, iy ) == Approx( dW_numerical( ix, iy ) ).margin( 0.05f ) );
    }

    // calc numerical derivatives
	Mat db_numerical( layer0.m_b.row(), layer0.m_b.col() );
	FOR_EACH_ELEMENT( layer0.m_b, ix, iy )
	{
		float h = 0.01f;
		float x = layer0.m_b( ix, iy );
		float x0 = x - h * 0.5f;
		float x1 = x + h * 0.5f;

		layer0.m_b( ix, iy ) = x0;
		float y0 = evalMSE();
		layer0.m_b( ix, iy ) = x1;
		float y1 = evalMSE();
		layer0.m_b( ix, iy ) = x;
		db_numerical( ix, iy ) = ( y1 - y0 ) / h;
	}
	FOR_EACH_ELEMENT( db_numerical, ix, iy )
	{
		REQUIRE( layer0.m_db( ix, iy ) == Approx( db_numerical( ix, iy ) ).margin( 0.05f ) );
	}
}