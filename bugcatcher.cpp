#define CATCH_CONFIG_MAIN
#include "catch.hpp"

#include "redpill.hpp"
using namespace rpml;
TEST_CASE( "misc", "[misc]" )
{
	REQUIRE( next_multiple( 0, 10 ) == 0 );
	REQUIRE( next_multiple( 4, 10 ) == 10 );
	REQUIRE( next_multiple( 10, 10 ) == 10 );
	REQUIRE( next_multiple( 11, 10 ) == 20 );
	REQUIRE( next_multiple( 65, 8 ) == 72 );
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
	Mat r = addVectorForEachRow( a, b );

    Mat truth = fromRowMajor( 2, 3, { 
        4, 4, 4, 
        7, 7, 7,
    });
	FOR_EACH_ELEMENT( truth, ix, iy )
	{
		REQUIRE( r( ix, iy ) == truth( ix, iy ) );
	}
}
TEST_CASE( "Mat vertialSum", "[Mat vertialSum]" )
{
    Mat a = fromRowMajor( 2, 3, { 
        1, 2, 3, 
        6, 5, 4,
    });
	Mat r = vertialSum( a );
	REQUIRE( r.row() == 1 );
	REQUIRE( r.col() == 3 );

	REQUIRE( r( 0, 0 ) == 7.0f );
	REQUIRE( r( 1, 0 ) == 7.0f );
	REQUIRE( r( 2, 0 ) == 7.0f );
}
TEST_CASE( "Mat sliceH", "[Mat sliceH]" )
{
    Mat a = fromRowMajor( 3, 3, { 
        1, 2, 3, 
        6, 5, 4,
        7, 8, 9,
    });
	Mat r = sliceH( a, 1, 2 );
	REQUIRE( r.row() == 1 );
	REQUIRE( r.col() == 3 );
	REQUIRE( r( 0, 0 ) == 6 );
	REQUIRE( r( 1, 0 ) == 5 );
	REQUIRE( r( 2, 0 ) == 4 );
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
	LayerContext context;
	Mat o = layer.forward( x, &context );

    REQUIRE( o( 0, 0 ) == 1 * 1 + 1 * 1 + 1 );
    REQUIRE( o( 1, 0 ) == 1 * 2 + 1 * 4 + 1 );
    REQUIRE( o( 2, 0 ) == 1 * 3 + 1 * 6 + 1 );

    Mat xs = fromRowMajor(3, 2, { 
        1, 1, // data 1
        2, 3, // data 2
        4, 5, // data 3
    });
	Mat os = layer.forward( xs, &context );

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

	LayerContext context0;
	LayerContext context1;
	LayerContext contextS;

    SigmoidLayer sigmoidLayer( 0, 0 );

	Mat x = fromRowMajor( 1, 2, { 1, 1 } );
	Mat o;
	Mat ref = fromRowMajor( 1, 1, { 10.0f } );

	auto evalMSE = [&]()
	{
		o = layer1.forward( sigmoidLayer.forward( layer0.forward( x, &context0 ), &contextS ), &context1 );
		float L = mse( o, ref );
		return L;
	};
	float mse = evalMSE();
	
    // back propagation
	layer0.setupPropagation();
	sigmoidLayer.setupPropagation();
	layer1.setupPropagation();
	layer0.backward( sigmoidLayer.backward( layer1.backward( mse_backward( o, ref ), &context1 ), &contextS ), &context0 );

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