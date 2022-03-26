﻿#include "pr.hpp"
#include <iostream>
#include <memory>

#define RPML_DISABLE_ASSERT
#include "redpill.hpp"
using namespace rpml;

//#include <json.hpp>
//#include <sciplot/sciplot.hpp>
//using namespace sciplot;


inline float nnAnd( float x0, float x1 )
{
    // 0.0 < x0 * w0 + x1 * w1 - b 
    float b = -0.75f;
    float w0 = 0.5f;
    float w1 = 0.5f;
    return 0.0f < x0 * w0 + x1 * w1 + b ? 1.0f : 0.0f;
}
inline float nnNand(float x0, float x1)
{
    // 0.0 < x0 * w0 + x1 * w1 - b 
    float b = 0.75f;
    float w0 = -0.5f;
    float w1 = -0.5f;
    return 0.0f < x0 * w0 + x1 * w1 + b ? 1.0f : 0.0f;
}

inline float nnOr(float x0, float x1)
{
    // 0.0 < x0 * w0 + x1 * w1 - b 
    float b = -0.25f;
    float w0 = 0.5f;
    float w1 = 0.5f;
    return 0.0f < x0 * w0 + x1 * w1 + b ? 1.0f : 0.0f;
}
inline float nnXor(float x0, float x1)
{
    return nnAnd(nnNand(x0, x1), nnOr(x0, x1));
}

float sigmoid(float x)
{
    return 1.0f / (1.0f + std::exp(-x));
}

inline float nnMax( const Mat& m )
{
    float v = -FLT_MAX;
    for( int iy = 0; iy < m.row(); iy++ )
    for( int ix = 0; ix < m.col(); ix++)
    {
        v = maxss( v, m(ix, iy) );
    }
    return v;
}

std::pair<int, int> nnMaxIndex( const Mat& m )
{
    std::pair<int, int> idx( -1, -1 );
    float v = -FLT_MAX;

    for (int iy = 0; iy < m.row(); iy++)
    for (int ix = 0; ix < m.col(); ix++)
    {
        if( v <= m( ix, iy ) )
        {
            idx = std::pair<int, int>(ix, iy);
            v = m(ix, iy);
        }
    }
    return idx;
}

Mat nnSoftmax( const Mat& m )
{
    // Mat r = m;
    float c = -nnMax( m );

    float denom = 0.0f;
    for( int iy = 0; iy < m.row(); iy++ )
    for( int ix = 0; ix < m.col(); ix++)
    {
        float k = m(ix, iy) + c;
        denom += std::exp( k );
    }

    Mat r( m.row(), m.col() );
    for( int iy = 0; iy < m.row(); iy++ )
    for( int ix = 0; ix < m.col(); ix++)
    {
        float k = m(ix, iy) + c;
        r(ix, iy) = std::exp( k ) / denom;
    }
    return r;
}

void print( const Mat& m )
{
    for (int iy = 0; iy < m.row(); iy++)
    {
        printf("{ ");
        for (int ix = 0; ix < m.col(); ix++)
        {
            printf("{ %f, ", m(ix, iy));
        }
        printf("} \n");
    }
}

#if 1
int main()
{
	using namespace pr;

	SetDataDir( ExecutableDir() );

	Image2DRGBA8 image;
	//image.load( "img/small_albert.jpg" );
	image.load( "img/coyote.jpg" );
	Image2DRGBA8 estimatedImage;
	estimatedImage.allocate( image.width(), image.height() );

	ITexture *texture = CreateTexture();

	MLP mlp( MLPConfig()
				 .shape( { 2, 64, 64, 3 } )
				 .learningRate( 0.01f )
				 .initType( InitializationType::He )
				 .optimType( OptimizerType::Adam )
				 .activationType( ActivationType::ReLU )
				 .encoderType( EncoderType::Frequency ) );


	Config config;
	config.ScreenWidth = 1500;
	config.ScreenHeight = 1200;
	config.SwapInterval = 0;
	Initialize( config );

	Camera3D camera;
	camera.origin = { 0, 0, 4 };
	camera.lookat = { 0, 0, 0 };
	camera.zUp = false;

	double e = GetElapsedTime();

	static float learning = 0.005f;

	while( pr::NextFrame() == false )
	{
		if( IsImGuiUsingMouse() == false )
		{
			UpdateCameraBlenderLike( &camera );
		}

		ClearBackground( 0.1f, 0.1f, 0.1f, 1 );

		BeginCamera( camera );

		PushGraphicState();

		DrawGrid( GridAxis::XY, 1.0f, 10, { 128, 128, 128 } );
		DrawXYZAxis( 1.0f );

		// Batch
		static StandardRng rng;
		float loss = 0;
		int NData = 256 * 16;
		static Mat inputs( NData, 2 );
		static Mat refs( NData, 3 );

		Stopwatch sw_train;
		for( int j = 0; j < 100; ++j )
		{
			for( int i = 0; i < NData; ++i )
			{
				// if it's freq then need to be carefull range of x?
				float u = rng.draw();
				float v = rng.draw(); 
				int ui = (int)glm::mix( 0.0f, (float)image.width(), u );
				int vi = (int)glm::mix( 0.0f, (float)image.height(), v );
				glm::uvec3 y = image( ui, vi );
				inputs( 0, i ) = u;
				inputs( 1, i ) = v;
				refs( 0, i ) = y.x / 255.0f;
				refs( 1, i ) = y.y / 255.0f;
				refs( 2, i ) = y.z / 255.0f;
			}
			loss = mlp.train( inputs, refs );
		}
		float sTrained = sw_train.elapsed();


		static Mat inUVs( estimatedImage.width() * estimatedImage.height(), 2 );
		static Mat outPixels;

		Stopwatch sw_estimate;

		for( int yi = 0; yi < estimatedImage.height(); yi++ )
		{
			for( int xi = 0; xi < estimatedImage.width(); xi++ )
			{
				int i = yi * estimatedImage.width() + xi;
				inUVs( 0, i ) = ( xi + 0.5f ) / image.width();
				inUVs( 1, i ) = ( yi + 0.5f ) / image.height();
			}
		}

		mlp.forwardMT( &outPixels, inUVs );

		float sEstimate = sw_estimate.elapsed();

		for( int yi = 0; yi < estimatedImage.height(); yi++ )
		{
			for( int xi = 0; xi < estimatedImage.width(); xi++ )
			{
				int i = yi * estimatedImage.width() + xi;
				estimatedImage( xi, yi ) = glm::uvec4(
					glm::clamp<int>( outPixels( 0, i ) * 255.0f, 0, 255 ),
					glm::clamp<int>( outPixels( 1, i ) * 255.0f, 0, 255 ),
					glm::clamp<int>( outPixels( 2, i ) * 255.0f, 0, 255 ),
					255
				);
			}
		}
		texture->upload( estimatedImage );

		PopGraphicState();
		EndCamera();

		BeginImGui();

		ImGui::SetNextWindowSize( { 1200, 1200 }, ImGuiCond_Once );
		ImGui::Begin( "Panel" );
		ImGui::Text( "fps = %f", GetFrameRate() );
		ImGui::Text( "mse = %.10f", loss / NData );
		ImGui::Text( "%f s train", sTrained );
		ImGui::Text( "%f s estimate", sEstimate );
		
		static float scale = 1.0f;
		ImGui::SliderFloat( "scale", &scale, 0, 1 );
		float w = texture->width() * scale;
		float h = texture->height() * scale;
		ImGui::Image( texture, ImVec2( w, h ) );

		ImGui::End();

		EndImGui();
	}

	pr::CleanUp();
}
#endif

#if 0

float f( float x )
{
	return 0.5f + sin( x * 3.5f * glm::pi<float>() ) * 0.25f;
	// x = 4.f * glm::pi<float>() * x;
	// return sin( x ) * 0.3f + 0.5f;
};
int main() {
    using namespace pr;

    SetDataDir(ExecutableDir());

	MLP mlp( MLPConfig()
		.shape( { 1, 64, 64, 1 } )
        .learningRate( 0.01f )
        .initType( InitializationType::He )
		.optimType( OptimizerType::Adam )
		.activationType( ActivationType::ReLU )
        .encoderType( EncoderType::Frequency )
    );


	//for( ;; )
	//{
	//	static StandardRng rng;
	//	Mat ma( 129, 129 );
	//	Mat mb( 129, 129 );

	//	float p = 0.0f;

	//	Stopwatch sw;

	//	for( int i = 0; i < 1000; ++i )
	//	{
	//		FOR_EACH_ELEMENT( ma, ix, iy )
	//		{
	//			ma( ix, iy ) = glm::mix( -1.2f, 1.2f, rng.draw() );
	//			mb( ix, iy ) = glm::mix( -1.2f, 1.2f, rng.draw() );
	//		}
	//		Mat m = ma * mb;

	//		p += m( 64, 64 );
	//	}
	//	printf( "%f s ---- %f\n", sw.elapsed(), p );
	//}

	Config config;
	config.ScreenWidth = 1920;
	config.ScreenHeight = 1080;
	config.SwapInterval = 0;
	Initialize( config );

	Camera3D camera;
	camera.origin = { 0, 0, 4 };
	camera.lookat = { 0, 0, 0 };
	camera.zUp = false;

	double e = GetElapsedTime();

    static float learning = 0.005f;

	while( pr::NextFrame() == false )
	{
		if( IsImGuiUsingMouse() == false )
		{
			UpdateCameraBlenderLike( &camera );
		}

		ClearBackground( 0.1f, 0.1f, 0.1f, 1 );

		BeginCamera( camera );

		PushGraphicState();

		DrawGrid( GridAxis::XY, 1.0f, 10, { 128, 128, 128 } );
		DrawXYZAxis( 1.0f );

        // Batch 
        static StandardRng rng;
		float loss = 0;
		int NData = 256 * 64;
		static Mat inputs( NData, 1 );
		static Mat refs( NData, 1 );

        Stopwatch sw;
        for(int j = 0 ; j < 10 ; ++j)
		{
		    for( int i = 0; i < NData; ++i )
		    {
                // if it's freq then need to be carefull range of x?
			    float x = glm::mix( 0.0f, 1.0f, rng.draw() );
			    float y = f( x );
			    inputs( 0, i ) = x;
			    refs( 0, i ) = y;
		    }
			loss = mlp.train( inputs, refs );
        }
		float msTrained = sw.elapsed();

        int N = 512;
		LinearTransform i2x( 0, N, 0, 1 );
		PrimBegin( PrimitiveMode::LineStrip );
		for( int i = 0; i < N ; ++i)
		{
			float x = i2x( i );
			float y = f( x );
			PrimVertex( { x, y, 0 }, { 255, 255, 255 } );
        }
		PrimEnd();


        static Mat estimateInputs( N, 1 );
		static Mat estimated;

		for( int i = 0; i < N; ++i )
		{
			float x = i2x( i );
			estimateInputs( 0, i ) = x;
		}
		mlp.forward( &estimated, estimateInputs );
		PrimBegin( PrimitiveMode::LineStrip, 2 );
		for( int i = 0; i < N; ++i )
		{
			float x = estimateInputs( 0, i );
			float y = estimated( 0, i );
			PrimVertex( { x, y, 0 }, { 255, 0, 0 } );
		}
		PrimEnd();

		PopGraphicState();
		EndCamera();

		BeginImGui();

		ImGui::SetNextWindowSize( { 500, 800 }, ImGuiCond_Once );
		ImGui::Begin( "Panel" );
		ImGui::Text( "fps = %f", GetFrameRate() );
		ImGui::Text( "mse = %.10f", loss / NData );
		ImGui::Text( "%f s train", msTrained );
		ImGui::End();

		EndImGui();
	}

	pr::CleanUp();

}

#endif