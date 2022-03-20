#include "pr.hpp"
#include <iostream>
#include <memory>

//#define RPML_DISABLE_ASSERT
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

//Mat fromJson( const nlohmann::json& j )
//{
//    int row = j["row"].get<int>();
//    int col = j["col"].get<int>();
//    std::vector<float> data = j["data"].get<std::vector<float>>();
//    return Mat(row, col, data);
//}

float f( float x )
{
	return 0.5f + sin( x * 2.5f * glm::pi<float>() ) * 0.25f;
};
int main() {
    using namespace pr;

    SetDataDir(ExecutableDir());

#if 1

    MLP mlp( { 1, 32, 32, 32, 1 }, 0.005f );

  //  auto rng = new StandardRng();
  //  BoxMular box( rng );
  //  for (int i = 0; i < 10000 ; ++i )
  //  {
  //      printf("%f\n", box.draw());
		////printf( "%f\n", rng->draw() );
  //  }

	Xoshiro128StarStar random;
	AffineLayer layer0( 1, 48, OptimizerType::SGD, 0.05f );
	ReLULayer sigmoidLayer( 48, 48 );
	AffineLayer layer1( 48, 48, OptimizerType::SGD, 0.05f );
	ReLULayer sigmoidLayer2( 48, 48 );
	AffineLayer layer2( 48, 1, OptimizerType::SGD, 0.05f );

    StandardRng rng;

    FOR_EACH_ELEMENT( layer0.m_W, ix, iy )
	{
		layer0.m_W( ix, iy ) = glm::mix( -1.0f, 1.0f, rng.draw() );
	}
	FOR_EACH_ELEMENT( layer0.m_b, ix, iy )
	{
		layer0.m_b( ix, iy ) = glm::mix( -1.0f, 1.0f, rng.draw() );
	}
	FOR_EACH_ELEMENT( layer1.m_W, ix, iy )
	{
		layer1.m_W( ix, iy ) = glm::mix( -1.0f, 1.0f, rng.draw() );
	}
	FOR_EACH_ELEMENT( layer1.m_b, ix, iy )
	{
		layer1.m_b( ix, iy ) = glm::mix( -1.0f, 1.0f, rng.draw() );
	}
	FOR_EACH_ELEMENT( layer2.m_W, ix, iy )
	{
		layer2.m_W( ix, iy ) = glm::mix( -1.0f, 1.0f, rng.draw() );
	}
	FOR_EACH_ELEMENT( layer2.m_b, ix, iy )
	{
		layer2.m_b( ix, iy ) = glm::mix( -1.0f, 1.0f, rng.draw() );
	}

	Config config;
	config.ScreenWidth = 1920;
	config.ScreenHeight = 1080;
	config.SwapInterval = 1;
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
        int NData = 1024; // super naiive
        Mat inputs( NData, 1 );
		Mat refs( NData, 1 );
		for( int i = 0; i < NData; ++i )
		{
			float x = glm::mix( -0.2f, 1.2f, random.uniformf() );
			float y = f( x );
			inputs( 0, i ) = x;
			refs( 0, i ) = y;
		}
		float loss = mlp.train( inputs, refs );
#if 0
		MeanSquaredErrorLayer mseLayer;
		mseLayer.m_y = refs;

        Mat m = inputs;
		m = layer0.forward( m );
		m = sigmoidLayer.forward( m );
		m = layer1.forward( m );
		m = sigmoidLayer2.forward( m );
		m = layer2.forward( m );
		Mat mse = mseLayer.forward( m );

        Mat bm;
		bm = mseLayer.backward( bm );
		bm = layer2.backward( bm );
		bm = sigmoidLayer2.backward( bm );
		bm = layer1.backward( bm );
		bm = sigmoidLayer.backward( bm );
		bm = layer0.backward( bm );

        // learn
		float s = learning / NData;
		layer0.m_b = layer0.m_b - multiplyScalar( layer0.m_db, s );
		layer0.m_W = layer0.m_W - multiplyScalar( layer0.m_dW, s );
		layer1.m_b = layer1.m_b - multiplyScalar( layer1.m_db, s );
		layer1.m_W = layer1.m_W - multiplyScalar( layer1.m_dW, s );
		layer2.m_b = layer2.m_b - multiplyScalar( layer2.m_db, s );
		layer2.m_W = layer2.m_W - multiplyScalar( layer2.m_dW, s );

#endif
#if 0
        Mat layer0_db( layer0.m_b.row(), layer0.m_b.col() );
		Mat layer0_dW( layer0.m_W.row(), layer0.m_W.col() );
        Mat layer1_db( layer1.m_b.row(), layer1.m_b.col() );
		Mat layer1_dW( layer1.m_W.row(), layer1.m_W.col() );

        float mseCur = 0;
		int NData = 1000; // super naiive
		for( int i = 0; i < NData; ++i )
		{
            float x = glm::mix( -0.2f, 1.2f, random.uniformf() );
		    float y = f( x );
            MeanSquaredErrorLayer mseLayer;
		    mseLayer.m_y = fromRowMajor( 1, 1, { y } );

            Mat input = fromRowMajor( 1, 1, { x } );
		    Mat o = layer1.forward( sigmoidLayer.forward( layer0.forward( input ) ) );
		    Mat mse = mseLayer.forward( o );
			mseCur += mse( 0, 0 );

            Mat k = fromRowMajor( 1, 1, { 0.05f * 1.0f / NData } );
            layer0.backward( sigmoidLayer.backward( layer1.backward( mseLayer.backward( k ) ) ) );

            layer0_db = layer0_db + layer0.m_db;
			layer0_dW = layer0_dW + layer0.m_dW;
			layer1_db = layer1_db + layer1.m_db;
			layer1_dW = layer1_dW + layer1.m_dW;
        }
		// learn
		layer0.m_b = layer0.m_b - layer0_db;
		layer0.m_W = layer0.m_W - layer0_dW;
		layer1.m_b = layer1.m_b - layer1_db;
		layer1.m_W = layer1.m_W - layer1_dW;
#endif

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


        Mat estimateInputs( N, 1 );
		for( int i = 0; i < N; ++i )
		{
			float x = i2x( i );
			estimateInputs( 0, i ) = x;
		}
		Mat estimated = mlp.forward( estimateInputs );
		PrimBegin( PrimitiveMode::LineStrip, 2 );
		for( int i = 0; i < N; ++i )
		{
			float x = estimateInputs( 0, i );
			float y = estimated( 0, i );
			PrimVertex( { x, y, 0 }, { 255, 0, 0 } );
		}
		PrimEnd();

		//PrimBegin( PrimitiveMode::LineStrip, 2 );
		//for( int i = 0; i < N; ++i )
		//{
		//	float x = i2x( i );
		//	Mat input = fromRowMajor( 1, 1, { x } );
		//	Mat m = mlp.forward( input );
		//	//Mat m = input;
		//	//m = layer0.forward( m );
		//	//m = sigmoidLayer.forward( m );
		//	//m = layer1.forward( m );
		//	//m = sigmoidLayer2.forward( m );
		//	//m = layer2.forward( m );
		//	float y = m( 0, 0 );
		//	PrimVertex( { x, y, 0 }, { 255, 0, 0 } );
		//}
		//PrimEnd();

		PopGraphicState();
		EndCamera();

		BeginImGui();

		ImGui::SetNextWindowSize( { 500, 800 }, ImGuiCond_Once );
		ImGui::Begin( "Panel" );
		ImGui::Text( "fps = %f", GetFrameRate() );
		ImGui::Text( "mse = %.5f", loss / NData );
		ImGui::InputFloat( "learning", &learning, 0.1f);

		ImGui::End();

		EndImGui();
	}

	pr::CleanUp();
#endif

#if 0
    nlohmann::json sample_weight;
    {
        std::ifstream ifs(GetDataPath("sample_weight.json"));
        ifs >> sample_weight;
    }

    Mat w1 = fromJson(sample_weight["W1"]);
    Mat w2 = fromJson(sample_weight["W2"]);
    Mat w3 = fromJson(sample_weight["W3"]);
    Mat b1 = fromJson(sample_weight["b1"]);
    Mat b2 = fromJson(sample_weight["b2"]);
    Mat b3 = fromJson(sample_weight["b3"]);

    Mat x(1, 2, { 1.0f, 0.5f });
    Mat y(2, 3, { 0.1f, 0.2f, 0.3f, 0.4f, 0.5f, 0.6f });
    Mat b(1, 3, { 0.1f, 0.2f, 0.3f });

    //Mat x(2, 2, { 1, 3, 2, 4 });
    //Mat y(2, 2, { 5, 7, 6, 8 });
    Mat a = apply( x * y + b, sigmoid );

    print(x);
    printf("\n");
    print(y);
    printf("\n");
    print(a);

    auto bitpair = { std::make_pair(0, 0),  std::make_pair(0, 1), std::make_pair(1, 0), std::make_pair(1, 1) };

    printf("nnAnd 0, 0 => %f\n", nnAnd(0, 0));
    printf("nnAnd 1, 0 => %f\n", nnAnd(1, 0));
    printf("nnAnd 0, 1 => %f\n", nnAnd(0, 1));
    printf("nnAnd 1, 1 => %f\n", nnAnd(1, 1));

    printf("\n");

    printf("nnNand 0, 0 => %f\n", nnNand(0, 0));
    printf("nnNand 1, 0 => %f\n", nnNand(1, 0));
    printf("nnNand 0, 1 => %f\n", nnNand(0, 1));
    printf("nnNand 1, 1 => %f\n", nnNand(1, 1));

    printf("\n");

    for (auto b : bitpair)
    {
        printf("nnOr %d, %d => %f\n", b.first, b.second, nnOr(b.first, b.second));
    }

    printf("\n");

    for( auto b : bitpair )
    {
        printf("nnXor %d, %d => %f\n", b.first, b.second, nnXor(b.first, b.second));
    }

    //std::vector<float> xs(100);
    //std::vector<float> ys(100);
    //LinearTransform i2x(0, 100, -5, 5);
    //for (int i = 0; i < 100; i++)
    //{
    //    xs[i] = i2x(i);
    //    ys[i] = sigmoid(xs[i]);
    //}

    //Plot plot;
    //// Plot the Bessel functions                                                                                        
    //plot.drawCurve(xs, ys).label("y");

    //// Show the plot in a pop-up window
    //plot.show();

    Config config;
    config.ScreenWidth = 1920;
    config.ScreenHeight = 1080;
    config.SwapInterval = 1;
    Initialize(config);

    Camera3D camera;
    camera.origin = { 4, 4, 4 };
    camera.lookat = { 0, 0, 0 };
    camera.zUp = true;

    double e = GetElapsedTime();

    std::vector<Image2DMono8> imgs(10);
    for (int i = 0; i < 10; i++)
    {
        char name[128];
        sprintf(name, "img/img%02d.png", i);
        imgs[i].load(name);
    }
    ITexture *imgTex = CreateTexture();
    int imgIndex = 0;

    while (pr::NextFrame() == false) {
        if (IsImGuiUsingMouse() == false) {
            UpdateCameraBlenderLike(&camera);
        }

        ClearBackground(0.1f, 0.1f, 0.1f, 1);

        BeginCamera(camera);

        PushGraphicState();

        DrawGrid(GridAxis::XY, 1.0f, 10, { 128, 128, 128 });
        DrawXYZAxis(1.0f);

        PopGraphicState();
        EndCamera();

        BeginImGui();

        ImGui::SetNextWindowSize({ 500, 800 }, ImGuiCond_Once);
        ImGui::Begin("Panel");
        ImGui::Text("fps = %f", GetFrameRate());

        ImGui::InputInt("index", &imgIndex);
        imgIndex = glm::clamp(imgIndex, 0, 9);
        imgTex->upload(imgs[imgIndex]);
        ImGui::Image(imgTex, ImVec2(100, 100));

        const Image2DMono8 &src = imgs[imgIndex];
        Mat x( 1, 28 * 28 );
        for( int i = 0; i < src.height(); i++ )
        {
            for (int j = 0; j < src.width(); j++)
            {
				x( j * 28 + i, 0 ) = (float)src( i, j ) / 255.0f;
            }
        }

        Mat a = x;
        a = apply( a * w1 + b1, sigmoid );
        a = apply( a * w2 + b2, sigmoid );
        a = apply( a * w3 + b3, sigmoid );
        Mat result = nnSoftmax( a );
        int theIndex = nnMaxIndex(result).first;

        for (int i = 0; i < 10; i++)
        {
            ImVec4 white(1, 1, 1, 1);
            ImVec4 red(1, 0, 0, 1);
            ImGui::TextColored(i == theIndex ? red : white, "[%d] = %.4f", i, result(i, 0));
        }

        ImGui::End();

        EndImGui();
    }

    pr::CleanUp();

#endif
}
