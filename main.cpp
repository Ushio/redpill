#include "pr.hpp"
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
	MLP mlp( MLPConfig()
		.shape( { 1, 128, 128, 1 } )
        .learningRate( 0.05f )
        .initType( InitializationType::He )
		.activationType( ActivationType::ReLU )
    );

  //  auto rng = new StandardRng();
  //  BoxMular box( rng );
  //  for (int i = 0; i < 10000 ; ++i )
  //  {
  //      printf("%f\n", box.draw());
		////printf( "%f\n", rng->draw() );
  //  }


 //   for( ;; )
	//{
	//	static StandardRng rng;
	//	Mat ma( 129, 129 );
	//	Mat mb( 129, 129 );
 //       
 //       float p = 0.0f;

 //       Stopwatch sw;

	//	for(int i = 0 ; i < 1000; ++i)
	//	{
	//		FOR_EACH_ELEMENT( ma, ix, iy )
	//		{
	//			ma( ix, iy ) = glm::mix( -1.2f, 1.2f, rng.draw() );
	//			mb( ix, iy ) = glm::mix( -1.2f, 1.2f, rng.draw() );
 //           }
	//		Mat m = ma * mb;

 //           p += m( 64, 64 );
 //       }
 //       printf( "%f s ---- %f\n", sw.elapsed(), p );
	//}

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
        StandardRng rng;
		int NData = 256 * 128; // super naiive
        Mat inputs( NData, 1 );
		Mat refs( NData, 1 );
		for( int i = 0; i < NData; ++i )
		{
			float x = glm::mix( -0.1f, 1.1f, rng.draw() );
			float y = f( x );
			inputs( 0, i ) = x;
			refs( 0, i ) = y;
		}
		float loss = mlp.train( inputs, refs );

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
