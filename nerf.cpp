#include "pr.hpp"
#include <iostream>
#include <memory>
#include <fstream>

#define RPML_DISABLE_ASSERT
#include "redpill.hpp"
#include "redpillg.hpp"
using namespace rpml;

#if defined( DrawText )
#undef DrawText
#endif

#include <json.hpp>

#define LINEAR_SPACE_LEARNING 0

struct NerfCamera
{
	glm::mat4 transform;
	float fovy;
	std::string path;
	pr::Image2DRGBA8 image;
};

glm::mat4 loadMatrix( nlohmann::json j )
{
	nlohmann::json row0 = j[0];
	nlohmann::json row1 = j[1];
	nlohmann::json row2 = j[2];
	nlohmann::json row3 = j[3];
	glm::mat4 m = glm::mat4(
		row0[0].get<float>(), row0[1].get<float>(), row0[2].get<float>(), row0[3].get<float>(),
		row1[0].get<float>(), row1[1].get<float>(), row1[2].get<float>(), row1[3].get<float>(),
		row2[0].get<float>(), row2[1].get<float>(), row2[2].get<float>(), row2[3].get<float>(),
		row3[0].get<float>(), row3[1].get<float>(), row3[2].get<float>(), row3[3].get<float>()
	);
	//for(int i = 0 ; i < 16 ; ++i)
	//{
	//	printf( "%f, ", glm::value_ptr( m )[i] );
	//}
	//printf( "\n" );
	m = glm::transpose( m );
	return m;
}

inline float compMin( glm::vec3 v )
{
	return glm::min( glm::min( v.x, v.y ), v.z );
}
inline float compMax( glm::vec3 v )
{
	return glm::max( glm::max( v.x, v.y ), v.z );
}
inline glm::vec2 slabs( glm::vec3 p0, glm::vec3 p1, glm::vec3 ro, glm::vec3 one_over_rd )
{
	glm::vec3 t0 = ( p0 - ro ) * one_over_rd;
	glm::vec3 t1 = ( p1 - ro ) * one_over_rd;

	glm::vec3 tmin = glm::min( t0, t1 );
	glm::vec3 tmax = glm::max( t0, t1 );
	float region_min = compMax( tmin );
	float region_max = compMin( tmax );

	region_min = glm::max( region_min, 0.0f );

	return glm::vec2( region_min, region_max );
}
inline glm::vec3 safe_inv_rd( glm::vec3 rd )
{
	return glm::clamp( glm::vec3( 1.0f, 1.0f, 1.0f ) / rd, glm::vec3( -FLT_MAX, -FLT_MAX, -FLT_MAX ), glm::vec3( FLT_MAX, FLT_MAX, FLT_MAX ) );
}

inline void invserseDistort( float* xpOut, float* ypOut, float xpp, float ypp, float k1, float k2, float p1, float p2, float k3 )
{
	float xp_i = xpp;
	float yp_i = ypp;
	for( int i = 0; i < 10; ++i )
	{
		float xp2 = xp_i * xp_i;
		float yp2 = yp_i * yp_i;
		float xpyp = xp_i * yp_i;
		float r2 = xp2 + yp2;
		float g = std::fma( std::fma( std::fma( k3, r2, k2 ), r2, k1 ), r2, 1.0f );

		float nx = ( xpp - ( 2.0f * p1 * xpyp + p2 * ( r2 + 2.0f * xp2 ) ) ) / g;
		float ny = ( ypp - ( p1 * ( r2 + 2.0f * yp2 ) + 2.0f * p2 * xpyp ) ) / g;
		xp_i = nx;
		yp_i = ny;
	}
	*xpOut = xp_i;
	*ypOut = yp_i;
}

struct OpenCVIntrinsicParam
{
	int width;
	int height;
	float fx;
	float fy;
	float cx;
	float cy;
	float k1;
	float k2;
	float p1;
	float p2;
};
class ColmapCamera
{
public:
	struct Camera
	{
		glm::mat4 transform;
		OpenCVIntrinsicParam intrinsicParam;
		std::string imagePath;
		pr::Image2DRGBA8 image;
	};
	void load( const char* configDir )
	{
		m_cameras.clear();

		std::string cameras = pr::JoinPath( configDir, "cameras.txt" );
		std::string images = pr::JoinPath( configDir, "images.txt" );
		std::string imagedir = pr::JoinPath( configDir, "images" );

		std::map<int, OpenCVIntrinsicParam> intrinsics;
		std::ifstream camerasStream( cameras );
		std::string line;
		while( std::getline( camerasStream, line ) )
		{
			if( line[0] == '#' )
				continue;

			int index;
			OpenCVIntrinsicParam intrinsic;
			int n = sscanf( line.c_str(), "%d OPENCV %d %d %f %f %f %f %f %f %f %f %f",
					&index, &intrinsic.width, &intrinsic.height,
					&intrinsic.fx, &intrinsic.fy, &intrinsic.cx, &intrinsic.cy,
					&intrinsic.k1, &intrinsic.k2, &intrinsic.p1, &intrinsic.p2 );

			if( n == 11 )
			{
				intrinsics[index] = intrinsic;
			}
		}

		std::ifstream imagesStream( images );
		while( std::getline( imagesStream, line ) )
		{
			if( line[0] == '#' )
				continue;

			Camera camera;
			int index;
			int intrinsicIndex;
			char imagename[256] = {};
			glm::quat rotation;
			glm::vec3 translation;
			int n = sscanf( line.c_str(), "%d %f %f %f %f %f %f %f %d %s",
							&index,
							&rotation.w, &rotation.x, &rotation.y, &rotation.z,
							&translation.x, &translation.y, &translation.z,
							&intrinsicIndex, imagename );
			if( n != 10 )
			{
				continue;
			}
			PR_ASSERT( intrinsics.count( intrinsicIndex ) == 1 );

			glm::mat4 transform = glm::translate( glm::identity<glm::mat4>(), translation ) * glm::mat4_cast( rotation );
			transform = glm::inverse( transform );

			// // x -> y -> z
			glm::mat4 adjustmentMatrix = glm::identity<glm::mat4>();

			// colmap 1
			//adjustmentMatrix = glm::translate( adjustmentMatrix, { 0.457007, 0.256999, 0.704056 } );
			//
			//adjustmentMatrix = glm::rotate( adjustmentMatrix, glm::radians( -356.442f ), { 0, 0, 1 } ); // Z
			//adjustmentMatrix = glm::rotate( adjustmentMatrix, glm::radians( 75.0906f ), { 0, 1, 0 } ); // Y
			//adjustmentMatrix = glm::rotate( adjustmentMatrix, glm::radians( -208.556f ), { 1, 0, 0 } ); // X

			//adjustmentMatrix = glm::scale( adjustmentMatrix, { 0.0779408f, 0.0779408f, 0.0779408f } );

			// photo2
			//adjustmentMatrix = glm::translate( adjustmentMatrix, { 0.544898, 0.235644, 0.504756 } );

			//adjustmentMatrix = glm::rotate( adjustmentMatrix, glm::radians( -2.50956f ), { 0, 0, 1 } ); // Z
			//adjustmentMatrix = glm::rotate( adjustmentMatrix, glm::radians( -30.7834f ), { 0, 1, 0 } ); // Y
			//adjustmentMatrix = glm::rotate( adjustmentMatrix, glm::radians( 151.98f ), { 1, 0, 0 } );	// X

			//adjustmentMatrix = glm::scale( adjustmentMatrix, { 0.055832, 0.055832, 0.055832 } );

			// photo3
			//adjustmentMatrix = glm::translate( adjustmentMatrix, { 0.580989, 0.408583, 0.465466 } );

			//adjustmentMatrix = glm::rotate( adjustmentMatrix, glm::radians( -0.258414f ), { 0, 0, 1 } ); // Z
			//adjustmentMatrix = glm::rotate( adjustmentMatrix, glm::radians( 65.5499f ), { 0, 1, 0 } );	// Y
			//adjustmentMatrix = glm::rotate( adjustmentMatrix, glm::radians( 144.194f ), { 1, 0, 0 } );	 // X

			//adjustmentMatrix = glm::scale( adjustmentMatrix, { 0.0516109, 0.0516109, 0.0516109 } );


			adjustmentMatrix = glm::translate( adjustmentMatrix, { 0.659591, 0.72655, 0.49641 } );

			adjustmentMatrix = glm::rotate( adjustmentMatrix, glm::radians( -0.258414f ), { 0, 0, 1 } ); // Z
			adjustmentMatrix = glm::rotate( adjustmentMatrix, glm::radians( 65.5499f ), { 0, 1, 0 } );	 // Y
			adjustmentMatrix = glm::rotate( adjustmentMatrix, glm::radians( 144.194f ), { 1, 0, 0 } );	 // X

			adjustmentMatrix = glm::scale( adjustmentMatrix, { 0.108665, 0.108665, 0.108665 } );

			camera.transform = adjustmentMatrix * transform;
			camera.intrinsicParam = intrinsics[intrinsicIndex];
			camera.imagePath = pr::JoinPath( imagedir, imagename );

			m_cameras.push_back( camera );

			// skip a line
			std::getline( imagesStream, line );
		}

		pr::ParallelFor( m_cameras.size(), [&]( int i )
		{
			m_cameras[i].image.load( m_cameras[i].imagePath.c_str() );
		} );
	}

	int numberOfCamera() const 
	{
		return m_cameras.size();
	}

	// note: nearClip, farClip is scaled by the transform
	void sample( glm::u8vec4* colorOut, glm::vec3* roOut, glm::vec3* rdOut, float nearClip, float farClip, int cameraIndex, float x0, float x1 )
	{
		const Camera& camera = m_cameras[cameraIndex];

		float u = x0 * camera.intrinsicParam.width;
		float v = x1 * camera.intrinsicParam.height;

		float xpp = ( u - camera.intrinsicParam.cx ) / camera.intrinsicParam.fx;
		float ypp = ( v - camera.intrinsicParam.cy ) / camera.intrinsicParam.fy;

		float xp;
		float yp;
		invserseDistort( &xp, &yp, xpp, ypp, camera.intrinsicParam.k1, camera.intrinsicParam.k2, camera.intrinsicParam.p1, camera.intrinsicParam.p2, 0.0f );

		glm::vec3 ro = { xp * nearClip, yp * nearClip, nearClip };
		glm::vec3 to = { xp * farClip, yp * farClip, farClip };

		int srcX = std::min( (int)u, camera.intrinsicParam.width - 1 );
		int srcY = std::min( (int)v, camera.intrinsicParam.height - 1 );

		// Apply Transform
		ro = camera.transform * glm::vec4( ro, 1.0f );
		to = camera.transform * glm::vec4( to, 1.0f );
		glm::vec3 rd = to - ro;

		*colorOut = camera.image( srcX, srcY );
		*roOut = ro;
		*rdOut = rd;
	}
	std::vector<Camera> m_cameras;
};

int main()
{
	using namespace pr;

	Stopwatch executionSW;

	SetDataDir( ExecutableDir() );

	if( oroInitialize( ( oroApi )( ORO_API_HIP | ORO_API_CUDA ), 0 ) )
	{
		printf( "failed to init..\n" );
		return 0;
	}

	oroError err;
	err = oroInit( 0 );
	oroDevice device;
	err = oroDeviceGet( &device, 0 );
	oroCtx ctx;
	err = oroCtxCreate( &ctx, 0, device );
	oroCtxSetCurrent( ctx );

	oroStream stream = 0;
	oroStreamCreate( &stream );

	oroDeviceProp props;
	oroGetDeviceProperties( &props, device );
	printf( "GPU: %s\n", props.name );

	ColmapCamera colmap;
	// colmap.load( pr::GetDataPath( "nerf/colmap" ).c_str() );
	// colmap.load( pr::GetDataPath( "nerf/photo2" ).c_str() );
	//colmap.load( pr::GetDataPath( "nerf/photo3" ).c_str() );

	NeRFg nerfg( pr::GetDataPath( "kernels" ) );
	std::vector<NerfCamera> cameras;

	// for( auto filePath : { "nerf/transforms_train.json", "nerf/transforms_test.json" ,"nerf/transforms_val.json"  } )

#if 1
	for( auto filePath : { "nerf/transforms_train.json" } )
	{
		std::ifstream ifs( GetDataPath( filePath ) );
		nlohmann::json j;
		ifs >> j;

		nlohmann::json camera_angle_x = j["camera_angle_x"];
		float fovy = camera_angle_x.get<float>();

		nlohmann::json frames = j["frames"];

		std::mutex mu;
		pr::ParallelFor( frames.size(), [&]( int i ) {
			nlohmann::json camera = frames[i];
			glm::mat4 m = loadMatrix( camera["transform_matrix"] );

			glm::mat4 rot = glm::rotate( glm::identity<glm::mat4>(), glm::radians( -90.0f ), glm::vec3( 1, 0, 0 ) );
			glm::mat4 s = glm::scale( glm::identity<glm::mat4>(), glm::vec3( 0.4f, 0.4f, 0.4f ) );
			glm::mat4 t = glm::translate( glm::identity<glm::mat4>(), glm::vec3( 0.5f, 0.5f, 0.5f ) );
			m = t * s * rot * m;

			NerfCamera nc = {};
			nc.fovy = fovy;
			nc.transform = m;
			nc.path = camera["file_path"].get<std::string>();

			std::string file = JoinPath( GetDataPath( "nerf" ), nc.path ) + ".png";
			Result r = nc.image.load( file.c_str() );
			PR_ASSERT( r == Result::Sucess, "" );
			// printf( "%d %d\n", nc.image.width(), nc.image.height() );

			std::lock_guard<std::mutex> lc( mu );
			cameras.push_back( nc );
		} );
	}
#endif

	float scale = 0.5f;
	const int n_rays_per_batch = 4096;

#if 0
	{
		static std::vector<NeRFInput> inputs;
		static std::vector<NeRFOutput> refs;
		static StandardRng rng;

		// for( int k = 0; k < 2048; ++k )
		for( int k = 0; executionSW.elapsed() < 60.0 * 7.0; ++k )
		{
			if( ( k % 16 ) == 0 )
			{
				printf( "train [%d] at %f\n", k, executionSW.elapsed() );
			}
			
			inputs.clear();
			refs.clear();

			for( int i = 0; i < n_rays_per_batch; ++i )
			{
				int camIdx = rng.drawUInt() % cameras.size();
				const NerfCamera& nc = cameras[camIdx];

				glm::vec3 o = { 0, 0, 0 };
				glm::vec3 up = { 0, 1, 0 };
				glm::vec3 lookat = { 0, 0, -1 };

				up = glm::mat3( glm::inverseTranspose( nc.transform ) ) * up;
				o = nc.transform * glm::vec4( o, 1.0f );
				lookat = nc.transform * glm::vec4( lookat, 1.0f );

				Camera3D cam3d;
				cam3d.origin = o;
				cam3d.lookat = lookat;
				cam3d.up = up;
				cam3d.fovy = nc.fovy;
				cam3d.zFar = 4.0f;
				cam3d.zNear = 0.1f;
				cam3d.zUp = false;
				glm::mat4 view, proj;
				GetCameraMatrix( cam3d, &proj, &view, 800, 800 );
				CameraRayGenerator raygen( view, proj, 800, 800 );

				glm::vec3 ro;
				glm::vec3 rd;
				int x = rng.drawUInt() % 800;
				int y = rng.drawUInt() % 800;
				raygen.shoot( &ro, &rd, x, y, rng.draw(), rng.draw() );
				rd = glm::normalize( rd );

				glm::vec3 one_over_rd = safe_inv_rd( rd );
				glm::vec3 input_ro = ro;

				NeRFInput input;
				input.ro[0] = input_ro.x;
				input.ro[1] = input_ro.y;
				input.ro[2] = input_ro.z;
				input.rd[0] = rd.x;
				input.rd[1] = rd.y;
				input.rd[2] = rd.z;
				inputs.push_back( input );

				glm::uvec4 color = nc.image( x, y );
				NeRFOutput output = {};
	#if LINEAR_SPACE_LEARNING
				output.color[0] = std::pow( (float)color.x / 255.0f, 2.2f );
				output.color[1] = std::pow( (float)color.y / 255.0f, 2.2f );
				output.color[2] = std::pow( (float)color.z / 255.0f, 2.2f );
	#else
				output.color[0] = (float)color.x / 255.0f;
				output.color[1] = (float)color.y / 255.0f;
				output.color[2] = (float)color.z / 255.0f;
	#endif
				output.color[3] = (float)color.z / 255.0f;
				refs.push_back( output );
			}
			nerfg.train( inputs.data(), refs.data(), inputs.size(), stream );
		}
	}


	pr::AbcArchive archive;
	std::string errorMessage;
	if( archive.open( GetDataPath( "camera.abc" ), errorMessage ) == AbcArchive::Result::Failure )
	{
		printf( "Alembic Error: %s\n", errorMessage.c_str() );
	}

	for( int i = 0; i < archive.frameCount() ; ++i )
	{
		printf( "render [%d] at %f\n", i, executionSW.elapsed() );

		auto scene = archive.readFlat( i, errorMessage );
		Camera3D camera;
		int imageWidth = 0;
		int imageHeight = 0;
		scene->visitCamera( [&]( std::shared_ptr<const pr::FCameraEntity> cameraEntity )
		{ 
			if( cameraEntity->visible() )
			{
				camera = cameraFromEntity( cameraEntity.get() ); 
				imageWidth = cameraEntity->imageWidth();
				imageHeight = cameraEntity->imageHeight();
			}
		} );

		std::vector<NeRFInput> nerf_in;
		std::vector<NeRFOutput> nerf_out;
		
		Image2DRGBA8 image;
		image.allocate( imageWidth, imageHeight );

		glm::mat4 viewMat, projMat;
		GetCameraMatrix( camera, &projMat, &viewMat, imageWidth, imageHeight );
		CameraRayGenerator rayGenerator( viewMat, projMat, imageWidth, imageHeight );
		for( int y = 0; y < image.height(); ++y )
		{
			for( int x = 0; x < image.width(); ++x )
			{
				glm::vec3 ro, rd;
				rayGenerator.shoot( &ro, &rd, x, y, 0.5f, 0.5f );
				rd = glm::normalize( rd );

				glm::vec3 input_ro = ro * scale + glm::vec3( 0.5f, 0.5f, 0.5f ); // -1 ~ +1 to 0 ~ 1

				NeRFInput input;
				input.ro[0] = input_ro.x;
				input.ro[1] = input_ro.y;
				input.ro[2] = input_ro.z;
				input.rd[0] = rd.x;
				input.rd[1] = rd.y;
				input.rd[2] = rd.z;
				nerf_in.push_back( input );
			}
		}

		nerf_out.resize( nerf_in.size() );
		nerfg.forward( nerf_in.data(), nerf_out.data(), nerf_in.size(), stream );

		int it = 0;
		for( int y = 0; y < image.height(); ++y )
		{
			for( int x = 0; x < image.width(); ++x )
			{
				NeRFOutput o = nerf_out[it++];
#if LINEAR_SPACE_LEARNING
				float r = glm::clamp( pow( o.color[0], 0.454545f ), 0.0f, 1.0f );
				float g = glm::clamp( pow( o.color[1], 0.454545f ), 0.0f, 1.0f );
				float b = glm::clamp( pow( o.color[2], 0.454545f ), 0.0f, 1.0f );
#else
				float r = glm::clamp( o.color[0], 0.0f, 1.0f );
				float g = glm::clamp( o.color[1], 0.0f, 1.0f );
				float b = glm::clamp( o.color[2], 0.0f, 1.0f );

#endif
				glm::u8vec4 color;
				color.r = r * 255.0f;
				color.g = g * 255.0f;
				color.b = b * 255.0f;
				color.a = 255;
				image( x, y ) = color;
			}
		}

		char fileout[256];
		sprintf( fileout, "image_%04d.png", i );
		image.saveAsPng( GetDataPath( fileout ).c_str() );
	}

	return 0;
#endif

	bool isLearning = true;

	ITexture* bg = CreateTexture();

	int imageWidth = 800;
	int imageHeight = 800;

	Image2DRGBA8 image;

	Config config;
	config.ScreenWidth = 800;
	config.ScreenHeight = 800;
	config.SwapInterval = 0;
	Initialize( config );

	Camera3D camera;
	camera.origin = { 3, 3, 3 };
	camera.lookat = { 0, 0, 0 };
	camera.zFar = 100;
	camera.zNear = 0.01f;
	camera.zUp = false;

	double e = GetElapsedTime();

	static float globalscale = 0.040;
	static glm::vec3 globallocation = glm::vec3( 0.074, 0.210, -0.192 );

	int _stride = 8;

	while( pr::NextFrame() == false )
	{
		if( IsImGuiUsingMouse() == false )
		{
			UpdateCameraBlenderLike( &camera );
		}

		// ClearBackground( 0.1f, 0.1f, 0.1f, 1 );
		ClearBackground( bg );
		

		BeginCamera( camera );

		PushGraphicState();

		DrawGrid( GridAxis::XZ, 0.1f, 20, { 128, 128, 128 } );
		DrawXYZAxis( 1.0f, 0.01f );

		DrawAABB( { -1, -1, -1 }, { 1, 1, 1 }, { 255, 0, 0 } );

		// colmap view
		for( int i = 0; i < colmap.numberOfCamera(); ++i )
		{
			glm::u8vec4 color;
			glm::vec3 ro;
			glm::vec3 rd;
			colmap.sample( &color, &ro, &rd, 0.01f, 1.0f, i, 0.0f, 0.0f );
			DrawLine( ro, ro + rd, { 128, 128, 128 } );
			colmap.sample( &color, &ro, &rd, 0.01f, 1.0f, i, 1.0f, 0.0f );
			DrawLine( ro, ro + rd, { 128, 128, 128 } );
			colmap.sample( &color, &ro, &rd, 0.01f, 1.0f, i, 0.0f, 1.0f );
			DrawLine( ro, ro + rd, { 128, 128, 128 } );
			colmap.sample( &color, &ro, &rd, 0.01f, 1.0f, i, 1.0f, 1.0f );
			DrawLine( ro, ro + rd, { 128, 128, 128 } );
		}

		// learning data
		float loss = 0;
		static std::vector<NeRFInput> inputs;
		static std::vector<NeRFOutput> refs;
		static StandardRng rng;

		

		static int iterations = 0; 

		if( isLearning )
		for(int k = 0 ; k < 16 ; ++k)
		{
			iterations++;

			inputs.clear();
			refs.clear();

			for( int i = 0; i < n_rays_per_batch; ++i )
			{
				// LEGO 
#if 1
				int camIdx = rng.drawUInt() % cameras.size();
				const NerfCamera& nc = cameras[camIdx];

				glm::vec3 o = { 0, 0, 0 };
				glm::vec3 up = { 0, 1, 0 };
				glm::vec3 lookat = { 0, 0, -1 };

				up = glm::mat3( glm::inverseTranspose( nc.transform ) ) * up;
				o = nc.transform * glm::vec4( o, 1.0f );
				lookat = nc.transform * glm::vec4( lookat, 1.0f );

				Camera3D cam3d;
				cam3d.origin = o;
				cam3d.lookat = lookat;
				cam3d.up = up;
				cam3d.fovy = nc.fovy;
				cam3d.zFar = 4.0f;
				cam3d.zNear = 0.1f;
				cam3d.zUp = false;
				glm::mat4 view, proj;
				GetCameraMatrix( cam3d, &proj, &view, 800, 800 );
				CameraRayGenerator raygen( view, proj, 800, 800 );

				glm::vec3 ro;
				glm::vec3 rd;
				int x = rng.drawUInt() % 800;
				int y = rng.drawUInt() % 800;
				raygen.shoot( &ro, &rd, x, y, rng.draw(), rng.draw() );
				rd = glm::normalize( rd );

				glm::vec3 one_over_rd = safe_inv_rd( rd );
				glm::vec3 input_ro = ro;

				NeRFInput input;
				input.ro[0] = input_ro.x;
				input.ro[1] = input_ro.y;
				input.ro[2] = input_ro.z;
				input.rd[0] = rd.x;
				input.rd[1] = rd.y;
				input.rd[2] = rd.z;
				inputs.push_back( input );
				glm::uvec4 color = nc.image( x, y );
#endif

#if 0
				int camIdx = rng.drawUInt() % colmap.numberOfCamera();
				glm::u8vec4 color;
				glm::vec3 ro;
				glm::vec3 rd;
				colmap.sample( &color, &ro, &rd, 0.1f, 1.0f, camIdx, rng.draw(), rng.draw() );
				rd = glm::normalize( rd );

				NeRFInput input;
				input.ro[0] = ro.x;
				input.ro[1] = ro.y;
				input.ro[2] = ro.z;
				input.rd[0] = rd.x;
				input.rd[1] = rd.y;
				input.rd[2] = rd.z;
				inputs.push_back( input );
#endif

				NeRFOutput output = {};
#if LINEAR_SPACE_LEARNING
				output.color[0] = std::pow( (float)color.x / 255.0f, 2.2f );
				output.color[1] = std::pow( (float)color.y / 255.0f, 2.2f );
				output.color[2] = std::pow( (float)color.z / 255.0f, 2.2f );
#else
				output.color[0] = (float)color.x / 255.0f;
				output.color[1] = (float)color.y / 255.0f;
				output.color[2] = (float)color.z / 255.0f;
#endif
				output.color[3] = (float)color.z / 255.0f;
				refs.push_back( output );
			}
			Stopwatch sw;
			nerfg.train( inputs.data(), refs.data(), inputs.size(), stream );
			//printf( "loss %f, t = %f\n", loss / inputs.size(), sw.elapsed() );
		}


#if 1
		static std::vector<NeRFInput> nerf_in;
		static std::vector<NeRFOutput> nerf_out;
		nerf_in.clear();

		image.allocate( GetScreenWidth() / _stride, GetScreenHeight() / _stride );

		glm::mat4 viewMat, projMat;
		GetCameraMatrix( camera, &projMat, &viewMat, GetScreenWidth(), GetScreenHeight() );
		CameraRayGenerator rayGenerator( viewMat, projMat, image.width(), image.height() );
		for( int y = 0; y < image.height(); ++y )
		{
			for( int x = 0; x < image.width(); ++x )
			{
				glm::vec3 ro, rd;
				rayGenerator.shoot( &ro, &rd, x, y, 0.5f, 0.5f );
				rd = glm::normalize( rd );

				glm::vec3 input_ro = ro * 0.5f + glm::vec3( 0.5f, 0.5f, 0.5f ); // -1 ~ +1 to 0 ~ 1

				NeRFInput input;
				input.ro[0] = input_ro.x;
				input.ro[1] = input_ro.y;
				input.ro[2] = input_ro.z;
				input.rd[0] = rd.x;
				input.rd[1] = rd.y;
				input.rd[2] = rd.z;
				nerf_in.push_back( input );
			}
		}

		nerf_out.resize( nerf_in.size() );
		nerfg.forward( nerf_in.data(), nerf_out.data(), nerf_in.size(), stream );

		int it = 0;
		for( int y = 0; y < image.height(); ++y )
		{
			for( int x = 0; x < image.width(); ++x )
			{
				NeRFOutput o = nerf_out[it++];
#if LINEAR_SPACE_LEARNING
				float r = glm::clamp( pow( o.color[0], 0.454545f ), 0.0f, 1.0f );
				float g = glm::clamp( pow( o.color[1], 0.454545f ), 0.0f, 1.0f );
				float b = glm::clamp( pow( o.color[2], 0.454545f ), 0.0f, 1.0f );
#else
				float r = glm::clamp( o.color[0], 0.0f, 1.0f );
				float g = glm::clamp( o.color[1], 0.0f, 1.0f );
				float b = glm::clamp( o.color[2], 0.0f, 1.0f );

#endif
				glm::u8vec4 color;
				color.r = r * 255.0f;
				color.g = g * 255.0f;
				color.b = b * 255.0f;
				color.a = 255;
				image( x, y ) = color;
			}
		}
		bg->upload( image );

#endif
		//static int iterations = 0;
		//if( iterations++ > 32 )
		//{
		//	auto tr = pr::ChromeTraceGetTrace();
		//	std::ofstream ofs( GetDataPath( "tr.json" ).c_str() );
		//	ofs << tr;
		//	break;
		//}
		 
		// mesh visualize
		//scene->visitPolyMesh( []( std::shared_ptr<const FPolyMeshEntity> polymesh ) 
		//{
		//	glm::mat4 objM = glm::identity<glm::mat4>();
		//	objM = glm::translate( objM, globallocation );
		//	objM = glm::scale( objM, glm::vec3( globalscale ) );

		//	SetObjectTransform( objM );

		//	ColumnView<int32_t> faceCounts( polymesh->faceCounts() );
		//	ColumnView<int32_t> indices( polymesh->faceIndices() );
		//	ColumnView<glm::vec3> positions( polymesh->positions() );
		//	ColumnView<glm::vec3> normals( polymesh->normals() );
		//	ColumnView<glm::vec2> uvs( polymesh->uvs() );

		//	// Show as Line Geometry
		//	pr::PrimBegin( pr::PrimitiveMode::Lines, 1 );
		//	for( int i = 0; i < positions.count(); i++ )
		//	{
		//		glm::vec3 p = positions[i];
		//		pr::PrimVertex( p, { 200, 200, 200 } );
		//	}
		//	int indexBase = 0;
		//	for( int i = 0; i < faceCounts.count(); i++ )
		//	{
		//		int nVerts = faceCounts[i];
		//		for( int j = 0; j < nVerts; ++j )
		//		{
		//			int i0 = indices[indexBase + j];
		//			int i1 = indices[indexBase + ( j + 1 ) % nVerts];
		//			pr::PrimIndex( i0 );
		//			pr::PrimIndex( i1 );
		//		}
		//		indexBase += nVerts;
		//	}
		//	pr::PrimEnd();
		//	SetObjectIdentify();
		//} );

		//for(int i = 0 ; i < cameras.size() ; i++) 
		//{
		//	glm::vec3 o = { 0, 0, 0 };
		//	glm::vec3 up = { 0, 1, 0 };
		//	glm::vec3 lookat = { 0, 0, -1 };

		//	NerfCamera nc = cameras[i];
		//	// NerfCamera nc = cameras[0];
		//	up = glm::mat3( glm::inverseTranspose( nc.transform ) ) * up;
		//	o = nc.transform * glm::vec4( o, 1.0f );
		//	lookat = nc.transform * glm::vec4( lookat, 1.0f );

		//	Camera3D cam3d;
		//	cam3d.origin = o;
		//	cam3d.lookat = lookat;
		//	cam3d.up = up;
		//	cam3d.zFar = 10.0f;
		//	cam3d.zNear = 0.001f;
		//	cam3d.fovy = nc.fovy;
		//	cam3d.zUp = false;
		//	glm::mat4 view, proj;
		//	GetCameraMatrix( cam3d, &proj, &view );
		//	CameraRayGenerator raygen( view, proj, 800, 800 );

		//	float L = 0.1f;
		//	glm::vec3 ro;
		//	glm::vec3 rd;
		//	raygen.shoot( &ro, &rd, 0, 0, 0.5f, 0.5f );
		//	DrawLine( ro, ro + glm::normalize( rd ) * L, { 255, 255, 255 } );
		//	raygen.shoot( &ro, &rd, 799, 0, 0.5f, 0.5f );
		//	DrawLine( ro, ro + glm::normalize( rd ) * L, { 255, 255, 255 } );
		//	raygen.shoot( &ro, &rd, 0, 799, 0.5f, 0.5f );
		//	DrawLine( ro, ro + glm::normalize( rd ) * L, { 255, 255, 255 } );
		//	raygen.shoot( &ro, &rd, 799, 799, 0.5f, 0.5f );
		//	DrawLine( ro, ro + glm::normalize( rd ) * L, { 255, 255, 255 } );

		//	DrawArrow( o, o + up * 0.1f, 0.005f, { 0, 255, 0 } );


		//	//raygen.shoot( &ro, &rd, 400, 400, 0.5f, 0.5f );
		//	//DrawLine( ro, ro + glm::normalize( rd ) * 10.0f, { 255, 0, 0 } );

		//	//rd = glm::normalize( rd );
		//	//glm::vec3 one_over_rd = safe_inv_rd( rd );
		//	//glm::vec3 localro = ro * 0.5f + glm::vec3( 0.5f, 0.5f, 0.5f );
		//	//glm::vec2 h = slabs( { 0, 0, 0 }, { 1, 1, 1 }, localro, one_over_rd );

		//	//DrawSphere( ro + glm::normalize( rd ) * h.x * 2.0f, 0.01f, { 0, 0, 255 } );
		//	//DrawSphere( ro + glm::normalize( rd ) * h.y * 2.0f, 0.01f, { 0, 0, 255 } );
		//}

		//{
		//	glm::vec3 o = { 0, 0, 0 };
		//	glm::vec3 up = { 0, 1, 0 };
		//	glm::vec3 lookat = { 0, 0, -1 };

		//	NerfCamera nc = cameras[0];
		//	up = glm::mat3( glm::inverseTranspose( nc.transform ) ) * up;
		//	o = nc.transform * glm::vec4( o, 1.0f );
		//	lookat = nc.transform * glm::vec4( lookat, 1.0f );

		//	glm::vec3 forward = lookat - o;
		//	DrawArrow( o, o + forward * 0.1f, 0.01f, { 0, 0, 255 } );
		//	DrawArrow( o, o + up * 0.1f, 0.01f, { 0, 255, 0 } );
		//}

		//for( auto imp : inputs )
		//{
		//	glm::vec3 ro = { imp.ro[0], imp.ro[1], imp.ro[2] };
		//	glm::vec3 rd = { imp.rd[0], imp.rd[1], imp.rd[2] };
		//	DrawArrow( ro, ro + rd *2.0f, 0.001f, { 255, 255, 255 } );
		//}
		
		static float guidebox_x = 0.0f;
		static float guidebox_y = 0.0f;
		static float guidebox_z = 0.0f;
		static float guidebox_sx = 1.0f;
		static float guidebox_sy = 0.9f;
		static float guidebox_sz = 0.9f;

		glm::vec3 center = { guidebox_x, guidebox_y, guidebox_z };
		glm::vec3 size = { guidebox_sx, guidebox_sy, guidebox_sz };
		DrawAABB( center - size * 0.5f, center + size * 0.5f, { 255, 255, 255 }, 2 );

		PopGraphicState();
		EndCamera();

		BeginImGui();

		ImGui::SetNextWindowSize( { 600, 200 }, ImGuiCond_Once );
		ImGui::Begin( "Panel" );
		ImGui::Text( "loss %.6f", loss / inputs.size() );

		ImGui::Text( "iterations %d", iterations );
		ImGui::Checkbox( "isLearning", &isLearning );

		static int index = 0;
		if( ImGui::InputInt( "index", &index ) )
		{
			index = glm::clamp( index, 0, (int)cameras.size() - 1 );

			glm::vec3 o = { 0, 0, 0 };
			glm::vec3 up = { 0, 1, 0 };
			glm::vec3 lookat = { 0, 0, -1 };

			
			NerfCamera nc = cameras[index];
			camera.up = glm::mat3( glm::inverseTranspose( nc.transform ) ) * up;
			camera.origin = nc.transform * glm::vec4( o, 1.0f );
			camera.lookat = nc.transform * glm::vec4( lookat, 1.0f );
			camera.fovy = nc.fovy;

			std::string file = JoinPath( GetDataPath( "nerf" ), nc.path ) + ".png";
			Image2DRGBA8 image;
			image.load( file.c_str() );
			bg->upload( image );

			SetDataDir( ExecutableDir() );
		}
		if( ImGui::Button( "stride = 16" ) )
		{
			_stride = 16;
		}	
		if( ImGui::Button( "stride = 8" ) )
		{
			_stride = 8;
		}	
		if( ImGui::Button( "stride = 4" ) )
		{
			_stride = 4;
		}
		if( ImGui::Button( "stride = 2" ) )
		{
			_stride = 2;
		}
		if( ImGui::Button( "stride = 1" ) )
		{
			_stride = 1;
		}
		if( ImGui::Button( "capture" ) )
		{
			Image2DRGBA8 img;
			CaptureScreen( &img );
			img.saveAsPng( "c.png" );
		}

		if( ImGui::Button( "Save Ref View" ) )
		{
			static std::vector<NeRFInput> nerf_in;
			static std::vector<NeRFOutput> nerf_out;
			nerf_in.clear();

			int cameraIndex = 0;
			const ColmapCamera::Camera& theCamera = colmap.m_cameras[cameraIndex];
			image.allocate( theCamera.image.width(), theCamera.image.height() );
			for( int y = 0; y < theCamera.image.height(); ++y )
			{
				for( int x = 0; x < theCamera.image.width(); ++x )
				{
					glm::u8vec4 color;
					glm::vec3 ro;
					glm::vec3 rd;
					colmap.sample( &color, &ro, &rd, 0.01f, 1.0f, cameraIndex, (float)x / theCamera.image.width(), (float)y / theCamera.image.height() );
					rd = glm::normalize( rd );

					NeRFInput input;
					input.ro[0] = ro.x;
					input.ro[1] = ro.y;
					input.ro[2] = ro.z;
					input.rd[0] = rd.x;
					input.rd[1] = rd.y;
					input.rd[2] = rd.z;
					nerf_in.push_back( input );
				}
			}

			nerf_out.resize( nerf_in.size() );
			nerfg.forward( nerf_in.data(), nerf_out.data(), nerf_in.size(), stream );

			int it = 0;
			for( int y = 0; y < image.height(); ++y )
			{
				for( int x = 0; x < image.width(); ++x )
				{
					NeRFOutput o = nerf_out[it++];
					float r = glm::clamp( o.color[0], 0.0f, 1.0f );
					float g = glm::clamp( o.color[1], 0.0f, 1.0f );
					float b = glm::clamp( o.color[2], 0.0f, 1.0f );

					glm::u8vec4 color;
					color.r = r * 255.0f;
					color.g = g * 255.0f;
					color.b = b * 255.0f;
					color.a = 255;
					image( x, y ) = color;
				}
			}

			char output[256];
			sprintf( output, "refView%04d.png", cameraIndex );
			image.saveAsPng( output );
		}

		ImGui::InputFloat( "guide box x", &guidebox_x, 0.1f );
		ImGui::InputFloat( "guide box y", &guidebox_y, 0.1f );
		ImGui::InputFloat( "guide box z", &guidebox_z, 0.1f );
		ImGui::InputFloat( "guide box sx", &guidebox_sx, 0.1f );
		ImGui::InputFloat( "guide box sy", &guidebox_sy, 0.1f );
		ImGui::InputFloat( "guide box sz", &guidebox_sz, 0.1f );

		ImGui::End();

		EndImGui();
	}

	pr::CleanUp();
}