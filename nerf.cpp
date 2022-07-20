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
int main()
{
	using namespace pr;

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

	NeRF nerf;
	NeRFg nerfg( pr::GetDataPath( "kernels" ) );
	std::vector<NerfCamera> cameras;

	// for( auto filePath : { "nerf/transforms_train.json", "nerf/transforms_test.json" ,"nerf/transforms_val.json"  } )
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

			glm::mat4 s = glm::scale( glm::identity<glm::mat4>(), glm::vec3( 0.4f, 0.4f, 0.4f ) );
			glm::mat4 t = glm::translate( glm::identity<glm::mat4>(), glm::vec3( 0.5f, 0.5f, 0.5f ) );
			m = t * s * m;

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
		//for( int i = 0; i < frames.size(); i++ )
		//{
		//	nlohmann::json camera = frames[i];
		//	glm::mat4 m = loadMatrix( camera["transform_matrix"] );

		//	glm::mat4 s = glm::scale( glm::identity<glm::mat4>(), glm::vec3( 0.4f, 0.4f, 0.4f ) );
		//	glm::mat4 t = glm::translate( glm::identity<glm::mat4>(), glm::vec3( 0.5f, 0.5f, 0.5f ) );
		//	m = t * s * m;

		//	NerfCamera nc = {};
		//	nc.fovy = fovy;
		//	nc.transform = m;
		//	nc.path = camera["file_path"].get<std::string>();

		//	std::string file = JoinPath( GetDataPath( "nerf" ), nc.path ) + ".png";
		//	Result r = nc.image.load( file.c_str() );
		//	PR_ASSERT( r == Result::Sucess, "" );
		//	// printf( "%d %d\n", nc.image.width(), nc.image.height() );
		//	cameras.push_back( nc );
		//}
	}

	bool isCPUEval = false;

	//std::string error;
	//std::shared_ptr<FScene> scene = ReadWavefrontObj( GetDataPath( "nerf/Bulldozer.obj" ), error );

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

	// tmp
	//{
	//	glm::vec3 o = { 0, 0, 0 };
	//	glm::vec3 up = { 0, 1, 0 };
	//	glm::vec3 lookat = { 0, 0, -1 };

	//	NerfCamera nc = cameras[2];
	//	camera.up = glm::mat3( glm::inverseTranspose( nc.transform ) ) * up;
	//	camera.origin = nc.transform * glm::vec4( o, 1.0f );
	//	camera.lookat = nc.transform * glm::vec4( lookat, 1.0f );
	//	camera.fovy = nc.fovy;
	//}

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

		// learning data
		float loss = 0;
		static std::vector<NeRFInput> inputs;
		static std::vector<NeRFOutput> refs;
		static StandardRng rng;

		float scale = 0.5f;

		const int n_rays_per_batch = 4096;

		static int iterations = 0; 
		for(int k = 0 ; k < 16 ; ++k)
		{
			iterations++;

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
				GetCameraMatrix( cam3d, &proj, &view );
				CameraRayGenerator raygen( view, proj, 800, 800 );

				glm::vec3 ro;
				glm::vec3 rd;
				int x = rng.drawUInt() % 800;
				int y = rng.drawUInt() % 800;
				raygen.shoot( &ro, &rd, x, y, rng.draw(), rng.draw() );
				rd = glm::normalize( rd );

				glm::vec3 one_over_rd = safe_inv_rd( rd );
				// glm::vec3 input_ro = ro * scale * 0.75f + glm::vec3( 0.5f, 0.5f, 0.5f ); // -1 ~ +1 to 0 ~ 1
				glm::vec3 input_ro = ro;
				glm::vec2 h = slabs( { 0, 0, 0 }, { 1, 1, 1 }, input_ro, one_over_rd );

				if( h.x /* min */ < h.y /* max */ )
				{
					input_ro = input_ro + rd * h.x; // move to inside

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
					// printf( "%f %f %f\n", output.color[0], output.color[1], output.color[2] );
					refs.push_back( output );
				}
			}
			//printf( "input: %d\n", (int)inputs.size() );
			Stopwatch sw;
			nerfg.train( inputs.data(), refs.data(), inputs.size(), stream );
			//loss = nerf.train( inputs.data(), refs.data(), inputs.size() );
			//printf( "loss %f, t = %f\n", loss / inputs.size(), sw.elapsed() );
		}

		static std::vector<NeRFInput> nerf_in;
		static std::vector<NeRFOutput> nerf_out;
		nerf_in.clear();

		image.allocate( GetScreenWidth() / _stride, GetScreenHeight() / _stride );

		glm::mat4 viewMat, projMat;
		GetCameraMatrix( camera, &projMat, &viewMat );
		CameraRayGenerator rayGenerator( viewMat, projMat, image.width(), image.height() );
		for( int y = 0; y < image.height(); ++y )
		{
			for( int x = 0; x < image.width(); ++x )
			{
				glm::vec3 ro, rd;
				rayGenerator.shoot( &ro, &rd, x, y, 0.5f, 0.5f );
				rd = glm::normalize( rd );

				glm::vec3 one_over_rd = safe_inv_rd( rd );
				glm::vec3 input_ro = ro * scale + glm::vec3( 0.5f, 0.5f, 0.5f ); // -1 ~ +1 to 0 ~ 1
				glm::vec2 h = slabs( { 0, 0, 0 }, { 1, 1, 1 }, input_ro, one_over_rd );

				if( h.x /* min */ < h.y /* max */ )
				{
					input_ro = input_ro + rd * h.x; // move to inside

					NeRFInput input;
					input.ro[0] = input_ro.x;
					input.ro[1] = input_ro.y;
					input.ro[2] = input_ro.z;
					input.rd[0] = rd.x;
					input.rd[1] = rd.y;
					input.rd[2] = rd.z;
					nerf_in.push_back( input );
				}
				else
				{
					//NeRFInput input;
					//input.ro[0] = 10;
					//input.ro[1] = 10;
					//input.ro[2] = 10;
					//input.rd[0] = 1;
					//input.rd[1] = 0;
					//input.rd[2] = 0;
					//nerf_in.push_back( input );
				}
			}
		}

		nerf_out.resize( nerf_in.size() );

		if( isCPUEval )
		{
			nerf.forward( nerf_in.data(), nerf_out.data(), nerf_in.size() );
		}
		else
		{
			//nerfg.takeReference( nerf );
			nerfg.forward( nerf_in.data(), nerf_out.data(), nerf_in.size(), stream );
		}

		int it = 0;
		for( int y = 0; y < image.height(); ++y )
		{
			for( int x = 0; x < image.width(); ++x )
			{
				glm::vec3 ro, rd;
				rayGenerator.shoot( &ro, &rd, x, y, 0.5f, 0.5f );
				rd = glm::normalize( rd );

				glm::vec3 one_over_rd = safe_inv_rd( rd );
				glm::vec3 input_ro = ro * scale + glm::vec3( 0.5f, 0.5f, 0.5f ); // -1 ~ +1 to 0 ~ 1
				glm::vec2 h = slabs( { 0, 0, 0 }, { 1, 1, 1 }, input_ro, one_over_rd );

				if( h.x /* min */ < h.y /* max */ )
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

					// printf( " %.5f %.5f %.5f\n", o.color[0], o.color[1], o.color[2] );
				}
				else
				{
					image( x, y ) = { 0, 0, 0, 255 };
				}
			}
		}
		bg->upload( image );

#if 0
		float step = 2.0f / NERF_OCCUPANCY_GRID_MIN_RES;
		PrimBegin( pr::PrimitiveMode::Points, 4 );
		if( !nerfg.m_grid.empty() )
		for( int i = 0; i < nerfg.m_grid.size() ; i++ )
		{
			int yi = i;
		    int index_z = yi / ( NERF_OCCUPANCY_GRID_MIN_RES * NERF_OCCUPANCY_GRID_MIN_RES );
			yi = yi % ( NERF_OCCUPANCY_GRID_MIN_RES * NERF_OCCUPANCY_GRID_MIN_RES );
			int index_y = yi / NERF_OCCUPANCY_GRID_MIN_RES;
			yi = yi % NERF_OCCUPANCY_GRID_MIN_RES;
			int index_x = yi;

			//if( ( i % 127 ) == 0 )
			//{
			//	printf( "den %f\n", nerfg.m_grid[i] );
			//	 printf("d %f %f %f\n", mlp, optical_thickness, MIN_CONE_STEPSIZE(), scalbnf(MIN_CONE_STEPSIZE(), level) );
			//}

			if( nerfg.m_avg < nerfg.m_grid[i] )
			{
				glm::vec3 p = {
					-1.0f + step * index_x + step * 0.5f,
					-1.0f + step * index_y + step * 0.5f,
					-1.0f + step * index_z + step * 0.5f
				};
				int c = minss( nerfg.m_grid[yi] * 255, 255 );
				PrimVertex( p, { c, c, c } );
			}
		}
		PrimEnd();
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

		PopGraphicState();
		EndCamera();

		BeginImGui();

		ImGui::SetNextWindowSize( { 600, 200 }, ImGuiCond_Once );
		ImGui::Begin( "Panel" );
		ImGui::Text( "loss %.6f", loss / inputs.size() );

		
		ImGui::Text( "iterations %d", iterations );
		//ImGui::InputFloat( "globalscale", &globalscale, 0.0001f );
		//ImGui::InputFloat( "globallocation.x", &globallocation.x, 0.001f );
		//ImGui::InputFloat( "globallocation.y", &globallocation.y, 0.001f );
		//ImGui::InputFloat( "globallocation.z", &globallocation.z, 0.001f );
		ImGui::Checkbox( "isCPUEval", &isCPUEval );

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

		ImGui::End();

		EndImGui();
	}

	pr::CleanUp();
}