#include "pr.hpp"
#include <iostream>
#include <memory>
#include <fstream>

//#define RPML_DISABLE_ASSERT
#include "redpill.hpp"
using namespace rpml;

#if defined( DrawText )
#undef DrawText
#endif

#include <json.hpp>

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

	NeRF nerf;

	std::vector<NerfCamera> cameras;
	{
		std::ifstream ifs( GetDataPath( "nerf/transforms_train.json" ) );
		nlohmann::json j;
		ifs >> j;

		nlohmann::json camera_angle_x = j["camera_angle_x"];
		float fovy = camera_angle_x.get<float>();

		nlohmann::json frames = j["frames"];
		for( int i = 0; i < frames.size() ; i++ )
		//for( int i = 0; i < 10; i++ )
		{
			nlohmann::json camera = frames[i];
			glm::mat4 m = loadMatrix( camera["transform_matrix"] );

			NerfCamera nc = {};
			nc.fovy = fovy;
			nc.transform = m;
			nc.path = camera["file_path"].get<std::string>();

			std::string file = JoinPath( GetDataPath( "nerf" ), nc.path ) + ".png";
			nc.image.load( file.c_str() );

			cameras.push_back( nc );
		}
		
	}

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

	int _stride = 4;

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

		for(int k = 0 ; k < 1 ; ++k)
		{
			inputs.clear();
			refs.clear();

			for( int i = 0; i < cameras.size(); i++ ) 
			{
				const NerfCamera& nc = cameras[i];

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
				cam3d.zFar = 10;
				cam3d.zNear = 0.001f;
				cam3d.zUp = false;
				glm::mat4 view, proj;
				GetCameraMatrix( cam3d, &proj, &view );
				CameraRayGenerator raygen( view, proj, 800, 800 );

				for( int j = 0; j < 32; ++j )
				{
					glm::vec3 ro;
					glm::vec3 rd;
					int x = rng.drawUInt() % imageWidth;
					int y = rng.drawUInt() % imageHeight;
					raygen.shoot( &ro, &rd, x, y, 0.5f, 0.5f );
					rd = glm::normalize( rd );

					glm::vec3 one_over_rd = safe_inv_rd( rd );
					glm::vec3 input_ro = ro * 0.5f + glm::vec3( 0.5f, 0.5f, 0.5f ); // -1 ~ +1 to 0 ~ 1
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
						NeRFOutput output;
						output.color[0] = std::pow( (float)color.x / 255.0f, 2.2f );
						output.color[1] = std::pow( (float)color.y / 255.0f, 2.2f );
						output.color[2] = std::pow( (float)color.z / 255.0f, 2.2f );
						refs.push_back( output );
					}
				}
			}
		
			loss = nerf.train( inputs.data(), refs.data(), inputs.size() );
			printf( "loss %f\n", loss / inputs.size() );
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

				glm::vec3 one_over_rd = safe_inv_rd( rd );
				glm::vec3 input_ro = ro * 0.5f + glm::vec3( 0.5f, 0.5f, 0.5f ); // -1 ~ +1 to 0 ~ 1
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
					NeRFInput input;
					input.ro[0] = 10;
					input.ro[1] = 10;
					input.ro[2] = 10;
					input.rd[0] = 1;
					input.rd[1] = 0;
					input.rd[2] = 0;
					nerf_in.push_back( input );
				}
			}
		}

		nerf_out.resize( nerf_in.size() );
		nerf.forward( nerf_in.data(), nerf_out.data(), nerf_in.size() );

		for( int y = 0; y < image.height(); ++y )
		{
			for( int x = 0; x < image.width(); ++x )
			{
				NeRFOutput o = nerf_out[y * image.height() + x];
				float r = glm::clamp( pow( o.color[0], 0.454545f ) , 0.0f, 1.0f );
				float g = glm::clamp( pow( o.color[1], 0.454545f ) , 0.0f, 1.0f );
				float b = glm::clamp( pow( o.color[2], 0.454545f ) , 0.0f, 1.0f );
				glm::u8vec4 color;
				color.r = r * 255.0f;
				color.g = g * 255.0f;
				color.g = g * 255.0f;
				color.a = 255;
				image( x, y ) = color;
			}
		}
		bg->upload( image );

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

		// for(int i = 0 ; i < cameras.size() ; i++) 
		{
			glm::vec3 o = { 0, 0, 0 };
			glm::vec3 up = { 0, 1, 0 };
			glm::vec3 lookat = { 0, 0, -1 };

			// NerfCamera nc = cameras[i];
			NerfCamera nc = cameras[0];
			up = glm::mat3( glm::inverseTranspose( nc.transform ) ) * up;
			o = nc.transform * glm::vec4( o, 1.0f );
			lookat = nc.transform * glm::vec4( lookat, 1.0f );

			Camera3D cam3d;
			cam3d.origin = o;
			cam3d.lookat = lookat;
			cam3d.up = up;
			cam3d.zFar = 10;
			cam3d.zNear = 0.001f;
			cam3d.zUp = false;
			glm::mat4 view, proj;
			GetCameraMatrix( cam3d, &proj, &view );
			CameraRayGenerator raygen( view, proj, 800, 800 );

			glm::vec3 ro;
			glm::vec3 rd;
			raygen.shoot( &ro, &rd, 0, 0, 0.5f, 0.5f );
			DrawLine( ro, ro + glm::normalize( rd ) * 10.0f, { 255, 255, 255 } );
			raygen.shoot( &ro, &rd, 799, 0, 0.5f, 0.5f );
			DrawLine( ro, ro + glm::normalize( rd ) * 10.0f, { 255, 255, 255 } );
			raygen.shoot( &ro, &rd, 0, 799, 0.5f, 0.5f );
			DrawLine( ro, ro + glm::normalize( rd ) * 10.0f, { 255, 255, 255 } );
			raygen.shoot( &ro, &rd, 799, 799, 0.5f, 0.5f );
			DrawLine( ro, ro + glm::normalize( rd ) * 10.0f, { 255, 255, 255 } );

			DrawArrow( o, o + up * 0.1f, 0.005f, { 0, 255, 0 } );


			raygen.shoot( &ro, &rd, 400, 400, 0.5f, 0.5f );
			DrawLine( ro, ro + glm::normalize( rd ) * 10.0f, { 255, 0, 0 } );

			rd = glm::normalize( rd );
			glm::vec3 one_over_rd = safe_inv_rd( rd );
			glm::vec3 localro = ro * 0.5f + glm::vec3( 0.5f, 0.5f, 0.5f );
			glm::vec2 h = slabs( { 0, 0, 0 }, { 1, 1, 1 }, localro, one_over_rd );

			DrawSphere( ro + glm::normalize( rd ) * h.x * 2.0f, 0.01f, { 0, 0, 255 } );
			DrawSphere( ro + glm::normalize( rd ) * h.y * 2.0f, 0.01f, { 0, 0, 255 } );
		}

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

		PopGraphicState();
		EndCamera();

		BeginImGui();

		ImGui::SetNextWindowSize( { 600, 400 }, ImGuiCond_Once );
		ImGui::Begin( "Panel" );
		ImGui::InputFloat( "globalscale", &globalscale, 0.0001f );
		ImGui::InputFloat( "globallocation.x", &globallocation.x, 0.001f );
		ImGui::InputFloat( "globallocation.y", &globallocation.y, 0.001f );
		ImGui::InputFloat( "globallocation.z", &globallocation.z, 0.001f );

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