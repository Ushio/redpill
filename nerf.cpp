#include "pr.hpp"
#include <iostream>
#include <memory>
#include <fstream>

#define RPML_DISABLE_ASSERT
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

int main()
{
	using namespace pr;

	SetDataDir( ExecutableDir() );

	std::vector<NerfCamera> cameras;
	{
		std::ifstream ifs( GetDataPath( "nerf/transforms_train.json" ) );
		nlohmann::json j;
		ifs >> j;

		nlohmann::json camera_angle_x = j["camera_angle_x"];
		float fovy = camera_angle_x.get<float>();

		nlohmann::json frames = j["frames"];
		for( int i = 0; i < frames.size() ; i++ )
		{
			nlohmann::json camera = frames[i];
			glm::mat4 m = loadMatrix( camera["transform_matrix"] );

			NerfCamera nc = {};
			nc.fovy = fovy;
			nc.transform = m;
			nc.path = camera["file_path"].get<std::string>();
			cameras.push_back( nc );
		}
		
	}

	std::string error;
	std::shared_ptr<FScene> scene = ReadWavefrontObj( GetDataPath( "nerf/Bulldozer.obj" ), error );

	ITexture* bg = CreateTexture();

	Config config;
	config.ScreenWidth = 800;
	config.ScreenHeight = 800;
	config.SwapInterval = 0;
	Initialize( config );

	Camera3D camera;
	camera.origin = { 0, 0, 2 };
	camera.lookat = { 0, 0, 0 };
	camera.zFar = 100;
	camera.zNear = 0.01f;
	camera.zUp = false;

	double e = GetElapsedTime();

	static float globalscale = 0.040;
	static glm::vec3 globallocation = glm::vec3( 0.074, 0.210, -0.192 );

	// tmp
	{
		glm::vec3 o = { 0, 0, 0 };
		glm::vec3 up = { 0, 1, 0 };
		glm::vec3 lookat = { 0, 0, -1 };

		NerfCamera nc = cameras[2];
		camera.up = glm::mat3( glm::inverseTranspose( nc.transform ) ) * up;
		camera.origin = nc.transform * glm::vec4( o, 1.0f );
		camera.lookat = nc.transform * glm::vec4( lookat, 1.0f );
		camera.fovy = nc.fovy;
	}

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

		scene->visitPolyMesh( []( std::shared_ptr<const FPolyMeshEntity> polymesh ) 
		{
			glm::mat4 objM = glm::identity<glm::mat4>();
			objM = glm::translate( objM, globallocation );
			objM = glm::scale( objM, glm::vec3( globalscale ) );

			SetObjectTransform( objM );

			ColumnView<int32_t> faceCounts( polymesh->faceCounts() );
			ColumnView<int32_t> indices( polymesh->faceIndices() );
			ColumnView<glm::vec3> positions( polymesh->positions() );
			ColumnView<glm::vec3> normals( polymesh->normals() );
			ColumnView<glm::vec2> uvs( polymesh->uvs() );

			// Show as Line Geometry
			pr::PrimBegin( pr::PrimitiveMode::Lines, 1 );
			for( int i = 0; i < positions.count(); i++ )
			{
				glm::vec3 p = positions[i];
				pr::PrimVertex( p, { 200, 200, 200 } );
			}
			int indexBase = 0;
			for( int i = 0; i < faceCounts.count(); i++ )
			{
				int nVerts = faceCounts[i];
				for( int j = 0; j < nVerts; ++j )
				{
					int i0 = indices[indexBase + j];
					int i1 = indices[indexBase + ( j + 1 ) % nVerts];
					pr::PrimIndex( i0 );
					pr::PrimIndex( i1 );
				}
				indexBase += nVerts;
			}
			pr::PrimEnd();
			SetObjectIdentify();
		} );

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