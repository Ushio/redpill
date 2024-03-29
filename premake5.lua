include "libs/PrLib"

workspace "MachineLearning"
    location "build"
    configurations { "Debug", "Release" }
    startproject "main"

architecture "x86_64"

externalproject "prlib"
	location "libs/PrLib/build" 
    kind "StaticLib"
    language "C++"

project "demo"
    kind "ConsoleApp"
    language "C++"
    targetdir "bin/"
    systemversion "latest"
    flags { "MultiProcessorCompile", "NoPCH" }
    cppdialect "C++17"
    characterset ("MBCS")

    -- Src
    files { "main.cpp" }
    files { "redpill.hpp" }
    files { "redpillg.hpp" }
    includedirs { "bin/kernels" }
    files { "bin/kernels/*.hpp" }

    -- Orochi
    includedirs { "libs/orochi" }
    files { "libs/orochi/Orochi/Orochi.h" }
    files { "libs/orochi/Orochi/Orochi.cpp" }
    includedirs { "libs/orochi/contrib/hipew/include" }
    files { "libs/orochi/contrib/hipew/src/hipew.cpp" }
    includedirs { "libs/orochi/contrib/cuew/include" }
    files { "libs/orochi/contrib/cuew/src/cuew.cpp" }
    links { "version" }

    -- UTF8
    postbuildcommands { 
        "mt.exe -manifest ../utf8.manifest -outputresource:\"$(TargetDir)$(TargetName).exe\" -nologo",
        "{COPYFILE} ../libs/orochi/contrib/bin/win64/*.dll ../bin"
    }

    -- Helper
    -- files { "libs/d3dx12/*.h" }
    -- includedirs { "libs/d3dx12/" }

    -- links { "dxgi" }
    -- links { "d3d12" }

    -- -- HLSL compiler
    -- includedirs { "libs/dxc_2021_12_08/inc" }
    -- links { "libs/dxc_2021_12_08/lib/x64/dxcompiler" }
    -- postbuildcommands { 
    --     "{COPYFILE} ../libs/dxc_2021_12_08/bin/x64/*.dll ../bin"
    -- }

    -- half
    includedirs { "libs/prlib/libs/src_ilmbase/" }
    

    -- json
    includedirs { "libs/json" }
    files { "libs/json/json.hpp" }

    -- plot
    includedirs { "libs/sciplot-0.2.2" }
    files { "libs/sciplot-0.2.2/sciplot/**.hpp" }

    -- prlib
    -- setup command
    -- git submodule add https://github.com/Ushio/prlib libs/prlib
    -- premake5 vs2017
    dependson { "prlib" }
    includedirs { "libs/prlib/src" }
    libdirs { "libs/prlib/bin" }
    filter {"Debug"}
        links { "prlib_d" }
    filter {"Release"}
        links { "prlib" }
    filter{}

    symbols "On"

    filter {"Debug"}
        runtime "Debug"
        targetname ("demo_Debug")
        optimize "Off"
    filter {"Release"}
        runtime "Release"
        targetname ("demo")
        optimize "Full"
    filter{}

project "nerf"
    kind "ConsoleApp"
    language "C++"
    targetdir "bin/"
    systemversion "latest"
    flags { "MultiProcessorCompile", "NoPCH" }
    cppdialect "C++17"
    characterset ("MBCS")
    
    -- Src
    files { "nerf.cpp" }
    files { "redpill.hpp" }
    files { "redpillg.hpp" }
    includedirs { "bin/kernels" }
    files { "bin/kernels/*.hpp" }

    -- Orochi
    includedirs { "libs/orochi" }
    files { "libs/orochi/Orochi/Orochi.h" }
    files { "libs/orochi/Orochi/Orochi.cpp" }
    includedirs { "libs/orochi/contrib/hipew/include" }
    files { "libs/orochi/contrib/hipew/src/hipew.cpp" }
    includedirs { "libs/orochi/contrib/cuew/include" }
    files { "libs/orochi/contrib/cuew/src/cuew.cpp" }
    links { "version" }

    -- UTF8
    postbuildcommands { 
        "mt.exe -manifest ../utf8.manifest -outputresource:\"$(TargetDir)$(TargetName).exe\" -nologo",
        "{COPYFILE} ../libs/orochi/contrib/bin/win64/*.dll ../bin"
    }

    -- json
    includedirs { "libs/json" }
    files { "libs/json/json.hpp" }

    -- half
    includedirs { "libs/prlib/libs/src_ilmbase/" }

    -- prlib
    -- setup command
    -- git submodule add https://github.com/Ushio/prlib libs/prlib
    -- premake5 vs2017
    dependson { "prlib" }
    includedirs { "libs/prlib/src" }
    libdirs { "libs/prlib/bin" }
    filter {"Debug"}
        links { "prlib_d" }
    filter {"Release"}
        links { "prlib" }
    filter{}

    symbols "On"

    filter {"Debug"}
        runtime "Debug"
        targetname ("nerf_Debug")
        optimize "Off"
    filter {"Release"}
        runtime "Release"
        targetname ("nerf")
        optimize "Full"
    filter{}

project "bugcatcher"
    kind "ConsoleApp"
    language "C++"
    targetdir "bin/"
    systemversion "latest"
    flags { "MultiProcessorCompile", "NoPCH" }

    -- Src
    files { "bugcatcher.cpp" }
    files { "redpill.hpp" }
    files { "libs/prlib/src/prth.cpp" }
    files { "libs/prlib/src/prth.hpp" }
    includedirs { "bin/kernels" }
    files { "bin/kernels/*.hpp" }

    includedirs { "." }
    includedirs { "libs/prlib/src" }

    -- UTF8
    postbuildcommands { 
        "mt.exe -manifest ../utf8.manifest -outputresource:\"$(TargetDir)$(TargetName).exe\" -nologo"
    }

    symbols "On"

    filter {"Debug"}
        runtime "Debug"
        targetname ("bugcatcher_Debug")
        optimize "Off"
    filter {"Release"}
        runtime "Release"
        targetname ("bugcatcher")
        optimize "Full"
    filter{}