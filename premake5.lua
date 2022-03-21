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

    -- Src
    files { "main.cpp" }
    files { "redpill.hpp" }

    -- UTF8
    postbuildcommands { 
        "mt.exe -manifest ../utf8.manifest -outputresource:$(TargetDir)$(TargetName).exe -nologo"
    }

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

project "bugcatcher"
    kind "ConsoleApp"
    language "C++"
    targetdir "bin/"
    systemversion "latest"
    flags { "MultiProcessorCompile", "NoPCH" }

    -- Src
    files { "bugcatcher.cpp" }
    files { "redpill.hpp" }
    includedirs { "." }
    includedirs { "libs/prlib/src" }

    -- UTF8
    postbuildcommands { 
        "mt.exe -manifest ../utf8.manifest -outputresource:$(TargetDir)$(TargetName).exe -nologo"
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