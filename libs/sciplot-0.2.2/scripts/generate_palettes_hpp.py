import sys, os

# Auxiliary commands
up = os.path.dirname
join = os.path.join

# Download the git submodule gnuplot-palettes
os.system('git submodule update --init')

# The root directory of the sciplot project
rootdir = up(up(sys.argv[0]))

# The directory where gnuplot-palettes are downloaded
palettesdir = join(rootdir, 'deps/gnuplot-palettes')

# The directory of the library sciplot (source dir)
plotdir = join(rootdir, 'sciplot')

# The sorted list of palette file names (that ends with .pal)
filenames = [filename for filename in os.listdir(palettesdir) if filename.endswith('.pal')]
filenames.sort()

# The list of pairs (palette name, palette .pal file contents)
palettes = []

for filename in filenames:
    file = open(join(palettesdir, filename), 'r')
    palettes.append((filename[:-4], file.read()))

# Open the sciplot/Palettes.hpp file
palettes_hpp = open(join(plotdir, 'Palettes.hpp'), 'w')

# Ensure the print commands below end up writing in the Palettes.hpp file
sys.stdout = palettes_hpp

# Print the header part of the Palettes.hpp file
print(
"""// sciplot - a modern C++ scientific plotting library powered by gnuplot
// https://github.com/sciplot/sciplot
//
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
//
// Copyright (c) 2018-2021 Allan Leal
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

// C++ includes
#include <map>
#include <string>

namespace sciplot {

/// Gnuplot color palettes for sciplot adapted from https://github.com/Gnuplotting/gnuplot-palettes""")

# Print the std::map with keys equal to palette names and values as the pal file contents
print('const std::map<std::string, std::string> palettes = {')

for (key, value) in palettes:
    key = repr(key).replace("'", '"')
    value = repr(value).replace('"', '')
    value = '"{0}"'.format(value)
    print("    {{ {0}, {1} }},".format(key, value))

print('};')

# Print the closing brace of namespace sciplot
print()
print('} // namespace sciplot')
