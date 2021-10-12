/*
 * @file Exception.cpp
 */

// This file is part of RealFusion.

// RealFusion is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.

// RealFusion is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with RealFusion. If not, see <https://www.gnu.org/licenses/>.

#include "Exception.hpp"

ostream &operator<<(ostream &os, const Exception &e)
{
    os << "Exception (file: " << e.file << ", line: " << e.line << ", function: " << e.func << ", code: " << e.code << ", msg: " << e.msg << ")\n";
}
