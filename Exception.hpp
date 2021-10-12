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

/*
 * @file Exception.cpp
 */

#ifndef __EXCEPTION_HPP__
#define __EXCEPTION_HPP__

#include <iostream>

using namespace std;

class Exception
{
    public:
        Exception(const string &iFile, int iLine, const string &iFunc, int iErrorCode, const string &iMsg)
        : file(iFile), line(iLine), func(iFunc), code(iErrorCode), msg(iMsg) { }

        string file;
        int line;
        string func;
        int code;
        string msg;
};

ostream &operator<<(ostream &os, const Exception &e);

#endif
