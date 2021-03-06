/*
 * Copyright (C) 2014 Andres Gongora
 * <https://yalneb.blogspot.com>
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
/* modified by Shun Li */

/**
 *  ID: GoogleEarthPath.hpp
 *   EDITED:  23-09-2014
 *    AUTOR:  Andres Gongora
 *
 *  +------ Description:
 *-----------------------------------------------------------+ |
 *| | Class to store GPS coordinates in a Google Earth compatible KML file.
 *| | This can then be loaded and displayed as a route on Google Earth.
 *| | | | | | EXAMPLE OF USE:
 *| | 1 #include <GoogleEarthPath.hpp>    //This class |
 *  | 2 #include <unistd.h>     //Sleep
 *| | 3 #include <magic_GPS_library.hpp>  //Your GPS library | |
 *4                 | |
 *5 int main()              | |
 *6 {               | |
 *7   // CREATE PATH            | |
 *8   GoogleEarthPath path(myPath.kml, NameToShowInGEarth)  | |
 *9   double longitude, latitude;       | |
 *10                  | |
 *11    for(int i=0; i < 10; i++)       | |
 *12    {             | |
 *13      //SOMEHOW GET GPS COORDINATES     | |
 *14      yourGPSfunction(longitude,latitude);    | |
 *15                  | |
 *16      //ADD POINT TO PATH       | |
 *17      path.addPoint(longitude,latitude);    | |
 *18                  | |
 *19      //WAIT FOR NEW POINTS       | |
 *20      sleep(10); //sleep 10 seconds     | |
 *21    }             | |
 *22    return 0;           | |
 *23  }               | |
 *| | This example code obtains GPS coordenates from an external function
 *| | and stores them in the GoogleEarthPath instance. It samples 10 points |
 *  | during 100 seconds. Once the desctructor is called, the file
 *| | myPath.kml is ready to be imported in G.E. | |
 *|
 *  +-------------------------------------------------------------------------------+
 *
 **/

#ifndef INCLUDE_TOOLS_GOOGLEEARTHPATH_HPP_
#define INCLUDE_TOOLS_GOOGLEEARTHPATH_HPP_

#include <fstream>  // File I/O
#include <iomanip>  // std::setprecision
#include <iostream>

namespace FFDS {
namespace TOOLS {

class GoogleEarthPath {
 public:
  GoogleEarthPath(const char*,
                  const char*);  // Requires log filename & Path name (to show
                                 // inside googleearth)
  ~GoogleEarthPath();            // Default destructor

  inline void addPoint(const double, const double);

 private:
  std::fstream fileDescriptor;  // File descriptor
};

inline GoogleEarthPath::GoogleEarthPath(const char* file,
                                        const char* pathName) {
  fileDescriptor.open(
      file,
      std::ios::out | std::ios::trunc);  // Delete previous file if it exists
  if (!fileDescriptor)
    std::cerr << "GoogleEarthPath::\tCould not open file file! " << file
              << std::endl;

  fileDescriptor << "<?xml version=\"1.0\" encoding=\"UTF-8\"?>"
                 << "\n"
                 << "<kml xmlns=\"http://earth.google.com/kml/2.1\">"
                 << "\n"
                 << "<Document>"
                 << "\n"
                 << "<Placemark>"
                 << "\n"
                 << "<name>" << pathName << "</name>"
                 << "\n"
                 << "<LineString>"
                 << "\n"
                 << "<tessellate>1</tessellate>"
                 << "\n"
                 << "<coordinates>"
                 << "\n";
}

inline GoogleEarthPath::~GoogleEarthPath() {
  if (fileDescriptor) {
    fileDescriptor << "</coordinates>"
                   << "\n"
                   << "</LineString>"
                   << "\n"
                   << "</Placemark>"
                   << "\n"
                   << "</Document>"
                   << "\n"
                   << "</kml>"
                   << "\n";
  }
}

inline void GoogleEarthPath::addPoint(const double longitude,
                                      const double latitude) {
  if (fileDescriptor) {
    fileDescriptor << std::setprecision(13) << longitude << ","
                   << std::setprecision(13) << latitude << ",0\n"
                   << std::flush;
  }
}
}  // namespace TOOLS
}  // namespace FFDS

#endif  // INCLUDE_TOOLS_GOOGLEEARTHPATH_HPP_
