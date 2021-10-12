/** 
 * \copyright Copyright (C) 2021 Edward Janne - All Rights Reserved
 * 
 * \section LICENSE
 * This file is part of FuseToEuler.
 * 
 * FuseToEuler is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * FuseToEuler is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with FuseToEuler. If not, see <https://www.gnu.org/licenses/>.
 * 
 * \brief FuseToEuler shows how to incorporate Sebastian Madgwick's sensor
 * fusion library to fuse raw IMU gyroscope and accelerometer telemetry
 * from RealSense cameras.
 * 
 * \file RealFusion.hpp
 * \author Edward Janne
 * \class RealFusion RealFusion.hpp
 * \brief A class to encapsulate librealsense2 and Fusion access.
 * 
 */

#ifndef __REALFUSION_HPP__
#define __REALFUSION_HPP__

#include <librealsense2/rs.hpp>

#include "IMUFusion.hpp"
#include "Exception.hpp"

class RealFusion
{
    public:
        /**
         * \fn RealFusion()
         * \brief Constructor. Checks IMU available, initializes librealsense.
         * \returns void
         */
        RealFusion() throw(Exception);

        /**
         * \fn ~RealFusion()
         * \brief Destructor
         * \returns void
         */
        virtual ~RealFusion() { }

        /**
         * \fn bool checkIMU()
         * \brief Checks availability of IMU.
         * \returns true if IMU exists, false if not
         */
        bool checkIMU();

        /**
         * \fn void tick()
         * \brief Retrieves frames from librealsense2, extracts IMU data, and
         * calls Fusion to perform Madgwick fusion to obtain Euler orientation
         * which it caches in the angles public member
         * \returns void
         */
        void tick();

        FusionEulerAngles angles; //< Member to cache Euler orientations in degrees

    protected:
        rs2::pipeline pl;
        IMUFusion fuser;
};

#endif
