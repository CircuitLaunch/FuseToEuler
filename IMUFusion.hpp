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
 * \file IMUFusion.hpp
 * \author Edward Janne
 * \class IMUFusion IMUFusion.hpp
 * \brief A class to encapsulate Madgwick Fusion access.
 * 
 */

#ifndef __IMUFUSION_HPP__
#define __IMUFUSION_HPP__

#include <librealsense2/rs.hpp>
#include "Fusion/Fusion.h"

// For the RealSense L515
// See the datasheet for your own camera for suitable settings
#define GYRO_SENSITIVITY 262.144f
#define ACCEL_SENSITIVITY 16384.0f

class IMUFusion
{
    public:
        /**
         * \fn IMUFusion()
         * \brief Constructor. Configures Madgwick Fusion for RealSense camera IMU
         */
        IMUFusion(float iSamplePeriod = 0.004f, float iStationaryThreshold = 0.5f, float iAhrsGain = 0.5f, float iGyroSensitivity = GYRO_SENSITIVITY, float iAccelSensitivity = ACCEL_SENSITIVITY);
        
        /**
         * \fn ~IMUFusion()
         * \brief Destructor
         */
        virtual ~IMUFusion() { }

        /**
         * \fn FusionEulerAngles fuse(rs2_vector &gyroData, rs2_vector &accelData)
         * \param[in] gyroData A gyroscopic orientation 3-vector of type rs2_vector
         * \param[in] accelData An acceleration 3-vector of type rs2_vector
         * \returns A FusionEulerAngles structure containing the Roll, Pitch, and Yaw in degrees
         */
        FusionEulerAngles fuse(rs2_vector &gyroData, rs2_vector &accelData);

    protected:
        FusionVector3 gyroSensitivity;
        FusionVector3 accelSensitivity;
        FusionVector3 hardIronBias;

        FusionBias bias;
        FusionAhrs ahrs;

        float samplePeriod;
};

#endif
