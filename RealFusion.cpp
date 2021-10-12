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
 * \file RealFusion.cpp
 * \author Edward Janne
 */

#include "RealFusion.hpp"

RealFusion::RealFusion() throw(Exception)
: pl(), fuser()
{
    // Ensure an IMU is available. If not, throw exception.
    if(!checkIMU()) throw Exception(__FILE__, __LINE__, string("RealSense::Realsense"), 0, string("IMU not available"));

    // Configure librealsense2 to enable gyro and accel streams
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_GYRO, RS2_FORMAT_MOTION_XYZ32F);
    cfg.enable_stream(RS2_STREAM_ACCEL, RS2_FORMAT_MOTION_XYZ32F);

    // Start streaming
    pl.start(cfg);
}

bool RealFusion::checkIMU()
{
    bool gyro, accel;
    rs2::context ctx;
    // Loop through all connected devices
    for(auto dev : ctx.query_devices()) {
        gyro = accel = false;
        // Loop through all sensors on each device
        for(auto sensor : dev.query_sensors()) {
            // Loop through all profiles on each sensor
            for(auto profile : sensor.get_stream_profiles()) {
                switch(profile.stream_type()) {
                    // If the profile specifies this as a gyro stream eureka
                    case RS2_STREAM_GYRO:
                        gyro = true;
                        break;
                    // If the profile specifies this as an accel stream eureka
                    case RS2_STREAM_ACCEL:
                        accel = true;
                        break;
                }
            }
        }
        return (gyro && accel);
    }
    return false;
}

void RealFusion::tick()
{
    // Wait for frames
    rs2::frameset frames = pl.wait_for_frames();

    rs2_vector gyroData, accelData;
    bool gyroObtained = false, accelObtained = false;

    // Loop through all returned frames
    for(auto frame : frames) {
        // If this is a motion frame
        if(frame.is<rs2::motion_frame>()) {
            auto motion = frame.as<rs2::motion_frame>();
            auto profile = motion.get_profile();

            // If this is a gyro frame, get the gyro data
            if(motion && profile.stream_type() == RS2_STREAM_GYRO && profile.format() == RS2_FORMAT_MOTION_XYZ32F) {
                gyroObtained = true;
                gyroData = motion.get_motion_data();
            }

            // If this is a accel frame, get the accel data
            if(motion && profile.stream_type() == RS2_STREAM_ACCEL && profile.format() == RS2_FORMAT_MOTION_XYZ32F) {
                accelObtained = true;
                accelData = motion.get_motion_data();
            }
        }
    }

    // If both types of telemetry are available
    if(gyroObtained && accelObtained) {
        // Call Madgwick Fusion to fuse the data into Euler orientations.
        angles = fuser.fuse(gyroData, accelData);
    }
}

