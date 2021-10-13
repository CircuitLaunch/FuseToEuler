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

rs2::pointcloud *RealFusion::pcPtr = new rs2::pointcloud();

RealFusion::RealFusion() throw(Exception)
: pl(), fuser(), pc(*pcPtr)
{
    // Ensure an IMU is available. If not, throw exception.
    if(!checkIMU()) throw Exception(__FILE__, __LINE__, string("RealSense::Realsense"), 0, string("IMU not available"));

    // Configure librealsense2 to enable gyro and accel streams
    rs2::config cfg;
    cfg.enable_stream(RS2_STREAM_DEPTH);
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

        const rs2::depth_frame depth = frames.get_depth_frame();
        rs2::points p = pc.calculate(depth);
        auto width = depth.get_width();
        auto height = depth.get_height();
        const rs2::vertex *v = p.get_vertices();

        vec3 *n = new vec3[width * height];
        computeNormals(v, width, height, 20, n);
        delete [] n;
    }
}

const int clockwise_i[8] = { 0, 1, 2, 5, 8, 7, 6, 3 };

vec3 operator-(const rs2::vertex &v1, const rs2::vertex &v2)
{
    return vec3(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}

rs2::vertex operator+(const rs2::vertex &vtx, vec3 &vec)
{
    rs2::vertex newVtx(vtx);
    newVtx.x += vec[0];
    newVtx.y += vec[1];
    newVtx.z += vec[2];
    return newVtx;
}

void RealFusion::computeNormals(const rs2::vertex *vtxArray, int frameWidth, int frameHeight, int delta, vec3 *normArray)
{   
    vec3 tanArray[9];
    bool avail[9];

    for(size_t y = frameHeight - delta - 1; y >= delta; y--) {
        for(size_t x = frameWidth - delta - 1; x >= delta; x--) {
            size_t thisPoint = y * frameWidth + x;
            normArray[thisPoint] = vec3();
            if(vtxArray[thisPoint].z != 0.0) {
                
                int n = 9;
                // cout << "Computing tangents for point " << thisPoint << ": ";
                while(n--) {
                    if(n == 4) continue;
                    int i = x + delta * (n % 3 - 1);
                    int j = y + delta * (n / 3 - 1);
                    int thatPoint = j * frameWidth + i;
                    if((avail[n] = (vtxArray[thatPoint].z != 0.0))) {
                        tanArray[n] = vtxArray[thatPoint] - vtxArray[thisPoint];
                        // cout << "  (" << tanArray[n].x << "," << tanArray[n].y << "," << tanArray[n].z << ")";
                    }
                }
                
                n = 8;
                int n2;
                int count = 0;
                while(n) {
                    while(n && !avail[clockwise_i[--n]]);
                    if(n) {
                        n2 = n;
                        while(n2 && ((n-n2+1) < 4) && !avail[clockwise_i[--n2]]);
                        if(avail[clockwise_i[n2]]) {
                            vec3 orthogonal(tanArray[clockwise_i[n]] ^ tanArray[clockwise_i[n2]]);
                            float mag = orthogonal.mag();
                            if(mag > 0.0) {
                                orthogonal /= mag;
                                normArray[thisPoint] += orthogonal;
                                count++;
                            }
                            n = n2;
                        } else n = 0;
                    }
                }
                if(count) {
                    normArray[thisPoint] /= (float) count;
                }
                
            }
        }
    }
}
