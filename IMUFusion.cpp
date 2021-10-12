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
 * \file IMUFusion.cpp
 * \author Edward Janne
 */

#include "IMUFusion.hpp"

IMUFusion::IMUFusion(float iSamplePeriod, float iStationaryThreshold, float iAhrsGain, float iGyroSensitivity, float iAccelSensitivity)
: gyroSensitivity(), accelSensitivity(), hardIronBias(), bias(), ahrs(), samplePeriod(iSamplePeriod)
{
    // Initialise gyroscope bias correction algorithm
    FusionBiasInitialise(&bias, iStationaryThreshold, samplePeriod); // stationary threshold = 0.5 degrees per second
    
    // Initialise AHRS algorithm
    FusionAhrsInitialise(&ahrs, iAhrsGain); // gain = 0.5

    // Set optional magnetic field limits
    FusionAhrsSetMagneticField(&ahrs, 20.0f, 70.0f); // valid magnetic field range = 20 uT to 70 uT

    gyroSensitivity.axis.x = iGyroSensitivity;
    gyroSensitivity.axis.y = iGyroSensitivity;
    gyroSensitivity.axis.z = iGyroSensitivity;

    accelSensitivity.axis.x = iAccelSensitivity;
    accelSensitivity.axis.y = iAccelSensitivity;
    accelSensitivity.axis.z = iAccelSensitivity;

    hardIronBias.axis.x = 0.0f;
    hardIronBias.axis.y = 0.0f;
    hardIronBias.axis.z = 0.0f;
}

FusionEulerAngles IMUFusion::fuse(rs2_vector &gyroData, rs2_vector &accelData)
{
    // Raw to calibrated gyro data
    FusionVector3 uncalibratedGyroscope = {
        gyroData.x,
        gyroData.y,
        gyroData.z
    };
    FusionVector3 calibratedGyroscope = FusionCalibrationInertial(uncalibratedGyroscope, FUSION_ROTATION_MATRIX_IDENTITY, gyroSensitivity, FUSION_VECTOR3_ZERO);

    // Raw to calibrated accel data
    FusionVector3 uncalibratedAccelerometer = {
        accelData.x,
        accelData.y,
        accelData.z
    };
    FusionVector3 calibratedAccelerometer = FusionCalibrationInertial(uncalibratedAccelerometer, FUSION_ROTATION_MATRIX_IDENTITY, accelSensitivity, FUSION_VECTOR3_ZERO);

    // Magnetometer not available on RealSense cameras, just set to arbitrary value
    FusionVector3 dummyMagnetometer = {
        0.0f,
        0.0f,
        0.0f
    };

    // Update gyroscope bias correction algorithm
    calibratedGyroscope = FusionBiasUpdate(&bias, calibratedGyroscope);
    
    // Update AHRS algorithm
    FusionAhrsUpdate(&ahrs, calibratedGyroscope, calibratedAccelerometer, dummyMagnetometer, samplePeriod);

    // Compute Euler angles
    return FusionQuaternionToEulerAngles(FusionAhrsGetQuaternion(&ahrs));
}

