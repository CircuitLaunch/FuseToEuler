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
 * \file main.cpp
 * \author Edward Janne
 * 
 * \mainpage FuseToEuler Main
 * \section intro_sec Introduction
 * \par
 * Many RealSense cameras come with an embedded 6DOF Inertial Motion 
 * Unit (IMU) which measures rotation with a gyroscope and acceleration 
 * with an accelerometer. Unfortunately, with the exception of the 
 * RealSense T265 tracking camera, this data is unfused, meaning that the 
 * data from each is NOT used to compensate for noise in the other. For 
 * example, when used naïvely, the raw orientation data is affected by 
 * any camera motion because there is no inherent way for the sensor to 
 * know whether an applied force is from gravity or another motive force.
 * \par
 * Fusion algorithms "merge" the telemetry from independent sensor 
 * streams to create more stable  results.
 * \par
 * The Madgwick algorithm, named after Sebastian Madgwick who 
 * developed this algorithm in 2009, is one of the most popular ones. 
 * Fortunately for us, there is an open source library which implements 
 * this algorithm, ready to plunk into our code. In this tutorial, I will 
 * outline how to use it to derive the Euler orientation from the 
 * RealSense gyroscope and accelerometer data.
 * \par
 * I am programming in C/C++ on a Jetson AGX Xavier and using a 
 * RealSense L515 LiDAR camera.
 * \par
 * If you haven't set up `librealsense2` yet, please see this 
 * [tutorial](https://www.notion.so/How-to-install-librealsense-and-pylibrealsense-on-Jetson-5b909aeb1b6c409fb21464f2db869d41) 
 * on installing it.
 * \section download_sec Obtaining the Madgwick Library
 * \par
 * Clone the repo from github.
 * \code{.sh}
 * cd ~/Documents
 * git clone https://github.com/xioTechnologies/Fusion.git Fusion-master
 * \endcode
 * \par
 * I did not create a library from this source, but rather compiled and 
 * linked the source code directly into my program.
 * \par
 * Copy or move the Fusion directory from Fusion-master to your project 
 * directory. My project is contained in a folder named "FuseToEuler".
 * \code{.sh}
 * mkdir FuseToEuler
 * cp -R Fusion-master/Fusion FuseToEuler
 * \endcode
 */

#include <iostream>
#include <csignal>
#include <chrono>
#include <atomic>

#include "RealFusion.hpp"

using namespace std;
using namespace std::chrono;

// Create an interrupt safe variable
atomic<bool> alive{true}; ///< Atomically accessed variable shared between main process and interrupt handler. Enables interrupt handler to signal that the process should terminate.

/**
 * \fn void sigintHandler(int iSigNum)
 * \brief Handler function for SIGINT
 * \details The address of this function is attached as the SIGINT 
 * interrupt handler in the main function. It sets the atomic variable, 
 * alive, to false.
 */
void sigintHandler(int iSigNum)
{
    alive = false;
}

/**
 * \fn int main(int argc, char **argv)
 * \brief Execution starts here.
 * \details Installs an interrupt handler to intercept Ctrl-C interrupts. 
 * Then instantiates a RealFusion class, and enters an infinite loop in
 * which it repeatedly calls RealFusion::tick() to get IMU telemetry and
 * fuse it to derive stable Euler orientation. Every second, the current
 * orientation is printed to the console.
 */
int main(int argc, char **argv)
{
    // Install custom SIGINT handler
    signal(SIGINT, sigintHandler);

    try {
        cout << "FuseToEuler initializing..." << endl;

        // Instantiate RealFusion class to initialize Madwick Fusion and librealsense2
        RealFusion r;

        // Get current time reference
        auto start = high_resolution_clock::now();

        cout << "Type Ctl-C to terminate." << endl;
        cout << "Starting telemetry stream..." << endl;

        // Loop until user presses Ctl-C
        while(alive) {

            // Call RealFusion::tick() to get IMU frame and fuse motion telemetry
            r.tick();

            // Get current time reference
            auto timestamp = high_resolution_clock::now();

            // Throttle output to 1 second frequency
            if(duration_cast<milliseconds>(timestamp - start) > milliseconds(1000)) {
                cout << "R: " << r.angles.angle.roll << ", P: " << r.angles.angle.pitch << ", Y: " << r.angles.angle.yaw << endl;
                start = timestamp;
            }
        }

    // Catch any exceptions
    } catch(const Exception &e) {
        cerr << e;
        return -1;
    } catch(const rs2::error &e) {
        cerr << "RealSense error in " << e.get_failed_function() << "(" << e.get_failed_args() << "): " << e.what() << endl;
        return -1;
    } catch(const exception &e) {
        cerr << "std::exception: " << e.what() << endl;
        return -1;
    }

    // Exit cleanly
    cout << "\n\nSIGINT intercepted. Terminating." << endl;

    return 0;
}
