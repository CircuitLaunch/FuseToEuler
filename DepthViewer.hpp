#ifndef __DEPTHVIEWER_HPP__
#define __DEPTHVIEWER_HPP__

#include "Application.hpp"
#include "Window.hpp"

class DepthViewer : public Application
{
    public:
        DepthViewer(int argc, char **argv);

        virtual void tick();

    protected:
        virtual Window *newWindow();
};

#endif
