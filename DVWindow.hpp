#ifndef __DVWINDOW_HPP__
#define __DVWINDOW_HPP__

#include "Window.hpp"
#include "RealFusion.hpp"

class DepthViewer;

class DVWindow : public Window
{
    public:
        DVWindow(DepthViewer &iApp);

        virtual void keyUp(unsigned char ascii, int x, int y);

        virtual void draw();

    protected:
        RealFusion fusion;
};

#endif
