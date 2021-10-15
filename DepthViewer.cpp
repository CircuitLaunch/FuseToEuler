#include "DepthViewer.hpp"

#include "DVWindow.hpp"

DepthViewer::DepthViewer(int argc, char **argv)
: Application(argc, argv)
{ }

void DepthViewer::tick()
{
    redrawAll();
}

Window *DepthViewer::newWindow()
{
    return new DVWindow(*this);
}