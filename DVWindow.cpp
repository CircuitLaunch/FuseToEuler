#include <GL/freeglut.h>

#include "DVWindow.hpp"
#include "DepthViewer.hpp"

DVWindow::DVWindow(DepthViewer &iApp)
: Window((Application &) iApp), fusion()
{ }

void DVWindow::keyUp(unsigned char ascii, int x, int y)
{
    if(ascii == 17) // Ctrl-q
        glutLeaveMainLoop();
}

void DVWindow::draw()
{
    fusion.tick();
    glClear(GL_COLOR_BUFFER_BIT);
    glutSwapBuffers();
}