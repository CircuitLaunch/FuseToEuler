#include <GL/gl.h>
#include <GL/glu.h>
// #include <GL/freeglut_std.h>
// #include <GL/freeglut_ext.h>
#include <GL/freeglut.h>

#include "Application.hpp"

void _tick(void *iApplication)
{
    ((Application *) iApplication)->tick();
}

Application::Application(int argc, char **argv)
: windows()
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA);
    glutInitWindowSize(2048, 1024);
    glutInitWindowPosition(100, 100);
}

Application::~Application()
{
    while(windows.size()) {
        unordered_map<int, Window *>::iterator i = windows.begin();
        Window *window = i->second;
        windows.erase(windows.begin());
        delete window;
    }
}

void Application::run(unsigned int iUpdatePeriod)
{
    glutIdleFuncUcall(_tick, (void *) this);
    glutMainLoop();
    glutIdleFuncUcall(nullptr, nullptr);
}

void Application::tick()
{
}

void Application::redrawAll()
{
    for(auto window : windows) {
        glutSetWindow(window.first);
        glutPostRedisplay();
    }
}

Window &Application::createWindow(const string &iTitle, int x, int y, int w, int h)
{
    Window *window = newWindow();

    window->setTitle(iTitle);
    window->setPosition(x, y);
    window->setShape(w, h);
    
    windows[window->windowId] = window;

    return *window;
}

Window *Application::newWindow()
{
    return new Window(*this);
}

void Application::removeWindow(Window *iWindow)
{
    unordered_map<int, Window *>::iterator i = windows.find(iWindow->windowId);
    if(i != windows.end()) {
        Window *window = i->second;
        windows.erase(i);
        delete window;
    }
}
