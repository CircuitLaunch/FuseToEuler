#include "Window.hpp"
#include "Application.hpp"

#include <GL/freeglut.h>
#include <GL/freeglut_ucall.h>

#include <iostream>

using namespace std;

void _didReshape(int x, int y, void *iWindow)
{
    ((Window *) iWindow)->didReshape(x, y);
}

void _didMove(int x, int y, void *iWindow)
{
    ((Window *) iWindow)->didMove(x, y);
}

void _willClose(void *iWindow)
{
    ((Window *) iWindow)->willClose();
}

void _keyDown(unsigned char ascii, int x, int y, void *iWindow)
{
    ((Window *) iWindow)->keyDown(ascii, x, y);
}

void _keyUp(unsigned char ascii, int x, int y, void *iWindow)
{
    ((Window *) iWindow)->keyUp(ascii, x, y);
}

void _specialDown(int key, int x, int y, void *iWindow)
{
    ((Window *) iWindow)->specialDown(key, x, y);
}

void _specialUp(int key, int x, int y, void *iWindow)
{
    ((Window *) iWindow)->specialUp(key, x, y);
}

void _mouseMove(int x, int y, void *iWindow)
{
    ((Window *) iWindow)->mouseMove(x, y);
}

void _mouse(int button, int state, int x, int y, void *iWindow)
{
    Window::MOUSE_BUTTON flag;
    switch(button) {
        case GLUT_LEFT_BUTTON:
            flag = Window::PRIMARY;
            break;
        case GLUT_RIGHT_BUTTON:
            flag = Window::SECONDARY;
            break;
        case GLUT_MIDDLE_BUTTON:
            flag = Window::TERTIARY;
            break;
    }

    switch(state) {
        case GLUT_DOWN:
            ((Window *) iWindow)->setButtonState(flag);
            ((Window *) iWindow)->mouseDown(flag, x, y);
            break;
        case GLUT_UP:
            ((Window *) iWindow)->clearButtonState(flag);
            ((Window *) iWindow)->mouseUp(flag, x, y);
            break;
    }
}

void _mouseDrag(int x, int y, void *iWindow)
{
    ((Window *) iWindow)->mouseDrag(x, y);
}

void _mouseSense(int iState, void *iWindow)
{
    switch(iState) {
        case GLUT_ENTERED:
            ((Window *) iWindow)->mouseEntered();
            break;
        case GLUT_LEFT:
            ((Window *) iWindow)->mouseExited();
            break;
    }
}

void _draw(void *iWindow)
{
    ((Window *) iWindow)->draw();
}

void _drawOverlay(void *iWindow)
{
    ((Window *) iWindow)->drawOverlay();
}

Window::Window(Application &iApp)
: app(iApp)
{
    windowId = glutCreateWindow("");
    glutSetWindow(windowId);

    glutReshapeFuncUcall(_didReshape, (void *) this);
    glutPositionFuncUcall(_didMove, (void *) this);

    glutCloseFuncUcall(_willClose, (void *) this);

    glutKeyboardFuncUcall(_keyDown, (void *) this);
    glutKeyboardUpFuncUcall(_keyUp, (void *) this);

    glutSpecialFuncUcall(_specialDown, (void *) this);
    glutSpecialUpFuncUcall(_specialUp, (void *) this);

    glutPassiveMotionFuncUcall(_mouseMove, (void *) this);
    glutMouseFuncUcall(_mouse, (void *) this);
    
    glutMotionFuncUcall(_mouseDrag, (void *) this);

    glutDisplayFuncUcall(_draw, (void *) this);
    glutOverlayDisplayFuncUcall(_drawOverlay, (void *) this);

    glutEntryFuncUcall(_mouseSense, (void *) this);
}

Window::~Window()
{
    // glutDestroyWindow(windowId);
}

void Window::setTitle(const string &iTitle)
{
    glutSetWindow(windowId);
    glutSetWindowTitle(iTitle.c_str());
}

void Window::setPosition(int x, int y)
{
    glutSetWindow(windowId);
    glutPositionWindow(x, y);
}

void Window::setShape(int w, int h)
{
    glutSetWindow(windowId);
    glutReshapeWindow(w, h);
}

void Window::didReshape(int w, int h)
{
}

void Window::didMove(int x, int y)
{
}

void Window::willClose()
{
    app.removeWindow(this);
}

void Window::keyDown(unsigned char iKey, int x, int y)
{
}

void Window::keyUp(unsigned char iKey, int x, int y)
{
}

void Window::specialDown(int iKey, int x, int y)
{
}

void Window::specialUp(int iKey, int x, int y)
{
}

void Window::mouseMove(int x, int y)
{
}

void Window::mouseDrag(int x, int y)
{
}

void Window::setButtonState(MOUSE_BUTTON iButtonFlag)
{
    mouseButtonState |= iButtonFlag;
}

void Window::clearButtonState(MOUSE_BUTTON iButtonFlag)
{
    mouseButtonState &= ~iButtonFlag;
}

void Window::mouseDown(MOUSE_BUTTON iButton, int x, int y)
{
}

void Window::mouseUp(MOUSE_BUTTON iButton, int x, int y)
{
}

void Window::mouseEntered()
{
}

void Window::mouseExited()
{
}

void Window::draw()
{
    glClear(GL_COLOR_BUFFER_BIT);
    glutSwapBuffers();
}

void Window::drawOverlay()
{
}

