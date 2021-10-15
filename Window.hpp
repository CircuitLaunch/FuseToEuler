#ifndef __WINDOW_HPP__
#define __WINDOW_HPP__

#include <string>

using namespace std;

class Application;

class Window
{
    friend class Application;

    public:
        typedef enum {
            PRIMARY = 1,
            SECONDARY = 2,
            TERTIARY = 4
        } MOUSE_BUTTON;

    public:
        Window(Application &iApp);
        virtual ~Window();

        virtual void setTitle(const string &iTitle);
        virtual void setPosition(int x, int y);
        virtual void setShape(int w, int h);

        virtual void didReshape(int w, int h);
        virtual void didMove(int x, int y);
        virtual void willClose();

        virtual void keyDown(unsigned char iKey, int x, int y);
        virtual void keyUp(unsigned char iKey, int x, int y);
        
        virtual void specialDown(int iKey, int x, int y);
        virtual void specialUp(int iKey, int x, int y);

        virtual void mouseMove(int x, int y);

        virtual void setButtonState(MOUSE_BUTTON iButtonFlag);
        virtual void clearButtonState(MOUSE_BUTTON iButtonFlag);

        virtual void mouseDown(MOUSE_BUTTON iButton, int x, int y);
        virtual void mouseUp(MOUSE_BUTTON iButton, int x, int y);

        virtual void mouseDrag(int x, int y);

        virtual void mouseEntered();
        virtual void mouseExited();

        virtual void draw();
        virtual void drawOverlay();

    protected:
        Application &app;
        int windowId;
        unsigned int mouseButtonState;
};

#endif
