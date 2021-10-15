#ifndef __APPLICATION_HPP__
#define __APPLICATION_HPP__

#include <unordered_map>

#include "Window.hpp"

using namespace std;

class Application
{
    friend class Window;
    
    public:
        Application(int argc, char **argv);
        virtual ~Application();

        void run(unsigned int iUpdatePeriod);

        void tick();

        void redrawAll();

        Window &createWindow(const string &iTitle, int x, int y, int w, int h);

    protected:
        virtual Window *newWindow();
        virtual void removeWindow(Window *);

    protected:
        unordered_map<int, Window *> windows;
};

#endif
