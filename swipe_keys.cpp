
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <chrono>
#include <thread>
#include <iostream>


constexpr int SWIPE_PX = 150;      
constexpr int STEPS    = 15;       
constexpr int STEP_MS  = 10;       

void send_mouse_down() {
    INPUT input{};
    input.type = INPUT_MOUSE;
    input.mi.dwFlags = MOUSEEVENTF_LEFTDOWN;
    SendInput(1, &input, sizeof(INPUT));
}

void send_mouse_up() {
    INPUT input{};
    input.type = INPUT_MOUSE;
    input.mi.dwFlags = MOUSEEVENTF_LEFTUP;
    SendInput(1, &input, sizeof(INPUT));
}

void send_mouse_move_rel(int dx, int dy) {
    INPUT input{};
    input.type = INPUT_MOUSE;
    input.mi.dwFlags = MOUSEEVENTF_MOVE;
    input.mi.dx = dx;
    input.mi.dy = dy;
    SendInput(1, &input, sizeof(INPUT));
}

void swipe(int dx_total, int dy_total) {
    POINT orig;
    GetCursorPos(&orig);

    send_mouse_down();
    for (int i = 0; i < STEPS; ++i) {
        int dx = dx_total / STEPS;
        int dy = dy_total / STEPS;
        send_mouse_move_rel(dx, dy);
        std::this_thread::sleep_for(std::chrono::milliseconds(STEP_MS));
    }
    send_mouse_up();

    SetCursorPos(orig.x, orig.y);
    

    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

bool key_down(int vk) {
    return (GetAsyncKeyState(vk) & 0x8000) != 0;
}

int main() {

    bool prevLeft = false, prevRight = false, prevUp = false, prevDown = false;

    while (true) {
        if (key_down(VK_ESCAPE)) break;

        bool left  = key_down(VK_LEFT);
        bool right = key_down(VK_RIGHT);
        bool up    = key_down(VK_UP);
        bool down  = key_down(VK_DOWN);

        if (left  && !prevLeft)  swipe(-SWIPE_PX, 0);
        if (right && !prevRight) swipe( SWIPE_PX, 0);
        if (up    && !prevUp)    swipe(0, -SWIPE_PX);
        if (down  && !prevDown)  swipe(0,  SWIPE_PX);

        prevLeft  = left;
        prevRight = right;
        prevUp    = up;
        prevDown  = down;

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return 0;
}