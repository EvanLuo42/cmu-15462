#pragma once

#include <SDL.h>

#include "../app.h"
#include "../lib/mathlib.h"

class Platform {
public:
	Platform();
	~Platform();

	void loop(App& app);

	Vec2 window_draw();
	Vec2 window_size();
	Vec2 scale(Vec2 pt);

	void capture_mouse();
	void release_mouse();
	void set_mouse(Vec2 pos);
	Vec2 get_mouse();
	void grab_mouse();
	void ungrab_mouse();
	bool is_down(SDL_Scancode key);

	static void init_console();
	static void remove_console();
	static uint32_t console_width();
	static void strcpy(char* dest, const char* src, size_t limit);


	inline static float force_dpi = std::numeric_limits< float >::quiet_NaN();

private:
	float prev_dpi = 0.0f;
	float prev_scale = 0.0f;

	void set_dpi();
	void platform_init();
	void platform_shutdown();
	void begin_frame();
	void complete_frame();

	SDL_Window* window = nullptr;
	SDL_GLContext gl_context = nullptr;
	const uint8_t* keybuf = nullptr;
};
