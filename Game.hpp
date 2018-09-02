#pragma once

#include "GL.hpp"

#include <SDL.h>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

enum class HEX_DIR { EAST, NORTHEAST, NORTHWEST, WEST, SOUTHWEST, SOUTHEAST };

// The 'Game' struct holds all of the game-relevant state,
// and is called by the main loop.

struct Game {
  // Game creates OpenGL resources (i.e. vertex buffer objects) in its
  // constructor and frees them in its destructor.
  Game();
  ~Game();

  // handle_event is called when new mouse or keyboard events are received:
  // (note that this might be many times per frame or never)
  // The function should return 'true' if it handled the event.
  bool handle_event(SDL_Event const &evt, glm::uvec2 window_size);

  // update is called at the start of a new frame, after events are handled:
  void update(float elapsed);

  // draw is called after update:
  void draw(glm::uvec2 drawable_size);

  glm::ivec2 generate_goal_pos();

  glm::vec2 hex_to_pixel(int32_t x, int32_t y);

  //------- opengl resources -------

  // shader program that draws lit objects with vertex colors:
  struct {
    GLuint program = -1U;  // program object

    // uniform locations:
    GLuint object_to_clip_mat4 = -1U;
    GLuint object_to_light_mat4x3 = -1U;
    GLuint normal_to_light_mat3 = -1U;
    GLuint sun_direction_vec3 = -1U;
    GLuint sun_color_vec3 = -1U;
    GLuint sky_direction_vec3 = -1U;
    GLuint sky_color_vec3 = -1U;

    // attribute locations:
    GLuint Position_vec4 = -1U;
    GLuint Normal_vec3 = -1U;
    GLuint Color_vec4 = -1U;
  } simple_shading;

  // mesh data, stored in a vertex buffer:
  GLuint meshes_vbo = -1U;  // vertex buffer holding mesh data

  // The location of each mesh in the meshes vertex buffer:
  struct Mesh {
    GLint first = 0;
    GLsizei count = 0;
  };

  Mesh tile_mesh;
  Mesh snake_body_mesh;
  Mesh snake_head_mesh;
  Mesh goal_mesh;

  GLuint meshes_for_simple_shading_vao =
      -1U;  // vertex array object that describes how to connect the meshes_vbo
            // to the simple_shading_program

  //------- game state -------

  glm::uvec2 board_size = glm::uvec2(10, 10);
  constexpr static const float hex_offset_x = 0.86602540378f;
  constexpr static const float hex_offset_y = 0.5f;
  std::vector<glm::ivec2> snake_body_pos;
  glm::ivec2 goal_pos;
  std::default_random_engine generator;
  std::uniform_int_distribution<int32_t> distribution_x, distribution_y;

  HEX_DIR snake_dir = HEX_DIR::EAST;

  struct {
    bool roll_left = false;
    bool roll_right = false;
    bool roll_up = false;
    bool roll_down = false;
  } controls;
};
