#include "Game.hpp"

#include "data_path.hpp"  //helper to get paths relative to executable
#include "gl_errors.hpp"  //helper for dumpping OpenGL error messages
#include "read_chunk.hpp"  //helper for reading a vector of structures from a file

#include <glm/gtc/type_ptr.hpp>

#include <cstddef>
#include <ctime>
#include <fstream>
#include <iostream>
#include <map>
#include <random>

// helper defined later; throws if shader compilation fails:
static GLuint compile_shader(GLenum type, std::string const &source);

static const std::unordered_map<HEX_DIR, double> dir_to_rad{
    {HEX_DIR::EAST, 0.0f},
    {HEX_DIR::NORTHEAST, M_PI / 3.0},
    {HEX_DIR::NORTHWEST, 2.0 * M_PI / 3.0},
    {HEX_DIR::WEST, M_PI},
    {HEX_DIR::SOUTHWEST, 4.0 * M_PI / 3.0},
    {HEX_DIR::SOUTHEAST, 5.0 * M_PI / 3.0}};

static const std::unordered_map<HEX_DIR, HEX_DIR> clockwise{
    {HEX_DIR::EAST, HEX_DIR::SOUTHEAST},
    {HEX_DIR::SOUTHEAST, HEX_DIR::SOUTHWEST},
    {HEX_DIR::SOUTHWEST, HEX_DIR::WEST},
    {HEX_DIR::WEST, HEX_DIR::NORTHWEST},
    {HEX_DIR::NORTHWEST, HEX_DIR::NORTHEAST},
    {HEX_DIR::NORTHEAST, HEX_DIR::EAST}};

static const std::unordered_map<HEX_DIR, HEX_DIR> counterclockwise{
    {HEX_DIR::EAST, HEX_DIR::NORTHEAST},
    {HEX_DIR::NORTHEAST, HEX_DIR::NORTHWEST},
    {HEX_DIR::NORTHWEST, HEX_DIR::WEST},
    {HEX_DIR::WEST, HEX_DIR::SOUTHWEST},
    {HEX_DIR::SOUTHWEST, HEX_DIR::SOUTHEAST},
    {HEX_DIR::SOUTHEAST, HEX_DIR::EAST}};

static const std::vector<HEX_DIR> directions{
    {HEX_DIR::EAST, HEX_DIR::NORTHEAST, HEX_DIR::NORTHWEST, HEX_DIR::WEST,
     HEX_DIR::SOUTHWEST, HEX_DIR::SOUTHEAST}};

Game::Game()
    : generator(std::time(nullptr)),
      distribution_x(0, board_size.x - 1),
      distribution_y(0, board_size.y - 1) {
  {  // create an opengl program to perform sun/sky (well,
     // directional+hemispherical) lighting:
    GLuint vertex_shader = compile_shader(
        GL_VERTEX_SHADER,
        "#version 330\n"
        "uniform mat4 object_to_clip;\n"
        "uniform mat4x3 object_to_light;\n"
        "uniform mat3 normal_to_light;\n"
        "layout(location=0) in vec4 Position;\n"  // note: layout keyword used
                                                  // to make sure that the
                                                  // location-0 attribute is
                                                  // always bound to something
        "in vec3 Normal;\n"
        "in vec4 Color;\n"
        "out vec3 position;\n"
        "out vec3 normal;\n"
        "out vec4 color;\n"
        "void main() {\n"
        "	gl_Position = object_to_clip * Position;\n"
        "	position = object_to_light * Position;\n"
        "	normal = normal_to_light * Normal;\n"
        "	color = Color;\n"
        "}\n");

    GLuint fragment_shader = compile_shader(
        GL_FRAGMENT_SHADER,
        "#version 330\n"
        "uniform vec3 sun_direction;\n"
        "uniform vec3 sun_color;\n"
        "uniform vec3 sky_direction;\n"
        "uniform vec3 sky_color;\n"
        "in vec3 position;\n"
        "in vec3 normal;\n"
        "in vec4 color;\n"
        "out vec4 fragColor;\n"
        "void main() {\n"
        "	vec3 total_light = vec3(0.0, 0.0, 0.0);\n"
        "	vec3 n = normalize(normal);\n"
        "	{ //sky (hemisphere) light:\n"
        "		vec3 l = sky_direction;\n"
        "		float nl = 0.5 + 0.5 * dot(n,l);\n"
        "		total_light += nl * sky_color;\n"
        "	}\n"
        "	{ //sun (directional) light:\n"
        "		vec3 l = sun_direction;\n"
        "		float nl = max(0.0, dot(n,l));\n"
        "		total_light += nl * sun_color;\n"
        "	}\n"
        "	fragColor = vec4(color.rgb * total_light, color.a);\n"
        "}\n");

    simple_shading.program = glCreateProgram();
    glAttachShader(simple_shading.program, vertex_shader);
    glAttachShader(simple_shading.program, fragment_shader);
    // shaders are reference counted so this makes sure they are freed after
    // program is deleted:
    glDeleteShader(vertex_shader);
    glDeleteShader(fragment_shader);

    // link the shader program and throw errors if linking fails:
    glLinkProgram(simple_shading.program);
    GLint link_status = GL_FALSE;
    glGetProgramiv(simple_shading.program, GL_LINK_STATUS, &link_status);
    if (link_status != GL_TRUE) {
      std::cerr << "Failed to link shader program." << std::endl;
      GLint info_log_length = 0;
      glGetProgramiv(simple_shading.program, GL_INFO_LOG_LENGTH,
                     &info_log_length);
      std::vector<GLchar> info_log(info_log_length, 0);
      GLsizei length = 0;
      glGetProgramInfoLog(simple_shading.program, GLsizei(info_log.size()),
                          &length, &info_log[0]);
      std::cerr << "Info log: "
                << std::string(info_log.begin(), info_log.begin() + length);
      throw std::runtime_error("failed to link program");
    }
  }

  {  // read back uniform and attribute locations from the shader program:
    simple_shading.object_to_clip_mat4 =
        glGetUniformLocation(simple_shading.program, "object_to_clip");
    simple_shading.object_to_light_mat4x3 =
        glGetUniformLocation(simple_shading.program, "object_to_light");
    simple_shading.normal_to_light_mat3 =
        glGetUniformLocation(simple_shading.program, "normal_to_light");

    simple_shading.sun_direction_vec3 =
        glGetUniformLocation(simple_shading.program, "sun_direction");
    simple_shading.sun_color_vec3 =
        glGetUniformLocation(simple_shading.program, "sun_color");
    simple_shading.sky_direction_vec3 =
        glGetUniformLocation(simple_shading.program, "sky_direction");
    simple_shading.sky_color_vec3 =
        glGetUniformLocation(simple_shading.program, "sky_color");

    simple_shading.Position_vec4 =
        glGetAttribLocation(simple_shading.program, "Position");
    simple_shading.Normal_vec3 =
        glGetAttribLocation(simple_shading.program, "Normal");
    simple_shading.Color_vec4 =
        glGetAttribLocation(simple_shading.program, "Color");
  }

  struct Vertex {
    glm::vec3 Position;
    glm::vec3 Normal;
    glm::u8vec4 Color;
  };
  static_assert(sizeof(Vertex) == 28, "Vertex should be packed.");

  {  // load mesh data from a binary blob:
    std::ifstream blob(data_path("meshes.blob"), std::ios::binary);
    // The blob will be made up of three chunks:
    // the first chunk will be vertex data (interleaved position/normal/color)
    // the second chunk will be characters
    // the third chunk will be an index, mapping a name (range of characters) to
    // a mesh (range of vertex data)

    // read vertex data:
    std::vector<Vertex> vertices;
    read_chunk(blob, "dat0", &vertices);

    // read character data (for names):
    std::vector<char> names;
    read_chunk(blob, "str0", &names);

    // read index:
    struct IndexEntry {
      uint32_t name_begin;
      uint32_t name_end;
      uint32_t vertex_begin;
      uint32_t vertex_end;
    };
    static_assert(sizeof(IndexEntry) == 16, "IndexEntry should be packed.");

    std::vector<IndexEntry> index_entries;
    read_chunk(blob, "idx0", &index_entries);

    if (blob.peek() != EOF) {
      std::cerr << "WARNING: trailing data in meshes file." << std::endl;
    }

    // upload vertex data to the graphics card:
    glGenBuffers(1, &meshes_vbo);
    glBindBuffer(GL_ARRAY_BUFFER, meshes_vbo);
    glBufferData(GL_ARRAY_BUFFER, sizeof(Vertex) * vertices.size(),
                 vertices.data(), GL_STATIC_DRAW);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // create map to store index entries:
    std::map<std::string, Mesh> index;
    for (IndexEntry const &e : index_entries) {
      if (e.name_begin > e.name_end || e.name_end > names.size()) {
        throw std::runtime_error("invalid name indices in index.");
      }
      if (e.vertex_begin > e.vertex_end || e.vertex_end > vertices.size()) {
        throw std::runtime_error("invalid vertex indices in index.");
      }
      Mesh mesh;
      mesh.first = e.vertex_begin;
      mesh.count = e.vertex_end - e.vertex_begin;
      auto ret = index.insert(std::make_pair(
          std::string(names.begin() + e.name_begin, names.begin() + e.name_end),
          mesh));
      if (!ret.second) {
        throw std::runtime_error("duplicate name in index.");
      }
    }

    // look up into index map to extract meshes:
    auto lookup = [&index](std::string const &name) -> Mesh {
      auto f = index.find(name);
      if (f == index.end()) {
        throw std::runtime_error("Mesh named '" + name +
                                 "' does not appear in index.");
      }
      return f->second;
    };
    tile_mesh = lookup("board_tile");
    snake_body_mesh = lookup("snake_body");
    snake_head_mesh = lookup("snake_head");
    goal_mesh = lookup("goal");
  }

  {  // create vertex array object to hold the map from the mesh vertex buffer
     // to shader program attributes:
    glGenVertexArrays(1, &meshes_for_simple_shading_vao);
    glBindVertexArray(meshes_for_simple_shading_vao);
    glBindBuffer(GL_ARRAY_BUFFER, meshes_vbo);
    // note that I'm specifying a 3-vector for a 4-vector attribute here, and
    // this is okay to do:
    glVertexAttribPointer(simple_shading.Position_vec4, 3, GL_FLOAT, GL_FALSE,
                          sizeof(Vertex),
                          (GLbyte *)0 + offsetof(Vertex, Position));
    glEnableVertexAttribArray(simple_shading.Position_vec4);
    if (simple_shading.Normal_vec3 != -1U) {
      glVertexAttribPointer(simple_shading.Normal_vec3, 3, GL_FLOAT, GL_FALSE,
                            sizeof(Vertex),
                            (GLbyte *)0 + offsetof(Vertex, Normal));
      glEnableVertexAttribArray(simple_shading.Normal_vec3);
    }
    if (simple_shading.Color_vec4 != -1U) {
      glVertexAttribPointer(simple_shading.Color_vec4, 4, GL_UNSIGNED_BYTE,
                            GL_TRUE, sizeof(Vertex),
                            (GLbyte *)0 + offsetof(Vertex, Color));
      glEnableVertexAttribArray(simple_shading.Color_vec4);
    }
    glBindBuffer(GL_ARRAY_BUFFER, 0);
  }

  GL_ERRORS();

  //----------------
  // set up game board with meshes and rolls:

  snake_body_pos = {glm::ivec2(board_size.x / 2, board_size.y / 2)};
  //  snake_body_pos = {glm::ivec2(5, 0), glm::ivec2(5, 1), glm::ivec2(5, 2),
  //                    glm::ivec2(5, 3), glm::ivec2(5, 4), glm::ivec2(5, 5),
  //                    glm::ivec2(5, 6), glm::ivec2(5, 7), glm::ivec2(5, 8)};
  goal_pos = generate_goal_pos();
}

Game::~Game() {
  glDeleteVertexArrays(1, &meshes_for_simple_shading_vao);
  meshes_for_simple_shading_vao = -1U;

  glDeleteBuffers(1, &meshes_vbo);
  meshes_vbo = -1U;

  glDeleteProgram(simple_shading.program);
  simple_shading.program = -1U;

  GL_ERRORS();
}

bool Game::handle_event(SDL_Event const &evt, glm::uvec2 window_size) {
  // ignore any keys that are the result of automatic key repeat:
  if (evt.type == SDL_KEYDOWN && evt.key.repeat) {
    return false;
  }

  // move cursor on L/R/U/D press:
  if (evt.type == SDL_KEYDOWN && evt.key.repeat == 0 && !game_over) {
    if (evt.key.keysym.scancode == SDL_SCANCODE_LEFT) {
      snake_dir = clockwise.at(snake_dir);
      return true;
    } else if (evt.key.keysym.scancode == SDL_SCANCODE_RIGHT) {
      snake_dir = counterclockwise.at(snake_dir);
      return true;
    }
  }
  return false;
}

void Game::update(float elapsed) {
  static float time_since_update = 0.0f;
  if (time_since_update > 1.0f && !game_over) {
    time_since_update = 0.0f;
    glm::ivec2 new_pos = get_adjacent_hex(snake_body_pos[0], snake_dir);

    if (new_pos.x >= 0 && new_pos.y >= 0 && new_pos.x < board_size.x &&
        new_pos.y < board_size.y &&
        std::find(snake_body_pos.begin(), snake_body_pos.end(), new_pos) ==
            snake_body_pos.end() &&
        snake_body_pos.size() < (board_size.x * board_size.y)) {
      snake_body_pos.insert(snake_body_pos.begin(), new_pos);
      snake_body_pos.pop_back();

      // snake ate the goal, generate new goal, extend snake
      if (snake_body_pos[0] == goal_pos) {
        append_snake_body();
        goal_pos = generate_goal_pos();
      }
    } else {
      game_over = true;
      std::cout << "Game Over!" << std::endl;
      std::cout << "Final Score: " << snake_body_pos.size() << std::endl;
    }

  } else {
    time_since_update += elapsed;
  }
}

uint32_t Game::get_game_score() { return snake_body_pos.size() - 1; }

glm::ivec2 Game::get_adjacent_hex(glm::ivec2 pos, HEX_DIR dir) {
  glm::ivec2 new_pos;

  switch (dir) {
    case HEX_DIR::EAST:
      new_pos = pos + glm::ivec2(1.0, 0.0);
      break;
    case HEX_DIR::NORTHWEST:
      if (pos.y % 2 == 0) {
        new_pos = pos + glm::ivec2(0.0, -1.0);
      } else {
        new_pos = pos + glm::ivec2(-1.0, -1.0);
      }
      break;
    case HEX_DIR::NORTHEAST:
      if (pos.y % 2 == 0) {
        new_pos = pos + glm::ivec2(1.0, -1.0);
      } else {
        new_pos = pos + glm::ivec2(0.0, -1.0);
      }
      break;
    case HEX_DIR::WEST:
      new_pos = pos + glm::ivec2(-1.0, 0.0);
      break;
    case HEX_DIR::SOUTHEAST:
      if (pos.y % 2 == 0) {
        new_pos = pos + glm::ivec2(1.0, 1.0);
      } else {
        new_pos = pos + glm::ivec2(0.0, 1.0);
      }
      break;
    case HEX_DIR::SOUTHWEST:
      if (pos.y % 2 == 0) {
        new_pos = pos + glm::ivec2(0.0, 1.0);
      } else {
        new_pos = pos + glm::ivec2(-1.0, 1.0);
      }
      break;
  }

  return new_pos;
}

glm::ivec2 Game::generate_goal_pos() {
  int32_t x = distribution_x(generator);
  int32_t y = distribution_x(generator);

  while (std::find(snake_body_pos.begin(), snake_body_pos.end(),
                   glm::ivec2(x, y)) != snake_body_pos.end()) {
    x = distribution_x(generator);
    y = distribution_x(generator);
  }

  return glm::ivec2(x, y);
}

void Game::append_snake_body() {
  glm::ivec2 snake_tail = snake_body_pos.back();
  std::vector<HEX_DIR> valid_directions(directions);

  // don't generate tail at front of snake head
  if (snake_body_pos.size() == 1) {
    valid_directions.erase(
        std::find(valid_directions.begin(), valid_directions.end(), snake_dir));
  }

  for (const HEX_DIR dir : valid_directions) {
    glm::ivec2 new_pos = get_adjacent_hex(snake_tail, dir);

    if (new_pos.x >= 0 && new_pos.y >= 0 && new_pos.x < board_size.x &&
        new_pos.y < board_size.y &&
        std::find(snake_body_pos.begin(), snake_body_pos.end(), new_pos) ==
            snake_body_pos.end()) {
      snake_body_pos.push_back(new_pos);
      return;
    }
  }
}

void Game::draw(glm::uvec2 drawable_size) {
  // Set up a transformation matrix to fit the board in the window:
  glm::mat4 world_to_clip;
  {
    float aspect = float(drawable_size.x) / float(drawable_size.y);

    // want scale such that board * scale fits in [-aspect,aspect]x[-1.0,1.0]
    // screen box:
    float scale =
        glm::min(2.0f / float(board_size.x * hex_offset_x * 2),
                 2.0f * aspect / float(board_size.y * hex_offset_y * 3));

    // center of board will be placed at center of screen:
    glm::vec2 center = 0.5f * glm::vec2(float(board_size.x * hex_offset_x * 2),
                                        float(board_size.y * hex_offset_y * 3));

    // NOTE: glm matrices are specified in column-major order
    world_to_clip =
        glm::mat4(scale / aspect, 0.0f, 0.0f, 0.0f, 0.0f, scale, 0.0f, 0.0f,
                  0.0f, 0.0f, -1.0f, 0.0f, -(scale / aspect) * center.x,
                  -scale * center.y, 0.0f, 1.0f);
  }

  // set up graphics pipeline to use data from the meshes and the simple shading
  // program:
  glBindVertexArray(meshes_for_simple_shading_vao);
  glUseProgram(simple_shading.program);

  glUniform3fv(simple_shading.sun_color_vec3, 1,
               glm::value_ptr(glm::vec3(0.81f, 0.81f, 0.76f)));
  glUniform3fv(simple_shading.sun_direction_vec3, 1,
               glm::value_ptr(glm::normalize(glm::vec3(-0.2f, 0.2f, 1.0f))));
  glUniform3fv(simple_shading.sky_color_vec3, 1,
               glm::value_ptr(glm::vec3(0.2f, 0.2f, 0.3f)));
  glUniform3fv(simple_shading.sky_direction_vec3, 1,
               glm::value_ptr(glm::vec3(0.0f, 1.0f, 0.0f)));

  // helper function to draw a given mesh with a given transformation:
  auto draw_mesh = [&](Mesh const &mesh, glm::mat4 const &object_to_world) {
    // set up the matrix uniforms:
    if (simple_shading.object_to_clip_mat4 != -1U) {
      glm::mat4 object_to_clip = world_to_clip * object_to_world;
      glUniformMatrix4fv(simple_shading.object_to_clip_mat4, 1, GL_FALSE,
                         glm::value_ptr(object_to_clip));
    }
    if (simple_shading.object_to_light_mat4x3 != -1U) {
      glUniformMatrix4x3fv(simple_shading.object_to_light_mat4x3, 1, GL_FALSE,
                           glm::value_ptr(object_to_world));
    }
    if (simple_shading.normal_to_light_mat3 != -1U) {
      // NOTE: if there isn't any non-uniform scaling in the object_to_world
      // matrix, then the inverse transpose is the matrix itself, and computing
      // it wastes some CPU time:
      glm::mat3 normal_to_world =
          glm::inverse(glm::transpose(glm::mat3(object_to_world)));
      glUniformMatrix3fv(simple_shading.normal_to_light_mat3, 1, GL_FALSE,
                         glm::value_ptr(normal_to_world));
    }

    // draw the mesh:
    glDrawArrays(GL_TRIANGLES, mesh.first, mesh.count);
  };

  for (uint32_t y = 0; y < board_size.y; ++y) {
    for (uint32_t x = 0; x < board_size.x; ++x) {
      glm::vec2 current_offset = hex_to_pixel(x, y);
      draw_mesh(tile_mesh,
                glm::mat4(1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
                          0.0f, 1.0f, 0.0f, current_offset.x + 0.5f,
                          current_offset.y + 0.5f, -0.5f, 1.0f));
    }
  }

  for (uint32_t i = 0; i < snake_body_pos.size(); i++) {
    glm::vec2 current_offset =
        hex_to_pixel(snake_body_pos[i].x, snake_body_pos[i].y);
    if (i == 0) {
      draw_mesh(snake_head_mesh,
                glm::mat4(std::cos(dir_to_rad.at(snake_dir)),
                          -std::sin(dir_to_rad.at(snake_dir)), 0.0f, 0.0f,
                          std::sin(dir_to_rad.at(snake_dir)),
                          std::cos(dir_to_rad.at(snake_dir)), 0.0f, 0.0f, 0.0f,
                          0.0f, 1.0f, 0.0f, current_offset.x + 0.5f,
                          current_offset.y + 0.5f, 0.0f, 1.0f));
    } else {
      draw_mesh(snake_body_mesh,
                glm::mat4(1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f,
                          0.0f, 1.0f, 0.0f, current_offset.x + 0.5f,
                          current_offset.y + 0.5f, 0.0f, 1.0f));
    }
  }

  glm::vec2 goal_offset = hex_to_pixel(goal_pos.x, goal_pos.y);
  draw_mesh(goal_mesh, glm::mat4(1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,
                                 0.0f, 0.0f, 1.0f, 0.0f, goal_offset.x + 0.5f,
                                 goal_offset.y + 0.5f, 0.0f, 1.0f));

  glUseProgram(0);

  GL_ERRORS();
}

glm::vec2 Game::hex_to_pixel(int32_t x, int32_t y) {
  float current_offset_x;
  float current_offset_y = y * (3 * hex_offset_y);
  if (y % 2 == 0) {
    current_offset_x = x * hex_offset_x * 2 + hex_offset_x;
  } else {
    current_offset_x = x * hex_offset_x * 2;
  }

  return glm::vec2(current_offset_x, current_offset_y);
}

// create and return an OpenGL vertex shader from source:
static GLuint compile_shader(GLenum type, std::string const &source) {
  GLuint shader = glCreateShader(type);
  GLchar const *str = source.c_str();
  GLint length = GLint(source.size());
  glShaderSource(shader, 1, &str, &length);
  glCompileShader(shader);
  GLint compile_status = GL_FALSE;
  glGetShaderiv(shader, GL_COMPILE_STATUS, &compile_status);
  if (compile_status != GL_TRUE) {
    std::cerr << "Failed to compile shader." << std::endl;
    GLint info_log_length = 0;
    glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &info_log_length);
    std::vector<GLchar> info_log(info_log_length, 0);
    GLsizei length = 0;
    glGetShaderInfoLog(shader, GLsizei(info_log.size()), &length, &info_log[0]);
    std::cerr << "Info log: "
              << std::string(info_log.begin(), info_log.begin() + length);
    glDeleteShader(shader);
    throw std::runtime_error("Failed to compile shader.");
  }
  return shader;
}
