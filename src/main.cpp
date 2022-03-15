#define GLEW_STATIC

#include "GL/glew.h"
#include "GLFW/glfw3.h"

#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

#include <cstdlib>
#include <cassert>
#include <cstdio>
#include <ctime>
#include <cstdint>

#include <chrono>

#include "malloc.h"
#include "windows.h"

typedef int8_t  int8;
typedef int16_t int16;
typedef int32_t int32;
typedef int64_t int64;

typedef uint8_t  uint8;
typedef uint16_t uint16;
typedef uint32_t uint32;
typedef uint64_t uint64;

typedef uint32 uint;

#include "filesystem_api.cpp"
#include "filesystem_windows.cpp"
#include "dynamic_array.cpp"



const double PI  = 3.14159265358979323846;
const double TAU = 6.28318530717958647692;
const double MAX_POSSIBLE_PACKING_FACTOR = PI * sqrt(3) / 6.0;

const uint NUMBER_OF_NEIGHBOURS = 32; // @Incomplete: actually this number should depend on L value.
const uint CELL_IS_NOT_OCCUPIED = 0xFFFFFFFF;

const float  left_boundary = -1.0f;
const float right_boundary =  1.0f;
const float lower_boundary = -1.0f;
const float upper_boundary =  1.0f;

#define max(x, y)        ( (x) > (y) ? (x) : (y) )
#define min(x, y)        ( (x) < (y) ? (x) : (y) )

#define clamp(w, mi, ma) ( min(max((w), (mi)), (ma)) )

#define square(x)     ( (x) * (x) )
#define round_down(x) ( (size_t)(round((float)(x) + 0.5f) - 1) )
#define array_size(x) ( sizeof( x )/sizeof( *(x) ) )
#define make_string(x) { (x), sizeof(x)-1 }

#define measure_scope() Timer ANONYMOUS_NAME

#define defer auto ANONYMOUS_NAME = Junk{} + [&]()
#define ANONYMOUS_NAME CONCAT(GAMMA, __LINE__)
#define CONCAT(A, B) CONCAT_IMPL(A, B)
#define CONCAT_IMPL(A, B) A##B

template<class T>
struct Defer {
  const T func;
  Defer(const T f) : func(f) {}
  ~Defer()         { func(); }
};

struct Junk {};
template<class T> inline const Defer<T> operator+(Junk, const T f) { return f; }

struct Timer {
  std::chrono::steady_clock::time_point start;
  std::chrono::steady_clock::time_point end;

  Timer()  {
    start = std::chrono::steady_clock::now();
  }

  ~Timer() { 
    end = std::chrono::steady_clock::now(); 
    double delta = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    if(delta < 1000.) {
      printf("%g ns\n", delta);
    } else if(delta >= 1000. && delta < 1000000.) {
      printf("%g us\n", delta/1000.);
    } else if(delta >= 1000000. && delta < 1000000000.) {
      printf("%g ms\n", delta/1000000.);
    } else {
      printf("%g s\n", delta/1000000000.);
    }
  }
};

struct string {
  char*  data;
  size_t count;
};

struct Vertex_And_Fragment_Shader_Sources {
  string vertex;
  string fragment;
};

struct Vec2 {
  float x, y;
};

Vec2 operator+(Vec2 a, Vec2 b)  { return { a.x + b.x, a.y + b.y }; }
Vec2 operator*(float c, Vec2 a) { return { c * a.x, c * a.y }; }
Vec2 operator*(Vec2 a, float c) { return c * a; }

struct Grid_Position {
  uint i, j;
};

typedef uint Grid_Cell;
struct Grid2D {
  Grid_Cell* data;

  size_t number_of_cells_per_dimension;
  size_t number_of_cells;

  // all cells are (cell_size x cell_size)
  float cell_size;

};

struct Graph {
  uint   count;

  uint8* connected_nodes_count;
  uint** connected_nodes;
  // 
  // @Incomplete: instead of using a pointer: uint**, we can go with just a uint* and address different nodes with indices, that will take 2 times less memory.
  // Approximately uint can address 4 billions of nodes, so for example if there is 6 connections per node, graph.count should not be greater than 700 million. So we can't address more circles than that number. (with a uint!) (pointers are fine).
  // 

  // for graph creation.
  uint*  graph_data;
};

struct Queue {
  uint* data;
  uint  first;
  uint  last;

  uint max_size;
};

bool string_compare(const char* a, string b) { return b.count && strncmp(a, b.data, b.count) == 0; }
bool string_compare(string a, const char* b) { return string_compare(b, a); }

void add_to_queue(Queue* queue, uint object) {
  assert(queue->last  <  queue->max_size);
  assert(queue->first <= queue->last);

  queue->data[queue->last] = object;
  queue->last++;
}

uint get_from_queue(Queue* queue) {
  assert(queue->last  <= queue->max_size);
  assert(queue->first <  queue->last);

  uint to_return = queue->data[queue->first];
  queue->first++;
  return to_return;
}

bool is_queue_empty(const Queue* queue) {
  assert(queue->last  <= queue->max_size);
  assert(queue->first <= queue->last);
  return queue->first == queue->last;
}

void add_connection_to_graph_node(Graph* graph, uint target_node, uint connected_node) {
  assert(graph->connected_nodes_count[target_node]+1 < UINT8_MAX);

  graph->connected_nodes[target_node][graph->connected_nodes_count[target_node]] = connected_node;
  graph->connected_nodes_count[target_node] += 1;
  graph->graph_data++;
}

Grid_Position get_circle_position_on_a_grid(const Grid2D* grid, Vec2 point) {
  float x = point.x - left_boundary;
  float y = point.y - lower_boundary;

  uint i = x / grid->cell_size;
  uint j = y / grid->cell_size;

  return { i, j };
}

uint get_circle_id_on_a_grid(const Grid2D* grid, Vec2 point) {
  Grid_Position pos = get_circle_position_on_a_grid(grid, point);

  return pos.i * grid->number_of_cells_per_dimension + pos.j;
}

void get_all_neighbours_on_a_grid(const Grid2D* grid, Grid_Position n, Grid_Cell* data, uint* count) {
  // 
  // @Incomplete: since we are using sqrt(2)*radius as a cell size, circles don't fit completely into a cell, so there are cases when we have to take more neighbours from each side, depending on how circle is placed in a particular cell
  // Naive   approach: take 2 layers of neighbours. Total 32 possible neighbours per search.
  // Another approach: divide a cell into 4 quadrants and figure out where our current circle is placed. Total 15 possible neighbours per search
  // 

  assert(data);
  assert(count);
  *count = 0;

  size_t width = grid->number_of_cells_per_dimension;

  uint i = n.i;
  uint j = n.j;
  Grid_Position neighbours[NUMBER_OF_NEIGHBOURS] = { {i+2, j-2}, {i+2, j-1}, {i+2, j}, {i+2, j+1}, {i+2, j+2},
                                                     {i+1, j-2}, {i+1, j-1}, {i+1, j}, {i+1, j+1}, {i+1, j+2},
                                                     {i,   j-2}, {i,   j-1},           {i,   j+1}, {i,   j+2},
                                                     {i-1, j-2}, {i-1, j-1}, {i-1, j}, {i-1, j+1}, {i-1, j+2},
                                                     {i-2, j-2}, {i-2, j-1}, {i-2, j}, {i-2, j+1}, {i-2, j+2} };



  for (size_t i = 0; i < array_size(neighbours); i++) {
    Grid_Position n = neighbours[i];

    if (n.i >= width || n.j >= width) { continue; }

    size_t index = n.i*width + n.j;
    assert(index < grid->number_of_cells);

    Grid_Cell cell_id = grid->data[index];

    if (cell_id != CELL_IS_NOT_OCCUPIED) {
      data[*count] = cell_id;
      *count += 1;
    }
  }
}

float distance_squared(Vec2 a, Vec2 b) {
  float x  = (a.x - b.x);
  float y  = (a.y - b.y);
  float x2 = square(x);
  float y2 = square(y);
  return x2 + y2;
}

bool check_for_connection(Vec2 a, Vec2 b, float radius, float L) {
  return distance_squared(a, b) < square(L + 2*radius);
}

bool check_circles_are_inside_a_box(dynamic_array<Vec2>* array, float radius) {
  for (size_t i = 0; i < array->size; i++) {
    Vec2 v = (*array)[i];

    float x = v.x;
    float y = v.y;

    assert(-1.0f <= x-radius && x+radius <= 1.0f);
    assert(-1.0f <= y-radius && y+radius <= 1.0f);
  }
  return true;
}

bool check_circles_do_not_intersect_each_other(dynamic_array<Vec2>* array, float radius) {
  const float min_distance_between_nodes = square(radius + radius);

  for (size_t i = 0; i < array->size; i++) {
    for (size_t j = 0; j < array->size; j++) {
      if (i == j) continue;

      double     dist1 = distance_squared((*array)[i], (*array)[j]);
      double min_dist1 = min_distance_between_nodes;

      double     dist2 = sqrt(dist1);
      double min_dist2 = radius + radius;

      assert(dist1 >= min_dist1);
      assert(dist2 >= min_dist2);
    }
  }
  return true;
}

bool check_hash_table_is_filled_up(bool* hash_table, size_t N) {
  for (size_t i = 0; i < N; i++) {
    assert(hash_table[i]);
  }
  return true;
}


Vec2 generate_random_vec2(float radius) {
  auto r1 = rand();
  auto r2 = rand();

  const float c = 2.0f * (1.0f-radius) / (double)RAND_MAX; // @Hardcoded: boundaries are from [-1.0f, 1.0f]
 
  float x = c * r1 - (1.0f-radius);
  float y = c * r2 - (1.0f-radius);

  assert(-1.0f <= x-radius && x+radius <= 1.0f);
  assert(-1.0f <= y-radius && y+radius <= 1.0f);

  return Vec2 { x, y };
}

Vec2 generate_random_vec2_around_a_point(Vec2 point, float radius, float min_radius, float max_radius) {
  size_t angle, dist;
  double phi, ro;
  float  x, y;

regenerate: 
  angle = rand();
  dist  = rand();

  phi = TAU * angle / (float)(RAND_MAX+1);
  ro  = (max_radius - min_radius) * dist / (float)(RAND_MAX+1) + min_radius;

  assert(0          <= phi && phi <= TAU);
  assert(min_radius <= ro  && ro  <= max_radius);

  x = point.x + ro*cos(phi);
  y = point.y + ro*sin(phi);

  // @Hack: 
  if (!(-1.0f <= x-radius && x+radius <= 1.0f) || 
      !(-1.0f <= y-radius && y+radius <= 1.0f)) {
    goto regenerate;
  }

  assert(-1.0f <= x-radius && x+radius <= 1.0f);
  assert(-1.0f <= y-radius && y+radius <= 1.0f);
  
  return Vec2 { x, y };
}

void naive_random_sampling(dynamic_array<Vec2>* array, float radius) {
  const int       MAX_ALLOWED_ITERATIONS = 1e3;
  const float min_distance_between_nodes = (2 * radius) * (2 * radius);

  while (1) {
    Vec2 possible_point;
    bool allow_new_point;

    size_t iterations = 0;

   regenerate: 
    iterations++;

    if (iterations > MAX_ALLOWED_ITERATIONS) break;

    
    possible_point  = generate_random_vec2(radius);
    allow_new_point = true;
    for (size_t j = 0; j < array->size; j++) {
      Vec2 already_a_point = (*array)[j];

      float distance = distance_squared(possible_point, already_a_point);
      if (distance < min_distance_between_nodes) {
        allow_new_point = false;
        break;
      }
    }

    if (allow_new_point) {
      array_add(array, possible_point);
      continue;
    } else {
      goto regenerate;
    }
  }
}

void poisson_disk_sampling(dynamic_array<Vec2>* array, size_t N, float radius) {
  const float min_radius = 2 * radius;
  const float min_distance_between_nodes = min_radius * min_radius;

  dynamic_array<Vec2> active_points;
  defer { array_free(&active_points); };

  array_reserve(&active_points, N);
  array_add    (&active_points, generate_random_vec2(radius));

  size_t cursor = 0;
  const size_t NUMBER_OF_SAMPLES_TO_USE = 30;

  while (active_points.size) {
    size_t index = rand() % active_points.size;
    Vec2 active_point = active_points[index];

    bool found_something = false;
    for (size_t i = 0; i < NUMBER_OF_SAMPLES_TO_USE; i++) {
      Vec2 possible_point = generate_random_vec2_around_a_point(active_point, radius, min_radius, min_radius);

      bool allow_new_point = true;
      for (size_t j = 0; j < array->size; j++) {
        Vec2 already_a_point = (*array)[j];

        float distance = distance_squared(possible_point, already_a_point);
        if (distance < min_distance_between_nodes) {
          allow_new_point = false;
          break;
        }
      }

      if (allow_new_point) {
        cursor++;
        array_add(array,          possible_point);
        array_add(&active_points, possible_point);
        found_something = true;
        break;
      }
    }

    if (!found_something) {
      array_remove(&active_points, index);
    }
  }
}

void tightest_packing_sampling(dynamic_array<Vec2>* array, float* max_random_walking_distance, float target_radius, float packing_factor) {
  assert(array->size == 0);

  // target_radius^2 -- packing_factor.
  // radius       ^2 -- MAX_POSSIBLE_PACKING_FACTOR.
  float radius = sqrt(square(target_radius) * MAX_POSSIBLE_PACKING_FACTOR / packing_factor);
  float height = sqrt(3) * radius;

  *max_random_walking_distance = sqrt(square(radius) + square(height));

  // centers of a circle.
  float x;
  float y;

  bool is_even = true;

  y = lower_boundary + radius;
  while (y < upper_boundary - radius) {

    x  = left_boundary;
    x += (is_even) ? radius : 2*radius;

    while (x < right_boundary - radius) {
      array_add(array, Vec2{x, y});
      x += 2*radius;
    }

    y += height;
    is_even = !is_even;
  }
}

void create_square_grid(Grid2D* grid, dynamic_array<Vec2>* positions) {
  for (size_t k = 0; k < positions->size; k++) {
    Vec2* point = &(*positions)[k];

    uint n = get_circle_id_on_a_grid(grid, *point);

    assert(grid->data[n] == CELL_IS_NOT_OCCUPIED);
    grid->data[n] = k;
  }
}

void process_random_walk(Grid2D* grid, dynamic_array<Vec2>* positions, float radius, float max_random_walking_distance) {
  uint count = 0;
  Grid_Cell neighbours[NUMBER_OF_NEIGHBOURS];

  for (size_t i = 0; i < positions->size; i++) {
    Vec2* point = &(*positions)[i];

    float     distance  = max_random_walking_distance;
    float     direction = TAU * rand() / (float)(RAND_MAX+1);    
    Vec2 unit_direction = { cos(direction), sin(direction) };

    assert(0 <= direction && direction < TAU);

    Vec2 jump_to;

    int iteraction_counter = 0;

  jump:
    iteraction_counter++;
    if (iteraction_counter > 6) { // @Incomplete: think about it later.
      continue;
    }

    jump_to = *point + distance*unit_direction;

    jump_to.x = clamp(jump_to.x,  left_boundary+radius, right_boundary-radius);
    jump_to.y = clamp(jump_to.y, lower_boundary+radius, upper_boundary-radius);


    uint p = get_circle_id_on_a_grid(grid, *point);
    uint n = get_circle_id_on_a_grid(grid, jump_to);

    Grid_Cell* previous_id = &grid->data[p];
    Grid_Cell* next_id     = &grid->data[n];
    bool cell_is_not_occupied = *next_id == CELL_IS_NOT_OCCUPIED;
    
    if (cell_is_not_occupied) {
      *point = jump_to;

      if (*previous_id != *next_id) {
        assert(*next_id == CELL_IS_NOT_OCCUPIED);
        *next_id     = *previous_id;
        *previous_id = CELL_IS_NOT_OCCUPIED;
      }
    } else {
      distance /= 2.0f; // @Incomplete: we should probably do rand() again, not just ... /= 2.0f.
      goto jump;
    }
  }
}


void naive_collect_points_to_graph(Graph* graph, const dynamic_array<Vec2>* array, float radius, float L) {
  const size_t N = graph->count;

  assert(array->size == N);

  for (size_t i = 0; i < N; i++) {
    graph->connected_nodes[i] = graph->graph_data;

    for (size_t j = 0; j < N; j++) {
      if (i == j) continue; // nodes are the same => should be no connection with itself.

      if (check_for_connection((*array)[i], (*array)[j], radius, L)) {
        add_connection_to_graph_node(graph, i, j);
      }
    }
  }
}

void collect_points_to_graph_via_grid(Graph* graph, const Grid2D* grid, const dynamic_array<Vec2>* array, float radius, float L) {
  assert(array->size == graph->count);
  assert(L < 2*radius); // otherwise number of neighbours should be higher than 32.

  uint count = 0;
  Grid_Cell neighbours[NUMBER_OF_NEIGHBOURS];

  size_t width = grid->number_of_cells_per_dimension;

  for (size_t k = 0; k < array->size; k++) {
    graph->connected_nodes[k] = graph->graph_data;

    const Vec2* point = &(*array)[k];
    Grid_Position n = get_circle_position_on_a_grid(grid, *point);

    get_all_neighbours_on_a_grid(grid, n, neighbours, &count);


#if 0
    size_t num   = ceil((2*radius + L) / grid->cell_size) + 2;
    size_t len   = 2*num + 1;
    size_t total = square(len) - 1;

    uint count_test = 0;
    Grid_Cell* neighbours_test = (Grid_Cell*) alloca(sizeof(Grid_Cell) * total);

    for (size_t i = 0; i < len; i++) {
      for (size_t j = 0; j < len; j++) {

        size_t pi = n.i + i - num;
        size_t pj = n.j + j - num;

        if (pi == n.i && pj == n.j)     { continue; }
        if (pi >= width || pj >= width) { continue; }

        size_t index = pi*width + pj;
        assert(index < grid->number_of_cells);

        Grid_Cell cell_id = grid->data[index];

        if (cell_id != CELL_IS_NOT_OCCUPIED) {
          neighbours_test[count_test] = cell_id;
          count_test += 1;
        }
      }
    }
    assert(count == count_test);
#endif


    for (uint c = 0; c < count; c++) {
      Grid_Cell cell_id = neighbours[c];

      assert(cell_id != CELL_IS_NOT_OCCUPIED);
      Vec2 neighbour = (*array)[cell_id];

      if (check_for_connection(*point, neighbour, radius, L)) {
        add_connection_to_graph_node(graph, k, cell_id);
      }
    }
  }
}

void breadth_first_search(const Graph* graph, Queue* queue, bool* hash_table, size_t starting_index, dynamic_array<uint>* cluster_sizes) {
  // add i-th node.
  hash_table[starting_index] = true;
  add_to_queue(queue, starting_index);

  uint* size = array_add(cluster_sizes, 1);

  while (!is_queue_empty(queue)) {
    uint node_id = get_from_queue(queue);
    
    const uint*   nodes_to_search = graph->connected_nodes[node_id];
    const uint16  nodes_count     = graph->connected_nodes_count[node_id];

    for (uint i = 0; i < nodes_count; i++) {
      uint new_node_id = nodes_to_search[i];

      if (hash_table[new_node_id]) {
        // the node is already in a queue, we will process it anyway.
      } else {
        hash_table[new_node_id] = true;
        add_to_queue(queue, new_node_id);
        *size += 1;
      }
    }
  }
}

























string read_entire_file(const char* filename) {
  File file;
  bool success = file_open(&file, filename);
  if (!success) { return {}; }

  defer { file_close(&file); };

  size_t size = file_get_size(&file);
  char*  data = (char*) malloc(size+1);
  memset(data, 0, size+1);

  size_t written = 0;
  success = file_read(&file, data, size, &written);
  if (!success)        { return {}; } // @LogError: @MemoryLeak: @DeferFileClose: 
  if (written != size) { return {}; } // @LogError: @MemoryLeak: @DeferFileClose: 

  return { data, size };
}

Vertex_And_Fragment_Shader_Sources load_shaders(const char* filename) {
  string s = read_entire_file(filename); // @MemoryLeak: 
  if (!s.count) return {};               // @MemoryLeak: 

  static const uint TAG_NOT_FOUND  = (uint) -1;
  static const string vertex_tag   = make_string("#vertex");
  static const string fragment_tag = make_string("#fragment");
  static const string tags[]       = { vertex_tag, fragment_tag, make_string("") };

  string shaders[2] = {};

  uint    index  = TAG_NOT_FOUND;
  char*   cursor = s.data;
  string* current_shader = NULL;

  while (*cursor != '\0') {
    index = TAG_NOT_FOUND;
    index = string_compare(cursor, tags[0]) ? 0 : index;
    index = string_compare(cursor, tags[1]) ? 1 : index;

    if (index == TAG_NOT_FOUND) {
      if (current_shader) current_shader->count++;
      cursor++;
    } else {
      cursor += tags[index].count;
      current_shader = &shaders[index];
      current_shader->data  = cursor;
      current_shader->count = 0;
    }
  }

  assert(shaders[0].data && shaders[0].count &&   "vertex shader was not found in a file! it should be specified with a '#vertex' tag");
  assert(shaders[1].data && shaders[1].count && "fragment shader was not found in a file! it should be specified with a '#fragment' tag");

  return { shaders[0], shaders[1] };
}

unsigned create_shader(string vertex, string fragment) {
  unsigned program = glCreateProgram();

  unsigned vs = glCreateShader(GL_VERTEX_SHADER);
  unsigned fs = glCreateShader(GL_FRAGMENT_SHADER);

  int length_vs =   vertex.count;
  int length_fs = fragment.count;

  glShaderSource(vs, 1, &vertex.data,   &length_vs);
  glShaderSource(fs, 1, &fragment.data, &length_fs);

  glCompileShader(vs);
  glCompileShader(fs);

  int status_vs = 0;
  int status_fs = 0;

  glGetShaderiv(vs, GL_COMPILE_STATUS, &status_vs);
  glGetShaderiv(fs, GL_COMPILE_STATUS, &status_fs);

  if (!status_vs || !status_fs) { goto error; }

  glAttachShader(program, vs);
  glAttachShader(program, fs);

  glLinkProgram(program);
  glValidateProgram(program);

  glDetachShader(program, vs);
  glDetachShader(program, fs);

  glDeleteShader(vs);
  glDeleteShader(fs);

  return program;

error:
  int   message_size_vs;
  int   message_size_fs;
  char* message_vs;
  char* message_fs;

  if (!status_vs) glGetShaderiv(vs, GL_INFO_LOG_LENGTH, &message_size_vs);
  if (!status_fs) glGetShaderiv(fs, GL_INFO_LOG_LENGTH, &message_size_fs);

  if (!status_vs) message_vs = (char*) alloca(message_size_vs);
  if (!status_fs) message_fs = (char*) alloca(message_size_fs);

  if (!status_vs) glGetShaderInfoLog(vs, message_size_vs, &message_size_vs, message_vs);
  if (!status_fs) glGetShaderInfoLog(fs, message_size_fs, &message_size_fs, message_fs);

  if (!status_vs) glDeleteShader(vs);
  if (!status_fs) glDeleteShader(fs);

  if (!status_vs) printf("[..] Failed to compile vertex shader!\n%s",   message_vs);
  if (!status_fs) printf("[..] Failed to compile fragment shader!\n%s", message_fs);
  
  return 0;
}

void gl_debug_callback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam) {
  printf("[..] %s: %.*s\n", (type == GL_DEBUG_TYPE_ERROR ? "GL ERROR" : "" ), (int)length, message);

  __debugbreak(); // @MSVC_Only: 
}


struct Create_Vertex_Buffer {
  const float* data;
  size_t       size;
};

template<class T>
struct Create_Index_Buffer {
  const T* data;
  size_t   size;
};

struct Vertex_Array {
  uint vao = 0;
  uint vbo = 0;
  uint ibo = 0;

  // @Incomplete:
  // union {
  //  struct ... 
        uint vbo_attribute_index_in_enabled_array = 0;
        uint vbo_number_of_vertices_to_draw       = 0;
  //  };
  //  struct ... 
        GLenum ibo_type_of_indices           = 0;
        uint   ibo_number_of_indices_to_draw = 0;
        const void* ibo_indices_array        = NULL;
  //  };
  // };
};

struct Vertex_Attribute {
  uint16 attribute_index = 0;
  uint16 number_of_values_in_an_attribute = 0;                         // should be 1, 2, 3 or 4.
  uint16 size_of_one_vertex_in_bytes = 0;
  uint16 size_to_an_attribute_from_beginning_of_a_vertex_in_bytes = 0; // should be 0 if there is no attribute.

  GLenum attribute_type = GL_FLOAT;
  bool is_normalized = false;                                          // should be false if a value is normalized.
};

struct Shader {
  uint program = 0;
  void (*setup_uniform)(void*) = NULL;
  void* data = NULL;
};

struct Basic_Shader_Data {
  int uniform_color = 0;
  int uniform_mvp   = 0;

  float color[4];
  glm::mat4 mvp;
};

void add_vertex_attribute(Vertex_Attribute va) {
  glEnableVertexAttribArray(va.attribute_index);
  glVertexAttribPointer(va.attribute_index, 
                        va.number_of_values_in_an_attribute,
                        va.attribute_type,
                        va.is_normalized,
                        va.size_of_one_vertex_in_bytes,
                        (const void*) va.size_to_an_attribute_from_beginning_of_a_vertex_in_bytes);
}

template<class T> GLenum get_index_type()   { return 0; }
template<> GLenum get_index_type<uint8 >()  { return GL_UNSIGNED_BYTE;  }
template<> GLenum get_index_type<uint16>()  { return GL_UNSIGNED_SHORT; }
template<> GLenum get_index_type<uint32>()  { return GL_UNSIGNED_INT;   }




uint create_vertex_array() {
  uint buffer;
  glGenVertexArrays(1, &buffer);
  glBindVertexArray(buffer);
  return buffer;
}

uint create_vertex_buffer(Create_Vertex_Buffer vbo) {
  uint buffer;
  glGenBuffers(1, &buffer);
  glBindBuffer(GL_ARRAY_BUFFER, buffer);
  glBufferData(GL_ARRAY_BUFFER, sizeof(*vbo.data) * vbo.size, vbo.data, GL_STATIC_DRAW);
  return buffer;
}

template<class T>
uint create_index_buffer(Create_Index_Buffer<T> ibo) {
  uint buffer;
  glGenBuffers(1, &buffer);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffer);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(*ibo.data) * ibo.size, ibo.data, GL_STATIC_DRAW);
  return buffer;
}

template<class T>
Vertex_Array create_vertex_array(Create_Vertex_Buffer vbo, Create_Index_Buffer<T> ibo) {
  uint vao_id = create_vertex_array();
  uint vbo_id = create_vertex_buffer(vbo);
  uint ibo_id = create_index_buffer(ibo);

  Vertex_Array va;
  va.vao = vao_id;
  va.vbo = vbo_id;
  va.ibo = ibo_id;

  //va.vbo_attribute_index_in_enabled_array = 0;
  va.vbo_number_of_vertices_to_draw = vbo.size;

  va.ibo_type_of_indices = get_index_type<T>();
  va.ibo_number_of_indices_to_draw = ibo.size;
  //va.ibo_indices_array = NULL;
  return va;
}

Vertex_Array create_vertex_array(Create_Vertex_Buffer vbo) {
  uint vao_id = create_vertex_array();
  uint vbo_id = create_vertex_buffer(vbo);

  Vertex_Array va;
  va.vao = vao_id;
  va.vbo = vbo_id;

  //va.vbo_attribute_index_in_enabled_array = 0;
  va.vbo_number_of_vertices_to_draw = vbo.size;
  return va;
}

void bind_vertex_array(uint vao)  { glBindVertexArray(vao); }
void bind_vertex_buffer(uint vbo) { glBindBuffer(GL_ARRAY_BUFFER, vbo); }
void bind_index_buffer(uint ibo)  { glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo); }

void bind_shader(Shader shader) {
  glUseProgram(shader.program);
  shader.setup_uniform(shader.data);
}

void draw_call(GLenum draw_mode, Vertex_Array va) {
  bind_vertex_array(va.vao);
  if (va.ibo) {
    glDrawElements(draw_mode,
                   va.ibo_number_of_indices_to_draw,
                   va.ibo_type_of_indices,
                   va.ibo_indices_array);

  } else if (va.vbo) {
    glDrawArrays(draw_mode,
                 va.vbo_attribute_index_in_enabled_array,
                 va.vbo_number_of_vertices_to_draw);
  } else {
    assert(0);
  }
  // @Log: Successfully issued a draw call!
}

void basic_shader_uniform(void* data) {
  Basic_Shader_Data* s = (Basic_Shader_Data*) data;

  glUniform4f       (s->uniform_color, s->color[0], s->color[1], s->color[2], s->color[3]);
  glUniformMatrix4fv(s->uniform_mvp,   1, GL_FALSE, (float*) &s->mvp);
}



int main(int argc, char** argv) {
  init_filesystem_api();

  float radius = 0.05;
  float L      = radius + radius/10.0f;
  float packing_factor = 0.7;
  float max_random_walking_distance;


  uint seed = time(NULL);
  srand(seed);

  dynamic_array<Vec2> positions;
  array_reserve(&positions, 200000000);

  defer { array_free(&positions); };

  {
    printf("[..] Generating nodes ... \n");
    printf("[..] Finished sampling points in := ");

    measure_scope();
    //naive_random_sampling(&positions, radius);
    //poisson_disk_sampling(&positions, N, radius);
    tightest_packing_sampling(&positions, &max_random_walking_distance, radius, packing_factor);
  }


  const size_t N = positions.size;
  float one_dimension_range = 2.0f;
  float max_window_area  = 2.0f * 2.0f;
  float max_circles_area = positions.size * PI * square(radius);
  float experimental_packing_factor = max_circles_area / max_window_area;

  printf("[..]\n");
  printf("[..] Radius of a circle   := %g\n", radius);
  printf("[..] Connection radius(L) := %g\n", L);
  printf("{..] Packing factor       := %g\n", packing_factor);
  printf("[..] Generated packing factor := %g\n", experimental_packing_factor);
  printf("[..] Generated points     := %zu\n", positions.size);

  assert(max_circles_area < max_window_area);
  assert(N < UINT_MAX);                                              // because we are using uints to address graph nodes, N is required to be less than that.
  assert(packing_factor < MAX_POSSIBLE_PACKING_FACTOR);              // packing factor must be less than 0.9069...
  //assert(fabs(experimental_packing_factor - packing_factor) < 1e-2);



  Grid2D grid; 
  grid.cell_size                     = sqrt(2)*radius;
  grid.number_of_cells_per_dimension = one_dimension_range / grid.cell_size;
  grid.number_of_cells               = square(grid.number_of_cells_per_dimension);
  grid.data                          = (Grid_Cell*) malloc(sizeof(*grid.data) * grid.number_of_cells);

  defer { free(grid.data); };
  memset(grid.data, 0xFF, sizeof(*grid.data) * grid.number_of_cells);

  printf("[..]\n");
  printf("[..] Cell size       := %g\n", grid.cell_size);
  printf("[..] Number of cells := %zu\n", grid.number_of_cells);
  printf("[..] Number of cells per dimension := %zu\n", grid.number_of_cells_per_dimension);

  {
    // @Incomplete: combine tightest_packing with creating_grid, so that we don't loop over positions array twice!
    printf("[..]\n");
    printf("[..] Collecting nodes to a grid ... \n");
    printf("[..] Finished creating a grid in := ");
    measure_scope();
    create_square_grid(&grid, &positions);
  }

#if 1
  {
    printf("[..]\n");
    printf("[..] Random walk ... \n");
    printf("[..] Finished random walk in := ");
    measure_scope();
    process_random_walk(&grid, &positions, radius, max_random_walking_distance);
  }
  //assert(check_circles_are_inside_a_box(&positions, radius));
  //assert(check_circles_do_not_intersect_each_other(&positions, radius));
#endif


  const size_t a_lot = 1e10; // ~ 10 gigabytes.

  Graph graph;
  graph.count = N;
  graph.connected_nodes_count = (uint8*) malloc(a_lot);
  graph.connected_nodes       = (uint**) ((char*)graph.connected_nodes_count + sizeof(*graph.connected_nodes_count) * N);
  graph.graph_data            = (uint*)  ((char*)graph.connected_nodes       + sizeof(*graph.connected_nodes)       * N);

  defer { free(graph.connected_nodes_count); };
  memset(graph.connected_nodes_count, 0, sizeof(*graph.connected_nodes_count) * N);

  {
    printf("[..]\n");
    printf("[..] Successfully allocated 10 gigabytes of memory!\n");
    printf("[..] Collecting nodes to graph ... \n");
    printf("[..] Finished creating graph in := ");
    measure_scope();
    //naive_collect_points_to_graph(&graph, &positions, radius, L);
    collect_points_to_graph_via_grid(&graph, &grid, &positions, radius, L);
  }



  dynamic_array<uint> cluster_sizes;
  defer { array_free(&cluster_sizes); };

  bool* hash_table = (bool*) malloc(sizeof(bool) * N); // @Incomplete: instead of using 1 byte, we can use 1 bit => 8 times less memory for a hash_table.

  defer { free(hash_table); };
  memset(hash_table, false, sizeof(bool) * N);

  Queue queue;
  queue.data     = (uint*) malloc(sizeof(uint) * N);
  queue.first    = 0;
  queue.last     = 0;
  queue.max_size = N;

  defer { free(queue.data); };

  {
    printf("[..]\n");
    printf("[..] Starting BFS ... \n");
    printf("[..] Finished BFS in := ");
    measure_scope();

    for (size_t i = 0; i < graph.count; i++) {
      assert(is_queue_empty(&queue));
      if (!hash_table[i]) {
        breadth_first_search(&graph, &queue, hash_table, i, &cluster_sizes);
      }
    }
  }
  //assert(check_hash_table_is_filled_up(hash_table, N));

  // Find the biggest cluster.
  uint max_cluster = cluster_sizes[0];
  for (size_t i = 0; i < cluster_sizes.size; i++) {
    max_cluster = max(cluster_sizes[i], max_cluster);
  }
  assert(max_cluster);

  {
    printf("[..] Percolating cluster size := %u\n",  max_cluster);
    printf("[..] Number of clusters       := %zu\n", cluster_sizes.size);
    #if 0
    printf("[..] Cluster sizes := [");
    for (size_t i = 0; i < cluster_sizes.size; i++) {
      printf("%lu%s", cluster_sizes[i], (i == cluster_sizes.size-1) ? "]\n" : ", ");
    }
    #endif

    puts("");
    puts("");
    puts("");
  }
  // 















  if (!glfwInit()) {
    puts("[..] failed glfwInit()");
    return 1;
  }

  GLFWwindow* window = glfwCreateWindow(640, 640, "The World", NULL, NULL);
  if (!window) {
    puts("[..] failed glfwCreateWindow()");
    glfwTerminate();
    return 1;
  }

  glfwMakeContextCurrent(window); // make opengl context.
  glfwSwapInterval(1);            // synchronizes our frame rate with vsync (60 fps)

  if (glewInit() != GLEW_OK) {
    puts("[..] failed glewInit()");
    return -1;
  }

  printf("GL version := %s\n", glGetString(GL_VERSION));

  glEnable(GL_DEBUG_OUTPUT);
  glDebugMessageCallback(gl_debug_callback, NULL);







  Shader basic_shader;
  Basic_Shader_Data data;
  {
    Vertex_And_Fragment_Shader_Sources shaders = load_shaders("src/Basic.shader"); // @MemoryLeak: 
    basic_shader.program       = create_shader(shaders.vertex, shaders.fragment);
    basic_shader.setup_uniform = basic_shader_uniform;
    basic_shader.data          = &data;

    data.uniform_mvp   = glGetUniformLocation(basic_shader.program, "uniform_mvp");   // get address of uniform variable
    data.uniform_color = glGetUniformLocation(basic_shader.program, "uniform_color");
    assert(data.uniform_mvp   != -1);
    assert(data.uniform_color != -1);
  }

  Vertex_Array vert_array1;
  {
    const uint NUM_VERTICES = 21;

    dynamic_array<float>  vertex;
    dynamic_array<uint16> index;
    array_resize(&vertex, NUM_VERTICES * 2);
    array_resize(&index, NUM_VERTICES * 3);

    defer { array_free(&vertex); };
    defer { array_free(&index); };


    {
      vertex[0] = 0.0f;
      vertex[1] = 0.0f;

      float  a = 0;
      float da = TAU / (double)(NUM_VERTICES-2);
      for (size_t i = 2; i < vertex.size; i += 2) {
        vertex[i+0] = cos(a);
        vertex[i+1] = sin(a);
        a += da;
      }

      size_t j = 0;
      for (size_t i = 1; i < NUM_VERTICES; i++) {
        index[j  ] = 0;
        index[j+1] = i;
        index[j+2] = i+1;
        j += 3;
      }
    }

    Create_Vertex_Buffer vbo;
    Create_Index_Buffer<uint16> ibo;

    vbo.data = vertex.data;
    vbo.size = vertex.size;

    ibo.data = index.data;
    ibo.size = index.size;

    vert_array1 = create_vertex_array(vbo, ibo);

    Vertex_Attribute va;
    va.number_of_values_in_an_attribute = 2;
    va.size_of_one_vertex_in_bytes      = sizeof(*vertex.data) * 2;
    add_vertex_attribute(va);
  }


  Vertex_Array vert_array2;
  {

    const float lines[] = {
      -1.0f, -1.0f,
       1.0f,  1.0f,
    };

    Create_Vertex_Buffer buffer;
    buffer.data = lines;
    buffer.size = array_size(lines);

    vert_array2 = create_vertex_array(buffer);

    Vertex_Attribute va;
    va.number_of_values_in_an_attribute = 2;
    va.size_of_one_vertex_in_bytes      = sizeof(*buffer.data) * 2;
    add_vertex_attribute(va);
  }


  Vertex_Array vert_array3;
  {
    const float quad[] = {
      -0.5f, -0.5f,
      -0.5f,  0.5f,
       0.5f, -0.5f,
       0.5f,  0.5f,
    };

    const uint8 indices[] = {
      0, 1,
      0, 2,
      1, 3,
      2, 3,
    };

    Create_Vertex_Buffer vertex;
    vertex.data = quad;
    vertex.size = array_size(quad);

    Create_Index_Buffer<uint8> index;
    index.data = indices;
    index.size = array_size(indices);

    vert_array3 = create_vertex_array(vertex, index);

    Vertex_Attribute va;
    va.number_of_values_in_an_attribute = 2;
    va.size_of_one_vertex_in_bytes      = sizeof(*vertex.data) * 2;
    add_vertex_attribute(va);
  }


  float r = 0.0f;
  float increment = 0.05f;

  while (!glfwWindowShouldClose(window)) {
    glClear(GL_COLOR_BUFFER_BIT);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glUseProgram(0);


    Basic_Shader_Data* data = (Basic_Shader_Data*) basic_shader.data;
    data->color[0] = r;
    data->color[1] = 0.3f;
    data->color[2] = 1.0f;
    data->color[3] = 0.87f;
    data->mvp      = glm::mat4(1);

#if 0
    bind_shader(basic_shader);
    draw_call(GL_LINES, vert_array2);
#endif

#if 1
    // 
    // @Incomplete: batch rendering...
    // @Incomplete: @CleanUp: we can draw circles better, just draw a quad and in the fragment shader fill up fragments that are sqrt(x*x + y*y) < radius.
    // 
    for (size_t i = 0; i < positions.size; i++) {
      float x = positions[i].x;
      float y = positions[i].y;

      glm::mat4 model = glm::mat4(1);
      model = glm::translate(model, glm::vec3(x, y, 0));
      model = glm::scale(model, glm::vec3(radius, radius, 0));

      Basic_Shader_Data* data = (Basic_Shader_Data*) basic_shader.data;
      data->mvp = model;

      bind_shader(basic_shader);
      draw_call(GL_LINES, vert_array1);
    }
#endif

#if 1
    for (size_t i = 0; i < grid.number_of_cells_per_dimension; i++) {
      for (size_t j = 0; j < grid.number_of_cells_per_dimension; j++) {
        float x = (i + 1/2.0f) * grid.cell_size - 1.0f;
        float y = (j + 1/2.0f) * grid.cell_size - 1.0f;

        glm::mat4 model = glm::mat4(1);
        model = glm::translate(model, glm::vec3(x, y, 0));
        model = glm::scale(model, glm::vec3(grid.cell_size, grid.cell_size, 0));

        Basic_Shader_Data* data = (Basic_Shader_Data*) basic_shader.data;
        data->color[0] = 1.0f;
        data->color[1] = 0.0f;
        data->color[2] = 0.0f;
        data->color[3] = 0.87f;
        data->mvp = model;

        bind_shader(basic_shader);
        draw_call(GL_LINES, vert_array3);
      }
    }
#endif


    if (r > 1.0f) {
      increment = -0.05f;
    } else if (r < 0.0f) {
      increment = 0.05f;
    }

    r += increment;


    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  glfwTerminate();
  return 0;
}

