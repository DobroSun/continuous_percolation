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

struct Vertex_And_Fragment_Shaders {
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
  uint index;
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

  uint index = i*grid->number_of_cells_per_dimension + j;
  assert(index < grid->number_of_cells);

  return {i, j, index};
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
    neighbours[i].index = n.i * width + n.j;
  }

  for (size_t i = 0; i < array_size(neighbours); i++) {
    Grid_Position n = neighbours[i];

    if (n.i >= width || n.j >= width) { continue; }

    size_t index = n.i * width + n.j;
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

bool check_for_collision(Vec2 a, Vec2 b, float radius) {
  return distance_squared(a, b) < square(2*radius);
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
  const float min_distance_between_nodes = (2 * radius) * (2 * radius);

  for (size_t i = 0; i < array->size; i++) {
    for (size_t j = 0; j < array->size; j++) {
      if (i == j) continue;

      double     dist1 = distance_squared((*array)[i], (*array)[j]);
      double min_dist1 = min_distance_between_nodes;

      double     dist2 = sqrt(dist1);
      double min_dist2 = radius + radius;

      #if 0
      printf("(i, j) := (%zu, %zu)\n", i, j);
      printf("(given, expected) := (%.17g, %.17g)\n", dist1, min_dist1);
      printf("(given, expected) := (%.17g, %.17g)\n", dist2, min_dist2);
      printf("[..]\n");
      #endif

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

    Grid_Position n = get_circle_position_on_a_grid(grid, *point);

    assert(grid->data[n.index] == CELL_IS_NOT_OCCUPIED);
    grid->data[n.index] = k;
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


    Grid_Position p = get_circle_position_on_a_grid(grid, *point);
    Grid_Position n = get_circle_position_on_a_grid(grid, jump_to);

    Grid_Cell* previous_id = &grid->data[p.index];
    Grid_Cell* next_id     = &grid->data[n.index];
    bool cell_is_occupied = *next_id != CELL_IS_NOT_OCCUPIED;
    
    if (!cell_is_occupied) {
      *point = jump_to;

      if (*previous_id != *next_id) {
        assert(*next_id == CELL_IS_NOT_OCCUPIED);
        *next_id     = *previous_id;
        *previous_id = CELL_IS_NOT_OCCUPIED;
      }
    } else {
      distance /= 2.0f;
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

  
  for (size_t k = 0; k < array->size; k++) {
    graph->connected_nodes[k] = graph->graph_data;

    const Vec2* point = &(*array)[k];
    Grid_Position   n = get_circle_position_on_a_grid(grid, *point);

    assert(grid->data[n.index] == k);

    get_all_neighbours_on_a_grid(grid, n, neighbours, &count);

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

Vertex_And_Fragment_Shaders load_shaders(const char* filename) {
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

int main(int argc, char** argv) {
  init_filesystem_api();

  float radius = 0.01;
  float L      = radius + radius/10.0f;
  float packing_factor = 0.7;
  float max_random_walking_distance;

#if 1
  dynamic_array<Vec2> circles_array;
  defer { array_free(&circles_array); };
#endif

  {
    uint seed = time(NULL);
    srand(seed);


    dynamic_array<Vec2> positions;
    array_reserve(&positions, 200000000);

    {
      printf("[..] Generating nodes ... \n");
      printf("[..] Finished sampling points in := ");

      measure_scope();
      //naive_random_sampling(&positions, radius);
      //poisson_disk_sampling(&positions, N, radius);
      tightest_packing_sampling(&positions, &max_random_walking_distance, radius, packing_factor);
    }
    //assert(check_circles_are_inside_a_box(&positions, radius));
    //assert(check_circles_do_not_intersect_each_other(&positions, radius));


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
    assert(fabs(experimental_packing_factor - packing_factor) < 1e-2);



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


    {
      printf("[..]\n");
      printf("[..] Random walk ... \n");
      printf("[..] Finished random walk in := ");
      measure_scope();
      process_random_walk(&grid, &positions, radius, max_random_walking_distance);
    }


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

#if 0
    array_free(&positions);
#endif


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

    circles_array = positions;
  }















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


  // circle vertices & indices.
  const uint NUM_VERTICES = 21;

  Vec2 positions[NUM_VERTICES];
  uint16 indices[NUM_VERTICES * 3];

  { // generating vertices & indices for a circle of unit radius.
    positions[0] = { 0.0f, 0.0f };
    
    float  a = 0;
    float da = TAU / (double)(NUM_VERTICES-2);
    for (size_t i = 1; i < NUM_VERTICES; i++) {
      positions[i] = { cos(a), sin(a) };
      a += da;
    }

    size_t j = 0;
    for (size_t i = 1; i < NUM_VERTICES; i++) {
      indices[j  ] = 0;
      indices[j+1] = i;
      indices[j+2] = i+1;
      j += 3;
    }
  }

  uint vao;
  glGenVertexArrays(1, &vao);
  glBindVertexArray(vao);

  uint buffer;
  glGenBuffers(1, &buffer);
  glBindBuffer(GL_ARRAY_BUFFER, buffer);
  glBufferData(GL_ARRAY_BUFFER, sizeof(positions), positions, GL_STATIC_DRAW);

  glEnableVertexAttribArray(0);
  // index is the index of an attribute we want to enable.

  glVertexAttribPointer(0, 2, GL_FLOAT, false, sizeof(float)*2, 0);
  // index := attribute index (it is the first attribute so 0, next one is going to be 1, 2, ...).
  // size  := number of values in an attribute.
  // type  := type of the thing in buffer.
  // normalizd := normalized flag (should be true if data in buffer is not normalized, i.e. not (0..1))
  // stride  := size of one vertex (in bytes)
  // pointer := size to next attribute from the beginning of a vertex (in bytes)

  uint ibo;
  glGenBuffers(1, &ibo);
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);


  Vertex_And_Fragment_Shaders shaders = load_shaders("src/Basic.shader");

  uint shader = create_shader(shaders.vertex, shaders.fragment);
  glUseProgram(shader);

  int uniform_color = glGetUniformLocation(shader, "uniform_color"); // get address of uniform variable
  int uniform_mvp   = glGetUniformLocation(shader, "uniform_mvp");

  assert(uniform_color != -1);
  assert(uniform_mvp   != -1);

  glm::mat4 projection = glm::ortho(-1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f);

  float r = 0.0f;
  float increment = 0.05f;

  while (!glfwWindowShouldClose(window)) {
    glClear(GL_COLOR_BUFFER_BIT);

#if 1
    // 
    // @Incomplete: batch rendering...
    // @Incomplete: @CleanUp: we can draw circles better, just draw a quad and in the fragment shader fill up fragments that are sqrt(x*x + y*y) < radius.
    // 
    for (size_t i = 0; i < circles_array.size; i++) {
      float x = circles_array[i].x;
      float y = circles_array[i].y;

      glm::mat4 model = glm::mat4(1);
      model = glm::translate(model, glm::vec3(x, y, 0));
      model = glm::scale(model, glm::vec3(radius, radius, 0));

      glm::mat4 mvp = projection * model;

      glUniform4f       (uniform_color, r, 0.3f, 0.8f, 1.0f);
      glUniformMatrix4fv(uniform_mvp,   1, GL_FALSE, (float*) &mvp);
      glDrawElements(GL_LINES, sizeof(indices)/sizeof(*indices), GL_UNSIGNED_SHORT, NULL);
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

