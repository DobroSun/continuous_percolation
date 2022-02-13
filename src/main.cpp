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

#include <memory> // @Incomplete: <- This is C++ std file. I can't figure out a C include file for alloca!

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


const double PI  = 3.14159265358979323846;
const double TAU = 6.28318530717958647692;

struct string {
  char*  data;
  size_t count;
};

#define make_string(x) { x, sizeof(x)-1 }

static bool string_compare(const char* a, string b) { return b.count && strncmp(a, b.data, b.count) == 0; }
static bool string_compare(string a, const char* b) { return string_compare(b, a); }


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

struct Vertex_And_Fragment_Shaders {
  string vertex;
  string fragment;
};

Vertex_And_Fragment_Shaders load_shaders(const char* filename) {
  string s = read_entire_file(filename); // @MemoryLeak: 
  if (!s.count) return {};               // @MemoryLeak: 

  static const string vertex_tag   = make_string("#vertex");
  static const string fragment_tag = make_string("#fragment");
  static const string tags[]       = { vertex_tag, fragment_tag, make_string("") };

  string shaders[2];

  size_t index  = 0;
  char*  cursor = s.data;

  string* current_shader = NULL;

  while (*cursor != '\0') {
    if (string_compare(cursor, tags[index])) {
      cursor += tags[index].count;

      current_shader = &shaders[index];
      current_shader->data  = cursor;
      current_shader->count = 0;
      index++;

    } else {
      if (current_shader) current_shader->count++;
      cursor++;
    }
  }

  assert(index == 2 && "shader file should contain #vertex and #fragment tags in order!");
  assert(shaders[0].count != 0 && shaders[1].count != 0 && "vertex and fragment shaders should not be empty!");

  return { shaders[0], shaders[1] };
}

unsigned compile_shader(string source, unsigned type) {
  unsigned int id = glCreateShader(type);

  int len = source.count;
  glShaderSource(id, 1, &source.data, &len);
  glCompileShader(id);

  int status;
  glGetShaderiv(id, GL_COMPILE_STATUS, &status);
  if (!status) {
    int length;
    char* message;

    glGetShaderiv(id, GL_INFO_LOG_LENGTH, &length);

    message = (char*) alloca(length);

    glGetShaderInfoLog(id, length, &length, message);
    glDeleteShader(id);

    const char* shader_name = type == GL_VERTEX_SHADER ? "vertex" : "fragment";
    printf("[..] Failed to compile %s shader!\n", shader_name);
    puts(message);
    return 0;
  }
  
  return id;
}

unsigned create_shader(string vertex, string fragment) {
  unsigned program = glCreateProgram();

  unsigned vs = compile_shader(vertex,   GL_VERTEX_SHADER);
  unsigned fs = compile_shader(fragment, GL_FRAGMENT_SHADER);

  if (!vs) return 0;
  if (!fs) return 0;

  glAttachShader(program, vs);
  glAttachShader(program, fs);
  glLinkProgram(program);
  glValidateProgram(program);

  glDetachShader(program, vs);
  glDetachShader(program, fs);

  glDeleteShader(vs);
  glDeleteShader(fs);

  return program;
}

void gl_debug_callback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam) {
  printf("[..] %s: %.*s\n", (type == GL_DEBUG_TYPE_ERROR ? "GL ERROR" : "" ), (int)length, message);

  __debugbreak(); // @MSVC_Only: 
}


struct Vec2 {
  float x, y;
};

struct Graph {
  uint    count;

  uint16* connected_nodes_count;
  uint**  connected_nodes; // @Incomplete: if we ever need we can make this use less space, instead of using pointers we can go with just uints.

  // for graph creation.
  uint*   graph_data;
};

struct Queue {
  uint* data;
  uint  first;
  uint  last;

  uint max_size;
};

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

float distance_squared(Vec2 a, Vec2 b) {
  float x  = (a.x - b.x);
  float y  = (a.y - b.y);
  float x2 = x * x;
  float y2 = y * y;
  return x2 + y2;
}

bool check_nodes_do_not_intersect_each_other(dynamic_array<Vec2>* array, float radius) {
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

bool check_hash_table_is_correct(bool* hash_table, size_t N) {
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

  phi = TAU * angle / (double)(RAND_MAX+1);
  ro  = (max_radius - min_radius) * dist / (double)(RAND_MAX+1) + min_radius;

  assert(0          <= phi && phi <= TAU);
  assert(min_radius <= ro  && ro  <= max_radius);

  x = point.x + ro*cos(phi);
  y = point.y + ro*sin(phi);

  // @Incomplete: assert on [-1.0f, 1.0f] boundaries.
  // @Hack: 
  if (!(-1.0f <= x-radius && x+radius <= 1.0f) || 
      !(-1.0f <= y-radius && y+radius <= 1.0f)) {
    goto regenerate;
  }

  assert(-1.0f <= x-radius && x+radius <= 1.0f);
  assert(-1.0f <= y-radius && y+radius <= 1.0f);
  
  return Vec2 { x, y };
}

void naive_random_sampling(dynamic_array<Vec2>* array, size_t N, float radius) {
  const float min_distance_between_nodes = (2 * radius) * (2 * radius);

  for (size_t i = 0; i < N; i++) {
    Vec2 possible_point;
    bool allow_new_point;

   regenerate: 

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

void tightest_packing_sampling(dynamic_array<Vec2>* array, float radius) {
  assert(array->size == 0);

  float left_boundary  = -1.0f;
  float right_boundary =  1.0f;
  float lower_boundary = -1.0f;
  float upper_boundary =  1.0f;

  float height = sqrt(3) * radius + 1e-6; // @RemoveMe: 

  // centers of a circle.
  float x;
  float y;

  size_t layer = 0;

  y = lower_boundary + radius;
  while (y < upper_boundary - radius) {

    bool is_even = layer % 2 == 0;
    x  = left_boundary;
    x += (is_even) ? radius : 2*radius;

    while (x < right_boundary - radius) {
      array_add(array, Vec2{x, y});
      x += 2*radius+ 1e-6; // @RemoveMe: 
    }

    y += height;
    layer++;
  }
}

bool check_for_connection(Vec2 a, Vec2 b, float radius, float L) {
  return sqrt(distance_squared(a, b)) - 2*radius < L;
}

void naive_collect_points_to_graph(Graph* graph, dynamic_array<Vec2>* array, float radius, float L) {
  const size_t N = graph->count;

  assert(array->size == N);

  for (size_t i = 0; i < N; i++) {
    graph->connected_nodes[i] = graph->graph_data;

    for (size_t j = 0; j < N; j++) {
      if (i == j) continue; // nodes are the same => should be no connection with itself.

      if (check_for_connection((*array)[i], (*array)[j], radius, L)) {
        graph->connected_nodes[i][graph->connected_nodes_count[i]] = j;
        graph->connected_nodes_count[i] += 1;
        graph->graph_data++;
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

/*
 TODO:
  1) Algorithm to generate random points in specified area.
  2) Collecting array of points to a graph.
  3) Visualization (2D, 3D?)

 + Blue noise generation algorithm ?
  ??? 

 + Acceleration structure ?
  Octree
  When all nodes are generated, we can create octree to be more efficient while creating a coupled graph.
  But I have no idea how to use it properly.

 + What data structure to use for a graph ?
  Let's mark all nodes with numbers (0, 1, 2, .., N) nodes.
  And use an array of arrays:
  [0] : [all edges connected to 0th node]
  [1] : [all edges connected to 1st node]
  [2] : [all edges connected to 2nd node]
  ... 
  [N] : [all edges connected to Nth node]

 Since there is only a constant number of surrounding nodes, we should use static arrays for all of them.
 Or should we allocate them dynamically? What is going to be more efficient / take less memory?
*/

/*
  (N * 8)data                                       bytes of generated points (2D)
  (N * 2)count + (N * 8)pointers + (N * 5 * 4)data  bytes of graph.                  // (5 is taken as an average number of connections per node)

  38*N bytes for a program

  if N == 10^7 => total 3.0 gb of data.
*/

int main(int argc, char** argv) {
  init_filesystem_api();

  const float radius = 0.0797885; // sqrt(proportion * max_window_area / (float)N / PI); // @Incomplete: this should be input to an algorithm.
  const float L = radius + radius/10.0f;

  dynamic_array<Vec2> circles_array;
  defer { array_free(&circles_array); };

  { // do_stuff() 
    uint seed = time(NULL);
    srand(seed);


    printf("[..] Generating nodes ... \n");

    dynamic_array<Vec2> positions;
    array_reserve(&positions, 1000);

    //naive_random_sampling(positions, N, radius);
    //poisson_disk_sampling(&positions, N, radius);
    tightest_packing_sampling(&positions, radius);
    assert(check_nodes_do_not_intersect_each_other(&positions, radius));

    const size_t N = positions.size;

    const double max_window_area  = 2.0f * 2.0f;
    const double max_circles_area = positions.size * PI * radius * radius;
    const double packing_factor   = max_circles_area / max_window_area;

    printf("[..] Radius of a circle   := %g\n", radius);
    printf("[..] Connection radius(L) := %g\n", L);
    printf("[..] Circles area         := %g\n", max_circles_area);
    printf("[..] Window  area         := %g\n", max_window_area);
    printf("[..] Generated            := %lu points!\n", positions.size);
    printf("[..] Resulting packing factor := %g\n", packing_factor);

    assert(max_circles_area < max_window_area);
    assert(N < UINT_MAX);                         // because we are using uints to address graph nodes, N is required to be less than that.
    assert(packing_factor <= PI * sqrt(3) / 6.0); // packing factor must be less than 0.9069...

    printf("[..]\n");

    const size_t a_lot = 1e10; // ~ 10 gigabytes.
    char* memory = (char*) malloc(a_lot);
    defer { free(memory); };

    printf("[..] Successfully allocated 10 gigabytes of memory!\n");
    printf("[..] Collecting nodes to a graph ... \n");

    Graph graph;
    graph.count = N;
    graph.connected_nodes_count = (uint16*) memory;
    graph.connected_nodes       = (uint**) ((char*)graph.connected_nodes_count + sizeof(uint16) * N);
    graph.graph_data            = (uint*)  ((char*)graph.connected_nodes       + sizeof(uint*)  * N);

    memset(graph.connected_nodes_count, 0, sizeof(uint16) * N);
    memset(graph.connected_nodes,       0, sizeof(uint*)  * N);

    naive_collect_points_to_graph(&graph, &positions, radius, L);

    printf("[..]\n");
    printf("[..] Starting BFS ... \n");

    dynamic_array<uint> cluster_sizes;
    defer { array_free(&cluster_sizes); };

    bool* hash_table = (bool*) alloca(sizeof(bool) * N); // @Incomplete: instead of using 1 byte, we can use 1 bit => 8 times less memory for a hash_table.
    memset(hash_table, 0, sizeof(bool) * N);

    Queue queue;
    queue.data     = (uint*) alloca(sizeof(uint) * N);
    queue.first    = 0;
    queue.last     = 0;
    queue.max_size = N;

    for (size_t i = 0; i < graph.count; i++) {
      assert(is_queue_empty(&queue));
      if (!hash_table[i]) {
        breadth_first_search(&graph, &queue, hash_table, i, &cluster_sizes);
      }
    }

    assert(check_hash_table_is_correct(hash_table, N));

    uint max_cluster = cluster_sizes[0];
    for (size_t i = 0; i < cluster_sizes.size; i++) {
      uint size = cluster_sizes[i];
      max_cluster = (size > max_cluster) ? size : max_cluster;
    }
    assert(max_cluster);

    printf("[..] Number of clusters := %lu\n", cluster_sizes.size);
    printf("[..] Cluster sizes := [");
    for (size_t i = 0; i < cluster_sizes.size; i++) {
      printf("%lu%s", cluster_sizes[i], (i == cluster_sizes.size-1) ? "]\n" : ", ");
    }

    printf("[..] Percolating cluster size := %lu\n", max_cluster);
    puts("");
    puts("");
    puts("");

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
  glm::mat4 model      = glm::mat4(1);

  float r = 0.0f;
  float increment = 0.05f;

  while (!glfwWindowShouldClose(window)) {
    glClear(GL_COLOR_BUFFER_BIT);

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

