#define GLEW_STATIC


#include <random>
#include "../std/pch.h"


const uint CELL_IS_EMPTY = 0xFFFFFFFF;
const size_t MEMORY_ALLOCATION_SIZE = 1e10; // ~ 10 gigabytes.

const double PI  = 3.14159265358979323846;
const double TAU = 6.28318530717958647692;
const double MAX_POSSIBLE_PACKING_FACTOR = PI * sqrt(3) / 6.0;

const float  left_boundary = -1.0f;
const float right_boundary =  1.0f;
const float lower_boundary = -1.0f;
const float upper_boundary =  1.0f;


const float lines_vertices_data[] = {
  -1.0f, -1.0f,
   1.0f,  1.0f,
};

const float quads_vertices_data[] = {
  -0.5f, -0.5f,
  -0.5f,  0.5f,
   0.5f, -0.5f,
   0.5f,  0.5f,
};

const uint quads_indices_data[] = {
  0, 1,
  0, 2,
  1, 3,
  2, 3,
};

const int NUMBER_OF_VERTICES_FOR_A_CIRCLE = 40;
float   circles_vertices_data[NUMBER_OF_VERTICES_FOR_A_CIRCLE] = {};
uint32  circles_indices_data [NUMBER_OF_VERTICES_FOR_A_CIRCLE] = {};

// @Incomplete: maybe just make precompiled circles_vertices_data & circles_indices_data?
void init_circles_vertices_and_indices_data() {
  float * vertex = circles_vertices_data;
  uint32* index  = circles_indices_data;

  float  a = 0;
  float da = TAU / (float)(NUMBER_OF_VERTICES_FOR_A_CIRCLE/2 - 1); // @Incomplete: add some checks to know this is correct.
  for (int i = 0; i < static_array_size(circles_vertices_data); i += 2) {
    vertex[i + 0] = cos(a);
    vertex[i + 1] = sin(a);
    a += da;
  }

  int j = 0;
  for (int i = 0; i < static_array_size(circles_indices_data); i += 2) {
    index[i+0] = j;
    index[i+1] = j+1;
    j++;
  }
}

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

struct Vec4 {
  float x, y, z, w;
};

struct Line {
  Vec2 origin;
  Vec2  direction;
};

struct Circle {
  Vec2 origin;
  float radius;
};

struct Matrix4x4 {
  float m11, m12, m13, m14, 
        m21, m22, m23, m24,
        m31, m32, m33, m34,
        m41, m42, m43, m44;
};

Matrix4x4 box_to_box_matrix(Vec4 from_min, Vec4 from_max, Vec4 to_min, Vec4 to_max) {
  // works on 2D plane.
  // min -> (-1.0f, -1.0f)
  // max -> ( 1.0f,  1.0f)

  float x1 = from_min.x;
  float x2 = from_max.x;

  float y1 = from_min.y;
  float y2 = from_max.y;

  float e1 = to_min.x;
  float e2 = to_max.x;

  float f1 = to_min.y;
  float f2 = to_max.y;

  Matrix4x4 m;
  m.m11 = (e2 - e1) / (x2 - x1); m.m12 = 0;                     m.m13 = 0; m.m14 = -x1 * (e2 - e1) / (x2 - x1) + e1;
  m.m21 = 0;                     m.m22 = (f2 - f1) / (y2 - y1); m.m23 = 0; m.m24 = -y1 * (f2 - f1) / (y2 - y1) + f1;
  m.m31 = 0;                     m.m32 = 0;                     m.m33 = 0; m.m34 = 0; // don't use z
  m.m41 = 0;                     m.m42 = 0;                     m.m43 = 0; m.m44 = 1;

  return m;
}

Matrix4x4 identity_matrix() {
  Matrix4x4 m;
  m.m11 = 1; m.m12 = 0; m.m13 = 0; m.m14 = 0;
  m.m21 = 0; m.m22 = 1; m.m23 = 0; m.m24 = 0;
  m.m31 = 0; m.m32 = 0; m.m33 = 1; m.m34 = 0;
  m.m41 = 0; m.m42 = 0; m.m43 = 0; m.m44 = 1;
  return m;
}

Matrix4x4 scale_matrix(Vec4 vec) {
  Matrix4x4 m;
  m.m11 = vec.x; m.m12 = 0;     m.m13 = 0;     m.m14 = 0;
  m.m21 = 0;     m.m22 = vec.y; m.m23 = 0;     m.m24 = 0;
  m.m31 = 0;     m.m32 = 0;     m.m33 = vec.z; m.m34 = 0;
  m.m41 = 0;     m.m42 = 0;     m.m43 = 0;     m.m44 = vec.w;
  return m;
}

Matrix4x4 translation_matrix(Vec4 vec) {
  Matrix4x4 m;
  m.m11 = 1; m.m12 = 0; m.m13 = 0; m.m14 = vec.x;
  m.m21 = 0; m.m22 = 1; m.m23 = 0; m.m24 = vec.y;
  m.m31 = 0; m.m32 = 0; m.m33 = 1; m.m34 = vec.z;
  m.m41 = 0; m.m42 = 0; m.m43 = 0; m.m44 = vec.w;
  return m;
}

Matrix4x4 transform_screen_space_to_normalized_space(float width, float height) {
  const Vec4 from_min = { 0.0f, height, 0.0f, 1.0f };
  const Vec4 from_max = { width, 0.0f,  0.0f, 1.0f };
  const Vec4 to_min   = { -1.0f, -1.0f, 0.0f, 0.0f };
  const Vec4 to_max   = {  1.0f,  1.0f, 0.0f, 0.0f };
  return box_to_box_matrix(from_min, from_max, to_min, to_max);
}

Vec4 operator*(Matrix4x4 m, Vec4 v) { return { m.m11 * v.x + m.m12 * v.y + m.m13 * v.z + m.m14 * v.w, 
                                               m.m21 * v.x + m.m22 * v.y + m.m23 * v.z + m.m24 * v.w, 
                                               m.m31 * v.x + m.m32 * v.y + m.m33 * v.z + m.m34 * v.w,
                                               m.m41 * v.x + m.m42 * v.y + m.m43 * v.z + m.m44 * v.w }; }


Matrix4x4 operator*(Matrix4x4 m, Matrix4x4 n) {
  Matrix4x4 r;
  r.m11 = m.m11*n.m11 + m.m12*n.m21 + m.m13*n.m31 + m.m14*n.m41; r.m12 = m.m11*n.m12 + m.m12*n.m22 + m.m13*n.m32 + m.m14*n.m42; r.m13 = m.m11*n.m13 + m.m12*n.m23 + m.m13*n.m33 + m.m14*n.m43; r.m14 = m.m11*n.m14 + m.m12*n.m24 + m.m13*n.m34 + m.m14*n.m44;
  r.m21 = m.m21*n.m11 + m.m22*n.m21 + m.m23*n.m31 + m.m24*n.m41; r.m22 = m.m21*n.m12 + m.m22*n.m22 + m.m23*n.m32 + m.m24*n.m42; r.m23 = m.m21*n.m13 + m.m22*n.m23 + m.m23*n.m33 + m.m24*n.m43; r.m24 = m.m21*n.m14 + m.m22*n.m24 + m.m23*n.m34 + m.m24*n.m44;
  r.m31 = m.m31*n.m11 + m.m32*n.m21 + m.m33*n.m31 + m.m34*n.m41; r.m32 = m.m31*n.m12 + m.m32*n.m22 + m.m33*n.m32 + m.m34*n.m42; r.m33 = m.m31*n.m13 + m.m32*n.m23 + m.m33*n.m33 + m.m34*n.m43; r.m34 = m.m31*n.m14 + m.m32*n.m24 + m.m33*n.m34 + m.m34*n.m44;
  r.m41 = m.m41*n.m11 + m.m42*n.m21 + m.m43*n.m31 + m.m44*n.m41; r.m42 = m.m41*n.m12 + m.m42*n.m22 + m.m43*n.m32 + m.m44*n.m42; r.m43 = m.m41*n.m13 + m.m42*n.m23 + m.m43*n.m33 + m.m44*n.m43; r.m44 = m.m41*n.m14 + m.m42*n.m24 + m.m43*n.m34 + m.m44*n.m44;
  return r;
}




Vec2 operator+(Vec2 a, Vec2 b)  { return { a.x + b.x, a.y + b.y }; }
Vec2 operator-(Vec2 a, Vec2 b)  { return { a.x - b.x, a.y - b.y }; }
Vec2 operator*(Vec2 a, float c) { return { c * a.x, c * a.y }; }
Vec2 operator/(Vec2 a, float c) { return { a.x / c, a.y / c }; }
Vec2 operator*(float c, Vec2 a) { return a * c; }

double dot(Vec2 a, Vec2 b) {
  return a.x * b.x + a.y * b.y;
}

float length_sq(Vec2 a) {
  return dot(a, a);
}

bool line_and_circle_intersection(Line a, Circle b, double* distance) {
  Vec2 dirr = a.direction;
  Vec2 diff = a.origin - b.origin;

  float distance_between_circle_origin_and_line = sqrt(length_sq(diff) - length_sq(dot(diff, dirr)/length_sq(dirr) * dirr));
  if (distance) *distance = distance_between_circle_origin_and_line;
  return distance_between_circle_origin_and_line < b.radius;
}


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

struct Cluster_Data {
  bool touches_left  = false;
  bool touches_right = false;
  bool touches_up    = false;
  bool touches_down  = false;
  size_t cluster_size = 0;
};

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

  union {
    struct { // vbo != NULL, ibo == NULL;
      uint   vbo_attribute_index_in_enabled_array;
      uint   vbo_number_of_vertices_to_draw;
      GLenum vbo_primitive_to_render;
    };
    struct { // vbo != NULL, ibo != NULL;
      GLenum ibo_type_of_indices;
      uint   ibo_number_of_elements_to_draw;
      void*  ibo_indices_array;               // we may want to keep index buffer on a client. if this is NULL draw call is going to use index buffer from GPU
    };
  };
};

// @Incomplete: could actually make this take 8 bytes instead of 16. sizeof(GLenum) == 4, but we only use: GL_FLOAT, GL_DOUBLE, ... so we can make a uint8 variable to keep the type.
struct Vertex_Attribute {
  uint16 attribute_index = 0;
  uint16 number_of_values_in_an_attribute = 0;                         // should be 1, 2, 3 or 4.
  uint16 size_of_one_vertex_in_bytes = 0;
  uint16 size_to_an_attribute_from_beginning_of_a_vertex_in_bytes = 0; // should be 0 if there is only one attribute.

  GLenum attribute_type = GL_FLOAT;
  bool is_normalized = false;                                          // should be false if a value is normalized.
};

struct Shader {
  uint program = 0;
  void (*setup_uniform)(void*) = NULL;
  void* data = NULL;
};

struct Basic_Shader_Data {
  int uniform_mvp = 0;
  float* mvp; // 16 floats.
};

struct Thread_Data {
  Memory_Arena* arena;

  float particle_radius;
  float jumping_conductivity_distance;
  float packing_factor;
};

static Vertex_Array circle = {};
static Vertex_Array line   = {};
static Vertex_Array quad   = {};
static Vertex_Array batched_quads   = {};
static Vertex_Array batched_circles = {};
static Shader            basic_shader = {};
static Basic_Shader_Data basic_shader_data = {};

static Thread computation_thread = {};
static Thread_Data thread_data   = {};
static Memory_Arena arena        = {};

static array<Vec2> global_positions = {};
static Grid2D              global_grid      = {};
static Graph               global_graph     = {};

static int64  result_largest_cluster_size = 0;


double generate_random_double_in_range(double from, double to) {
  static std::random_device rd;
  static std::mt19937 e(rd());

  std::uniform_real_distribution<double> dist(from, to);
  return dist(e);
}

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

void check_circle_touches_boundaries(Vec2 pos, float radius, uint cluster_size, Cluster_Data* result) {
  float diameter = 2*radius;

  bool touches_left  = pos.x <  left_boundary + diameter;
  bool touches_right = pos.x > right_boundary - diameter;
  bool touches_up    = pos.y < lower_boundary + diameter;
  bool touches_down  = pos.y > upper_boundary - diameter;

  bool* left  = &result->touches_left;
  bool* right = &result->touches_right;
  bool* up    = &result->touches_up;
  bool* down  = &result->touches_down;

  *left  = *left  ? *left  : touches_left;
  *right = *right ? *right : touches_right;
  *up    = *up    ? *up    : touches_up;
  *down  = *down  ? *down  : touches_down;

  if (*up && *down || *left && *right) {
    result->cluster_size = cluster_size;
  }
}

bool check_circles_are_inside_a_box(array<Vec2>* array, float radius) {
  for (size_t i = 0; i < array->size; i++) {
    Vec2 v = (*array)[i];

    float x = v.x;
    float y = v.y;

    assert(left_boundary  <= x-radius && x+radius <= right_boundary);
    assert(lower_boundary <= y-radius && y+radius <= upper_boundary);
  }
  return true;
}

bool check_circles_do_not_intersect_each_other(array<Vec2>* array, float radius) {
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

void naive_random_sampling(array<Vec2>* array, float radius) {
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

void poisson_disk_sampling(array<Vec2>* out, size_t N, float radius) {
  const float min_radius = 2 * radius;
  const float min_distance_between_nodes = min_radius * min_radius;

  array<Vec2> active_points;
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
      for (size_t j = 0; j < out->size; j++) {
        Vec2 already_a_point = (*out)[j];

        float distance = distance_squared(possible_point, already_a_point);
        if (distance < min_distance_between_nodes) {
          allow_new_point = false;
          break;
        }
      }

      if (allow_new_point) {
        cursor++;
        array_add(out,            possible_point);
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

void tightest_packing_sampling(array<Vec2>* array, float* max_random_walking_distance, float target_radius, float packing_factor) {
  assert(array->size == 0);

  // target_radius^2 -- packing_factor.
  // radius       ^2 -- MAX_POSSIBLE_PACKING_FACTOR.
  float radius = sqrt(square(target_radius) * MAX_POSSIBLE_PACKING_FACTOR / packing_factor);
  float height = sqrt(3) * radius;

  *max_random_walking_distance = radius - target_radius;

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

void create_square_grid(Grid2D* grid, array<Vec2>* positions) {
  for (size_t k = 0; k < positions->size; k++) {
    Vec2* point = &(*positions)[k];

    uint n = get_circle_id_on_a_grid(grid, *point);

    assert(grid->data[n] == CELL_IS_EMPTY);
    grid->data[n] = k;
  }
}


void take_all_neighbours_with_distance(Grid2D* grid, array<Grid_Cell>* neighbours, Vec2 point, float radius, float distance) {

  Vec2 new_point = point + Vec2{ distance, distance };

  Grid_Position p = get_circle_position_on_a_grid(grid,     point);
  Grid_Position n = get_circle_position_on_a_grid(grid, new_point);

  size_t width = grid->number_of_cells_per_dimension;
  size_t di = abs((int64)n.i - (int64)p.i);
  size_t dj = abs((int64)n.j - (int64)p.j);

  size_t delta = max(di, dj) + 1;


  // 
  // now let's just iterate over all of the: possible neighbours. and find those that contain circles.
  //  

  size_t pis = p.i - delta;
  size_t pie = p.i + delta;

  size_t pjs = p.j - delta;
  size_t pje = p.j + delta;

  for (size_t i = pis; i < pie; i++) {
    for (size_t j = pjs; j < pje; j++) {

      if (i >= width || j >= width) { continue; }
      if (i == p.i && j == p.j)     { continue; }

      size_t index = i*width + j;
      assert(index < grid->number_of_cells);

      Grid_Cell cell_id = grid->data[index];
      if (cell_id != CELL_IS_EMPTY) {
        array_add(neighbours, cell_id);
      }
    }
  }
}

void process_random_walk(Grid2D* grid, array<Vec2>* positions, float radius, float max_random_walking_distance) {
  for (size_t n = 0; n < 6; n++) {
    for (size_t i = 0; i < positions->size; i++) {

      array<Grid_Cell> neighbours;
      array<Grid_Cell> intersection;
      array<double>    intersection_distance;
      array<double>    distances;

      defer { array_free(&neighbours); };
      defer { array_free(&intersection); };
      defer { array_free(&intersection_distance); };
      defer { array_free(&distances); };

      Vec2* point = &(*positions)[i];

      // find all neighbours.
      take_all_neighbours_with_distance(grid, &neighbours, *point, radius, max_random_walking_distance);

      // 
      // now we should find the one that is in our way.
      // in order to do that, we should know the direction to move in.
      //

      double    direction = generate_random_double_in_range(0, TAU);
      Vec2 unit_direction = { cos(direction), sin(direction) };

      Line a;
      a.origin    = *point;
      a.direction = unit_direction;

      // find all circles that are intersecting unit_direction.
      for (size_t k = 0; k < neighbours.size; k++) {
        Grid_Cell cell = neighbours[k];
        Vec2*   origin = &(*positions)[cell];

        Circle b;
        b.origin = *origin;
        b.radius = radius; // because we want to put a circle in there with another radius, not just simple line & circle intersection check.

        double distance;

        if (dot(b.origin - a.origin, a.direction) > 0) { // in the same direction.
          if (line_and_circle_intersection(a, b, &distance)) {
            array_add(&intersection,          cell);
            array_add(&intersection_distance, distance);
          }
        }
      }

      // find all min distances that we can put circle to.
      for (size_t k = 0; k < intersection.size; k++) {
        Grid_Cell cell = intersection[k];
        Vec2*   origin = &(*positions)[cell];

        double p = intersection_distance[k];
        double l = 2*radius;

        double delta = sqrt(l*l - p*p);
        assert(0 <= delta && delta <= 2*radius);

        double  d = sqrt(length_sq(*origin - a.origin) - p*p);
        double rd = d - delta;
        array_add(&distances, rd);
      }

      // find min distance.
      double min_dist = max_random_walking_distance;
      for (size_t k = 0; k < distances.size; k++) {
        min_dist = min(min_dist, distances[k]);
      }

      double min_d = min(0, min_dist);
      double max_d = max(0, min_dist);
      double distance = generate_random_double_in_range(min_d, max_d);
      Vec2    jump_to = *point + distance*unit_direction;

      jump_to.x = clamp(jump_to.x,  left_boundary+radius, right_boundary-radius);
      jump_to.y = clamp(jump_to.y, lower_boundary+radius, upper_boundary-radius);

      uint p = get_circle_id_on_a_grid(grid, *point);
      uint n = get_circle_id_on_a_grid(grid, jump_to);

      Grid_Cell* previous_id = &grid->data[p];
      Grid_Cell* next_id     = &grid->data[n];
      bool cell_is_not_occupied = *next_id == CELL_IS_EMPTY;
      
      assert(*previous_id == *next_id ? true : cell_is_not_occupied); // if we did a jump, next_id should be empty.

      *point = jump_to;

      if (*previous_id != *next_id) {
        assert(*next_id == CELL_IS_EMPTY);
        *next_id     = *previous_id;
        *previous_id = CELL_IS_EMPTY;
      }
    }
  }
}


void naive_collect_points_to_graph(Graph* graph, const array<Vec2>* array, float radius, float L) {
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

void collect_points_to_graph_via_grid(Graph* graph, const Grid2D* grid, const array<Vec2>* array, float radius, float L) {
  assert(array->size == graph->count);

  size_t width = grid->number_of_cells_per_dimension;

  for (size_t k = 0; k < array->size; k++) {
    graph->connected_nodes[k] = graph->graph_data;

    const Vec2* point = &(*array)[k];
    Grid_Position n = get_circle_position_on_a_grid(grid, *point);

    size_t num   = floor(2.0f*(radius+L)/grid->cell_size);
    size_t len   = 2*num + 1;
    size_t total = square(len) - 1;

    uint count = 0;
    Grid_Cell* neighbours = (Grid_Cell*) alloca(sizeof(Grid_Cell) * total);

    for (size_t i = 0; i < len; i++) {
      for (size_t j = 0; j < len; j++) {

        size_t pi = n.i + i - num;
        size_t pj = n.j + j - num;

        if (pi == n.i && pj == n.j)     { continue; }
        if (pi >= width || pj >= width) { continue; }

        size_t index = pi*width + pj;
        assert(index < grid->number_of_cells);

        Grid_Cell cell_id = grid->data[index];

        if (cell_id != CELL_IS_EMPTY) {
          neighbours[count] = cell_id;
          count+= 1;
        }
      }
    }

    for (uint c = 0; c < count; c++) {
      Grid_Cell cell_id = neighbours[c];

      assert(cell_id != CELL_IS_EMPTY);
      Vec2 neighbour = (*array)[cell_id];

      if (check_for_connection(*point, neighbour, radius, L)) {
        add_connection_to_graph_node(graph, k, cell_id);
      }
    }
  }
}

void breadth_first_search(const Graph* graph, Queue* queue, const array<Vec2>* positions, bool* hash_table, size_t starting_index, float radius, array<uint>* cluster_sizes, Cluster_Data* result) {

  uint* size = array_add(cluster_sizes);

  // add i-th node.
  hash_table[starting_index] = true;
  add_to_queue(queue, starting_index);
  *size = 1;

  {
    Vec2 pos = (*positions)[starting_index];
    check_circle_touches_boundaries(pos, radius, *size, result);
  }

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

        Vec2 pos = (*positions)[new_node_id];
        check_circle_touches_boundaries(pos, radius, *size, result);
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
  static const string tags[]       = { vertex_tag, fragment_tag };

  bool vertex_found   = false;
  bool fragment_found = false;
  bool vertex_two_or_more_occurrences   = false;
  bool fragment_two_or_more_occurrences = false;

  string shaders[2] = {};

  uint    index  = TAG_NOT_FOUND;
  char*   cursor = s.data;
  string* current_shader = NULL;

  while (*cursor != '\0') {
    bool vertex   = string_compare(cursor, tags[0]);
    bool fragment = string_compare(cursor, tags[1]); 

    vertex_two_or_more_occurrences   = vertex_two_or_more_occurrences   || (vertex_found    && vertex);
    fragment_two_or_more_occurrences = fragment_two_or_more_occurrences || (fragment_found  && fragment);
    vertex_found                     = vertex_found   || vertex;
    fragment_found                   = fragment_found || fragment;
    
    index = TAG_NOT_FOUND;
    index = vertex   ? 0 : index;
    index = fragment ? 1 : index;

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

  assert(shaders[0].data && shaders[0].count && "vertex shader was not found in a source! it should be specified with a '#vertex' tag");
  assert(shaders[1].data && shaders[1].count && "fragment shader was not found in a source! it should be specified with a '#fragment' tag");
  assert(!vertex_two_or_more_occurrences     && "there are multiple #vertex shader tags in a source, but expected only one!");
  assert(!fragment_two_or_more_occurrences   && "there are multiple #fragment shader tags in a source, but expected only one!");

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

  glDetachShader(program, vs);
  glDetachShader(program, fs);

  glDeleteShader(vs);
  glDeleteShader(fs);

  return program;

error:
  int   message_size_vs = 0;
  int   message_size_fs = 0;
  char* message_vs = NULL;
  char* message_fs = NULL;

  if (!status_vs) glGetShaderiv(vs, GL_INFO_LOG_LENGTH, &message_size_vs);
  if (!status_fs) glGetShaderiv(fs, GL_INFO_LOG_LENGTH, &message_size_fs);

  message_vs = (char*) alloca(message_size_vs + 1);
  message_fs = (char*) alloca(message_size_fs + 1);

  memset(message_vs, 0, message_size_vs+1);
  memset(message_fs, 0, message_size_fs+1);

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


void add_vertex_attribute(Vertex_Attribute va) {
  glEnableVertexAttribArray(va.attribute_index);
  glVertexAttribPointer(va.attribute_index, 
                        va.number_of_values_in_an_attribute,
                        va.attribute_type,
                        va.is_normalized,
                        va.size_of_one_vertex_in_bytes,
                        (const void*) va.size_to_an_attribute_from_beginning_of_a_vertex_in_bytes);
}

template<class T> GLenum get_index_type()  { return assert(0 && "get_index_type expects only uint8, uint16, uint32 types!"); }
template<> GLenum get_index_type<uint8 >() { return GL_UNSIGNED_BYTE;  }
template<> GLenum get_index_type<uint16>() { return GL_UNSIGNED_SHORT; }
template<> GLenum get_index_type<uint32>() { return GL_UNSIGNED_INT;   }




uint create_vertex_array() {
  uint buffer;
  glGenVertexArrays(1, &buffer);
  glBindVertexArray(buffer);
  return buffer;
}

uint create_vertex_buffer(Create_Vertex_Buffer vbo, GLenum mode = GL_STATIC_DRAW) {
  uint buffer;
  glGenBuffers(1, &buffer);
  uint create_vertex_buffer_data(uint, Create_Vertex_Buffer, GLenum);
  return create_vertex_buffer_data(buffer, vbo, mode);
}

uint create_vertex_buffer_data(uint buffer, Create_Vertex_Buffer vbo, GLenum mode = GL_STATIC_DRAW) {
  glBindBuffer(GL_ARRAY_BUFFER, buffer);
  glBufferData(GL_ARRAY_BUFFER, sizeof(*vbo.data) * vbo.size, vbo.data, mode);
  return buffer;
}

uint update_vertex_buffer_data(uint buffer, Create_Vertex_Buffer vbo) {
  glBindBuffer(GL_ARRAY_BUFFER, buffer);
  glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(*vbo.data) * vbo.size, vbo.data);
  return buffer;
}

template<class T>
uint create_index_buffer(Create_Index_Buffer<T> ibo, GLenum mode = GL_STATIC_DRAW) {
  uint buffer;
  glGenBuffers(1, &buffer);
  return create_index_buffer_data(buffer, ibo, mode);
}

template<class T>
uint create_index_buffer_data(uint buffer, Create_Index_Buffer<T> ibo, GLenum mode = GL_STATIC_DRAW) {
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffer);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(*ibo.data) * ibo.size, ibo.data, mode);
  return buffer;
}

Vertex_Array create_vertex_array(Create_Vertex_Buffer vbo, GLenum primitive = 0, GLenum drawing_mode = GL_STATIC_DRAW) {
  assert(primitive && "rendering primitive was not provided when creating vertex buffer object!");

  int number_of_vertices_per_primitive;
  switch(primitive) {
  case GL_LINES:      number_of_vertices_per_primitive = 2; break;;
  case GL_TRIANGLES:  number_of_vertices_per_primitive = 3; break;;
  default: assert(0 && "unknown rendering primitive!");     break;
  }
  assert((vbo.size % number_of_vertices_per_primitive) == 0);

  uint vao_id = create_vertex_array();
  uint vbo_id = create_vertex_buffer(vbo, drawing_mode);
  uint ibo_id = 0;

  Vertex_Array va;
  va.vao = vao_id;
  va.vbo = vbo_id;
  va.ibo = ibo_id;

  va.vbo_attribute_index_in_enabled_array = 0;
  va.vbo_number_of_vertices_to_draw = vbo.size/number_of_vertices_per_primitive;
  va.vbo_primitive_to_render        = primitive;
  return va;
}

template<class T>
Vertex_Array create_vertex_array(Create_Vertex_Buffer vbo, Create_Index_Buffer<T> ibo, GLenum drawing_mode = GL_STATIC_DRAW) {
  uint vao_id = create_vertex_array();
  uint vbo_id = create_vertex_buffer(vbo, drawing_mode);
  uint ibo_id = create_index_buffer(ibo, drawing_mode);

  Vertex_Array va;
  va.vao = vao_id;
  va.vbo = vbo_id;
  va.ibo = ibo_id;

  va.ibo_type_of_indices = get_index_type<T>();
  va.ibo_number_of_elements_to_draw = ibo.size;
  va.ibo_indices_array = NULL;
  return va;
}

#if 0
template<class T>
Vertex_Array create_vertex_array(Create_Vertex_Buffer vbo, Create_Index_Buffer<T> ibo, Vertex_Attribute attrib, GLenum primitive = 0, GLenum mode = GL_STATIC_DRAW) {
  uint vao_id = create_vertex_array();
  uint vbo_id = create_vertex_buffer(vbo, mode);
  uint ibo_id = create_index_buffer(ibo, mode);

  Vertex_Array va = {};
  va.vao = vao_id;
  va.vbo = vbo_id;
  va.ibo = ibo_id;

  va.kind_of_primitive = primitive;
  return va;
}
#endif

#if 0
template<class T>
uint update_index_buffer_data(uint buffer, Create_Index_Buffer<T> ibo) {
  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, buffer);
  glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, sizeof(*ibo.data) * ibo.size, ibo.data);
  return buffer;
}

template<class T>
Vertex_Array update_vertex_array_data(Vertex_Array va, Create_Vertex_Buffer vbo) {
  assert(va.vao);
  assert(va.vbo);
  assert(va.ibo);

  udpate_vertex_buffer_data(va.vbo, vbo);
  udpate_index_buffer_data(va.ibo, ibo);

  va.vbo_attribute_index_in_enabled_array = 0;
  va.vbo_number_of_vertices_provided = vbo.size;
  return va;
}
#endif

#if 0
template<class T>
Vertex_Array update_vertex_array_data(Vertex_Array va, Create_Vertex_Buffer vbo, Create_Index_Buffer<T> ibo) {
  assert(va.vao);
  assert(va.vbo);
  assert(va.ibo);

  udpate_vertex_buffer_data(va.vbo, vbo);
  udpate_index_buffer_data(va.ibo, ibo);

  va.ibo_type_of_indices = get_index_type<T>();
  va.ibo_number_of_elements_provided = ibo.size;
  va.ibo_indices_array = NULL;
  return va;
}
#endif

void bind_vertex_array(uint vao)  { glBindVertexArray(vao); }
void bind_vertex_buffer(uint vbo) { glBindBuffer(GL_ARRAY_BUFFER, vbo); }
void bind_index_buffer(uint ibo)  { glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ibo); }

void bind_shader(Shader shader) {
  glUseProgram(shader.program);
  shader.setup_uniform(shader.data);
}

void draw_call(GLenum draw_mode, Vertex_Array va) {
  if (va.ibo) {
    assert(draw_mode);

    bind_vertex_array(va.vao);
    glDrawElements(draw_mode,
                   va.ibo_number_of_elements_to_draw,
                   va.ibo_type_of_indices,
                   va.ibo_indices_array);

  } else if (va.vbo) {
    assert(draw_mode == va.vbo_primitive_to_render);

    bind_vertex_array(va.vao);
    glDrawArrays(va.vbo_primitive_to_render,
                 va.vbo_attribute_index_in_enabled_array,
                 va.vbo_number_of_vertices_to_draw);
  } else {
  }
  // @Log: Successfully issued a draw call!
}

void basic_shader_uniform(void* data) {
  Basic_Shader_Data* s = (Basic_Shader_Data*) data;

  glUniformMatrix4fv(s->uniform_mvp, 1, GL_TRUE, (float*) s->mvp);
}


void do_the_thing(Memory_Arena* arena, float radius, float L, float packing_factor, size_t* largest_cluster_size) {
  float max_random_walking_distance;

  reset_memory_arena(arena);

  array<Vec2> positions = {};
  positions.allocator = arena->allocator;

  {
    printf("[..] Generating nodes ... \n");
    printf("[..] Finished sampling points in := ");

    measure_scope();
    //naive_random_sampling(&positions, radius);
    //poisson_disk_sampling(&positions, N, radius);
    tightest_packing_sampling(&positions, &max_random_walking_distance, radius, packing_factor);
  }


  const size_t N = positions.size;
  const float one_dimension_range = right_boundary - left_boundary;
  const float max_window_area  = square(one_dimension_range);
  const float max_circles_area = positions.size * PI * square(radius);
  const float experimental_packing_factor = max_circles_area / max_window_area;

  printf("[..]\n");
  printf("[..] Radius of a circle   := %g\n", radius);
  printf("[..] Connection radius(L) := %g\n", L);
  printf("{..] Packing factor       := %g\n", packing_factor);
  printf("[..] Generated packing factor := %g\n", experimental_packing_factor);
  printf("[..] Generated points     := %zu\n", positions.size);

  assert(max_circles_area < max_window_area);
  assert(N < UINT_MAX);                                              // because we are using uints to address graph nodes, N is required to be less than that.
  assert(packing_factor < MAX_POSSIBLE_PACKING_FACTOR);              // packing factor must be less than 0.9069...
  //assert(fabs(experimental_packing_factor - packing_factor) < 1e-1); // @Incomplete: 


  Grid2D grid; 
  grid.cell_size                     = sqrt(2)*radius;
  grid.number_of_cells_per_dimension = one_dimension_range / grid.cell_size;
  grid.number_of_cells               = square(grid.number_of_cells_per_dimension);
  grid.data                          = (Grid_Cell*) alloc(arena->allocator, sizeof(*grid.data) * grid.number_of_cells);

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
    printf("[..] Starting random walk ... \n");
    printf("[..] Finished random walk in := ");
    measure_scope();
    process_random_walk(&grid, &positions, radius, max_random_walking_distance);
  }
  //assert(check_circles_are_inside_a_box(&positions, radius));
  //assert(check_circles_do_not_intersect_each_other(&positions, radius));

  Graph graph;
  graph.count = N;
  graph.connected_nodes_count = (uint8*) alloc(arena->allocator, sizeof(uint8) * (arena->allocated - arena->top)); // allocate to max capacity.
  graph.connected_nodes       = (uint**) ((char*)graph.connected_nodes_count + sizeof(*graph.connected_nodes_count) * N);
  graph.graph_data            = (uint*)  ((char*)graph.connected_nodes       + sizeof(*graph.connected_nodes)       * N);

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


  array<uint> cluster_sizes;
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

  Cluster_Data percolation;

  {
    printf("[..]\n");
    printf("[..] Starting BFS ... \n");
    printf("[..] Finished BFS in := ");
    measure_scope();

    for (size_t i = 0; i < graph.count; i++) {
      if (!hash_table[i]) {
        Cluster_Data result;

        breadth_first_search(&graph, &queue, &positions, hash_table, i, radius, &cluster_sizes, &result);

        bool found_larger_cluster = result.cluster_size > percolation.cluster_size;
        percolation = found_larger_cluster ? result : percolation;
      }
    }
  }
  //assert(check_hash_table_is_filled_up(hash_table, N));

  uint max_size = cluster_sizes[0];
  for (size_t i = 0; i < cluster_sizes.size; i++) {
    max_size = cluster_sizes[i] > max_size ? cluster_sizes[i] : max_size;
  }

  *largest_cluster_size = max_size;

  global_positions = positions;
  global_grid      = grid;
  global_graph     = graph;

  {
    printf("[..]\n");
    printf("[..] l := %d, r := %d, u := %d, d := %d\n", percolation.touches_left, percolation.touches_right, percolation.touches_up, percolation.touches_down);
    printf("[..] Percolating cluster size := %zu\n",  percolation.cluster_size);
    printf("[..] Largest cluster size     := %u\n",  max_size);
    printf("[..] Number of clusters       := %zu\n", cluster_sizes.size);
    printf("[..] Cluster sizes := [");
    for (size_t i = 0; i < cluster_sizes.size; i++) {
      printf("%lu%s", cluster_sizes[i], (i == cluster_sizes.size-1) ? "]\n" : ", ");
    }
    puts("");
    puts("");
    puts("");
  }
}

static int computation_thread_proc(void* param) {
  Thread_Data data = *(Thread_Data*) param;

  size_t largest_cluster_size;
  do_the_thing(data.arena,
               data.particle_radius,
               data.jumping_conductivity_distance,
               data.packing_factor,
               &largest_cluster_size);

  InterlockedExchange64(&result_largest_cluster_size, (int64) largest_cluster_size);

  // 
  // 
  //
  return 0;
}

void abstract_the_thing(glm::vec2* positions, size_t positions_count, glm::vec2* source_vertex, size_t source_vertex_count, uint* source_index, size_t source_index_count, glm::vec2* out_vertex, size_t* out_vertex_count, uint* out_index, size_t* out_index_count) {

  for (size_t i = 0; i < positions_count; i++) {
    glm::vec2 world = positions[i];
    glm::mat4 transform = glm::translate(glm::mat4(1), glm::vec3(world.x, world.y, 0));

    for (size_t k = 0; k < source_vertex_count; k++) {
      out_vertex[k] = transform * glm::vec4(source_vertex[k], 0, 0);
      *out_vertex_count += sizeof(*out_vertex) / sizeof(float);
    }

    size_t idx = i * source_index_count;
    for (size_t k = 0; k < source_index_count; k++) {
      out_index[k] = idx + source_index[k];
      *out_index_count += 1;
    }
  }
}

void make_grid_positions(const Grid2D* grid, glm::vec2* grid_vertices) {
  size_t size = 0;
  for (size_t i = 0; i < grid->number_of_cells_per_dimension; i++) {
    for (size_t j = 0; j < grid->number_of_cells_per_dimension; j++) {
      grid_vertices[size] = { (float)i, (float)j };
      grid_vertices[size] /= grid->number_of_cells_per_dimension;
      size++;
    }
  }
}

void init_program(int width, int height) {
  init_filesystem_api();
  init_threads_api();

  check_filesystem_api();
  check_threads_api();

  {
    init_circles_vertices_and_indices_data();
  }

  { // @Incomplete: use std C++11 stuff.
    uint seed = time(NULL);
    srand(seed);
  }

  {
    // Initialize default input;
    Thread_Data* data = &thread_data;
    data->arena           = &arena;
    data->particle_radius = 0.1;
    data->jumping_conductivity_distance = 1.5 * data->particle_radius;
    data->packing_factor  = 0.5;

    begin_memory_arena(data->arena, MEMORY_ALLOCATION_SIZE);

    size_t junk;
    do_the_thing(data->arena,
                 data->particle_radius,
                 data->jumping_conductivity_distance,
                 data->packing_factor,
                 &junk);
  }

  { // loading shaders.
    Vertex_And_Fragment_Shader_Sources shaders = load_shaders("src/Basic.shader"); // @MemoryLeak: 
    basic_shader.program = create_shader(shaders.vertex, shaders.fragment);
    basic_shader.setup_uniform = basic_shader_uniform;
    basic_shader.data = &basic_shader_data;

    basic_shader_data.uniform_mvp = glGetUniformLocation(basic_shader.program, "uniform_mvp");   // get address of uniform variable
    assert(basic_shader_data.uniform_mvp != -1);
  }

#if 0
  { // line
    Create_Vertex_Buffer vbo;
    vbo.data = lines_vertices_data;
    vbo.size = static_array_size(lines_vertices_data);

    line = create_vertex_array(vbo, GL_LINES);

    Vertex_Attribute va;
    va.attribute_index = 0;
    va.number_of_values_in_an_attribute = 2;
    va.size_of_one_vertex_in_bytes = sizeof(float) * 2;
    add_vertex_attribute(va);
  }
  { // quad
    Create_Vertex_Buffer vbo;
    vbo.data = quads_vertices_data;
    vbo.size = static_array_size(quads_vertices_data);

    Create_Index_Buffer<uint32> ibo;
    ibo.data = quads_indices_data;
    ibo.size = static_array_size(quads_indices_data);

    quad = create_vertex_array(vbo, ibo);

    Vertex_Attribute va;
    va.attribute_index = 0;
    va.number_of_values_in_an_attribute = 2;
    va.size_of_one_vertex_in_bytes = sizeof(*quads_vertices_data) * 2;
    add_vertex_attribute(va);
  }
  { // circle
    Create_Vertex_Buffer vbo;
    Create_Index_Buffer<uint32> ibo;

    vbo.data = circles_vertices_data;
    vbo.size = static_array_size(circles_vertices_data);

    ibo.data = circles_indices_data;
    ibo.size = static_array_size(circles_indices_data);

    circle = create_vertex_array(vbo, ibo);

    Vertex_Attribute va;
    va.attribute_index = 0;
    va.number_of_values_in_an_attribute = 2;
    va.size_of_one_vertex_in_bytes = sizeof(*circles_vertices_data) * 2;
    add_vertex_attribute(va);
  }
  { // batched circles // @Incomplete: I need better circle drawing, because now we draw circles as GL_LINES, so we can't really batch them together, they are all connected to each other.
    size_t size = global_positions.size;
    float* vertex = (float*)alloca(size * static_array_size(circles_vertices_data) * sizeof(float));
    uint32* index = (uint32*)alloca(size * static_array_size(circles_indices_data) * sizeof(uint32));

    size_t vertex_count = 0;
    size_t index_count = 0;

    float norm_size = thread_data.particle_radius;

    for (size_t i = 0; i < size; i++) {
      Vec2 world = global_positions[i]; // these are world coordinates, but they are already normalized.
      

      for (size_t k = 0; k < static_array_size(circles_vertices_data); k += 2) {
        Vec2 local = { circles_vertices_data[k], circles_vertices_data[k + 1] }; // these are local coordinates.

        local = norm_size * local; // scale
        Vec2 pos = world + local;

        vertex[vertex_count++] = pos.x;
        vertex[vertex_count++] = pos.y;
      }

      size_t idx = i * static_array_size(circles_indices_data);
      for (size_t k = 0; k < static_array_size(circles_indices_data); k++) {
        index[index_count++] = idx + circles_indices_data[k];
      }
    }
    {
      Create_Vertex_Buffer vbo;
      Create_Index_Buffer<uint32> ibo;

      vbo.data = vertex;
      vbo.size = vertex_count;

      ibo.data = index;
      ibo.size = index_count;

      batched_circles = create_vertex_array(vbo, ibo);

      Vertex_Attribute va;
      va.attribute_index = 0;
      va.number_of_values_in_an_attribute = 2;
      va.size_of_one_vertex_in_bytes = sizeof(*vertex) * 2;
      add_vertex_attribute(va);
    }
  }


  { // batched quads
#if 0
  size_t     num_world_vertices = square(global_grid.number_of_cells_per_dimension);
  glm::vec2* world_vertices = (glm::vec2*) alloca(num_world_vertices * sizeof(*world_vertices));

  make_grid_positions(&global_grid, world_vertices);

  size_t num_vertices = num_world_vertices * static_array_size(quads_vertices_data);
  size_t num_indices  = num_world_vertices * static_array_size(quads_indices_data);

  size_t vertex_count = 0;
  size_t index_count  = 0;
  glm::vec2* vertex = (glm::vec2 *) alloca(num_vertices * sizeof(*vertex));
  uint32* index  = (uint32*) alloca(num_indices  * sizeof(*index));

  abstract_the_thing(world_vertices, num_world_vertices, (glm::vec2*) quads_vertices_data, static_array_size(quads_vertices_data), (uint*) quads_indices_data, static_array_size(quads_indices_data), vertex, &vertex_count, index, &index_count);

  // @Incomplete: 
  // apply scale & transform (can do that in a vertex shader).
  //

#if 0
  for (size_t i = 0; i < vertex_count; i++) {
    vertex[i] = 2.0f * vertex[i] - 1.0f;
    printf("%g\n", vertex[i]);
    assert(-1.0f <= vertex[i].x && vertex[i].x <= 1.0f);
    assert(-1.0f <= vertex[i].y && vertex[i].y <= 1.0f);
  }
#endif

#else
    size_t  n_cells   = global_grid.number_of_cells_per_dimension;
    float cell_size   = global_grid.cell_size;
    float norm_size   = 2.0f / (float)n_cells;
    float norm_offset = 0.5f;

    size_t number_of_batched_vertices = n_cells * n_cells * static_array_size(quads_vertices_data);
    float*  vertex = (float *) alloca(number_of_batched_vertices * sizeof(float));
    uint32* index  = (uint32*) alloca(number_of_batched_vertices * sizeof(uint32));

    size_t vertex_count = 0;
    size_t index_count  = 0;

    for (size_t i = 0; i < n_cells; i++) {
      for (size_t j = 0; j < n_cells; j++) {
        Vec2 center = { (float)(i / (float)n_cells), (float)(j) / (float)n_cells }; // this is my world coordinates!

        // 
        // I want to combine world & local coordinates.
        // how do I do that? 
        // world coordinates ARE NOT in normalized space.
        // local coordinates ARE in normalized space.
        //
        // first, we need to convert world cordinates to normalized space.
        // second, we need to translate and scale local quad.
        // then we can sum them up: center + local.
        //

        //center = norm_size * center  - Vec2{ 1.0f, 1.0f };


        for (size_t k = 0; k < static_array_size(quads_vertices_data); k += 2) {
          Vec2 local = { quads_vertices_data[k], quads_vertices_data[k+1] }; // this is my local coordinates.

          //local = norm_size * (local + Vec2{ norm_offset, norm_offset }); // translate & scale.
          Vec2 pos = center + local;

          pos = pos - Vec2{ norm_offset, norm_offset };

          vertex[vertex_count++] = pos.x;
          vertex[vertex_count++] = pos.y;
        }
          
        size_t idx = i * static_array_size(quads_indices_data);
        for (size_t k = 0; k < static_array_size(quads_indices_data); k++) {
          index[index_count++] = idx + quads_indices_data[k];
        }
      }
    }
#endif

    {
      Create_Vertex_Buffer vbo;
      vbo.data = (float*) vertex;
      vbo.size = vertex_count;

      Create_Index_Buffer<uint32> ibo;
      ibo.data = index;
      ibo.size = index_count;

      batched_quads = create_vertex_array(vbo, ibo);

      Vertex_Attribute va;
      va.attribute_index = 0;
      va.number_of_values_in_an_attribute = 2;
      va.size_of_one_vertex_in_bytes = sizeof(*vertex) * 2;
      add_vertex_attribute(va);
    }
  }
#endif
  {
    ImGui::StyleColorsDark();
  }
}

void deinit_program() {
  end_memory_arena(thread_data.arena);
}

void draw_triangle() {
  glBegin(GL_TRIANGLES);
    glVertex2f(-0.5f, -0.5f);
    glVertex2f(0.0f, 0.5f);
    glVertex2f(0.5f, -0.5f);
  glEnd();
}

void draw_circle(glm::vec2 position, glm::vec2 size) {
  size_t N = static_array_size(circles_vertices_data);

  glm::mat4 matrix = glm::translate(glm::mat4(1), glm::vec3(position, 0)) * glm::scale(glm::mat4(1), { size.x, size.y, 1.0f });

  glBegin(GL_LINES);
  for (size_t i = 0; i < N; i += 2) {
    glm::vec4 v;
    v.x = circles_vertices_data[i];
    v.y = circles_vertices_data[i+1];
    v.z = 0;
    v.w = 1;

    v = matrix * v;

    glVertex2f(v.x, v.y);

    v.x = circles_vertices_data[i+2];
    v.y = circles_vertices_data[i+3];
    v.z = 0;
    v.w = 1;

    v = matrix * v;

    if (i < N-2) glVertex2f(v.x, v.y);
  }
  glEnd();
}

void draw_grid() {
  size_t N        = global_grid.number_of_cells_per_dimension;
  float cell_size = global_grid.cell_size;
  float cap       = cell_size*N - 1;

  glBegin(GL_LINES);
  for (size_t i = 0; i < N+1; i++) { // N cells so (N+1) lines to draw.
    float w = i*cell_size - 1;
    glVertex2f(-1.0f, w);
    glVertex2f( cap, w);
    glVertex2f(w, -1.0f);
    glVertex2f(w,  cap);
  }
  glEnd();
}

void draw_circles(Thread_Data* data) {
  size_t N = global_positions.size;

  for (size_t i = 0; i < N; i++) {
    Vec2 v = global_positions[i];
    draw_circle(glm::vec2(v.x, v.y), glm::vec2(data->particle_radius, data->particle_radius));
  }
}


void update_and_render(GLFWwindow* window) {
  static bool show_demo_window  = false;
  static bool thread_is_paused  = false;
  static bool show_visual_ui    = true;
  static bool show_visual_line  = false;
  static bool show_visual_grid  = false;
  static bool show_visual_spheres = false;

  Vec4 visual_ui_min = {};
  Vec4 visual_ui_max = {};

  int width, height;
  glfwGetFramebufferSize(window, &width, &height);


  ImGuiIO& io = ImGui::GetIO();

  // Start the Dear ImGui frame
  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  if (show_demo_window) {
      ImGui::ShowDemoWindow(&show_demo_window);
  }

  // Dummy window.
  ImGui::Begin("#", NULL, ImGuiWindowFlags_NoCollapse || ImGuiWindowFlags_NoResize || ImGuiWindowFlags_NoTitleBar || ImGuiWindowFlags_NoScrollbar);
  ImVec2 v_min = ImGui::GetWindowContentRegionMin();
  ImVec2 v_max = ImGui::GetWindowContentRegionMax();

  v_min.x += ImGui::GetWindowPos().x;
  v_min.y += ImGui::GetWindowPos().y;
  v_max.x += ImGui::GetWindowPos().x;
  v_max.y += ImGui::GetWindowPos().y;
  ImGui::End();

  // y axis is flipped because screen coordinates increase from up to down. We can't workaround that :(
  const Vec4 min = { -1.0f,  1.0f, 0.0f, 1.0f };
  const Vec4 max = {  1.0f, -1.0f, 0.0f, 1.0f };

  visual_ui_min = { v_min.x, v_min.y, 0, 1 };
  visual_ui_max = { v_max.x, v_max.y, 0, 1 };

  Matrix4x4 t = transform_screen_space_to_normalized_space(width, height);
  visual_ui_min = t * visual_ui_min;
  visual_ui_max = t * visual_ui_max;
  // 


  {
    static const float step      = 0.0f;
    static const float step_fast = 0.0f;
    static const char* format    = "%.4f";
    static const ImGuiInputTextFlags flags = ImGuiInputTextFlags_CharsScientific;

    float particle_radius               = thread_data.particle_radius;
    float jumping_conductivity_distance = thread_data.jumping_conductivity_distance;
    float packing_factor                = thread_data.packing_factor;


    ImGui::Begin("Control Window");
    ImGui::Checkbox("Demo Window", &show_demo_window);
    ImGui::Checkbox("Visual UI", &show_visual_ui);
    ImGui::Checkbox("Visual Line",    &show_visual_line);
    ImGui::Checkbox("Visual Grid",    &show_visual_grid);
    ImGui::Checkbox("Visual Spheres", &show_visual_spheres);

    ImGui::InputFloat("Particle radius",               &particle_radius,               step, step_fast, format, flags);
    ImGui::InputFloat("Jumping conductivity distance", &jumping_conductivity_distance, step, step_fast, format, flags);
    ImGui::InputFloat("Packing factor",                &packing_factor,                step, step_fast, format, flags);

    ImGui::Text("Particle radius               := %.3f", particle_radius);
    ImGui::Text("Jumping conductivity distance := %.3f", jumping_conductivity_distance);
    ImGui::Text("Packing factor                := %.3f", packing_factor);
    ImGui::Text("Largest cluster size          := %lu",  result_largest_cluster_size);

    InterlockedExchange((uint*) &thread_data.particle_radius,               *(uint*) &particle_radius);
    InterlockedExchange((uint*) &thread_data.jumping_conductivity_distance, *(uint*) &jumping_conductivity_distance);
    InterlockedExchange((uint*) &thread_data.packing_factor,                *(uint*) &packing_factor);

  
    const char* start_or_continue = thread_is_paused ? "Continue" : "Start";
    if (ImGui::Button(start_or_continue)) {
      if (thread_is_paused) {
        resume_thread(&computation_thread);
        thread_is_paused = false;
      } else {
        if (!is_thread_running(&computation_thread)) {
          start_thread(&computation_thread, computation_thread_proc, &thread_data);
        }
      }
    }
    ImGui::SameLine();
    if (ImGui::Button("Pause")) {
      if (!thread_is_paused && is_thread_running(&computation_thread)) {
        suspend_thread(&computation_thread);
        thread_is_paused = true;
      }
    }
    ImGui::SameLine();
    if (ImGui::Button("Stop")) {
      if (is_thread_running(&computation_thread)) {
        kill_thread(&computation_thread);
      }
    }

    auto framerate = io.Framerate;
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f/framerate, framerate);

    ImGui::End();
  }

  // Render
  ImGui::Render();
  glViewport(0, 0, width, height);
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());


  draw_grid();
  draw_circles(&thread_data);

  Matrix4x4 projection = box_to_box_matrix(min, max, visual_ui_min, visual_ui_max);

  Basic_Shader_Data* data = (Basic_Shader_Data*)basic_shader.data;
  data->mvp = (float*) &projection;

  bind_shader(basic_shader);


  if (show_visual_ui) {

    if (show_visual_line) {
      draw_call(GL_LINES, line);
    }

    if (show_visual_grid) {
      draw_call(GL_LINES, batched_quads);
    }

    if (show_visual_spheres) {
      draw_call(GL_LINES, batched_circles);
    }
  }
}
