#define GLEW_STATIC


#include <random>
#include "../../lib/pch.cpp"


const uint CELL_IS_EMPTY = 0xFFFFFFFF;
const size_t MEMORY_ALLOCATION_SIZE = 1e10; // ~ 10 gigabytes.


static const size_t DIM = 3;

const double PI  = 3.14159265358979323846;
const double TAU = 6.28318530717958647692;
const double MAX_POSSIBLE_PACKING_FACTOR_2D = PI * sqrt(3) / 6.0;
const double MAX_POSSIBLE_PACKING_FACTOR_3D = 0.74;

const float  left_boundary = -1.0f;
const float right_boundary =  1.0f;
const float lower_boundary = -1.0f;
const float upper_boundary =  1.0f;
const float near_boundary  = -1.0f;
const float far_boundary   =  1.0f;


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

struct Vertex_And_Fragment_Shader_Sources {
  literal vertex;
  literal fragment;
};

template<size_t M>
struct Vec;

template<>
struct Vec<2> {
  float x, y;
};

template<>
struct Vec<3> {
  float x, y, z;
};

typedef Vec<2> Vec2;

struct Vec4 {
  float x, y, z, w;
};

template<size_t M>
struct Line {
  Vec<M> origin;
  Vec<M> direction;
};

template<size_t M>
struct Circle {
  Vec<M> origin;
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

Vec<3> operator+(Vec<3> a, Vec<3> b)  { return { a.x + b.x, a.y + b.y, a.z + b.z }; }
Vec<3> operator-(Vec<3> a, Vec<3> b)  { return { a.x - b.x, a.y - b.y, a.z - b.z }; }
Vec<3> operator*(Vec<3> a, float c) { return { c * a.x, c * a.y, c * a.z }; }
Vec<3> operator/(Vec<3> a, float c) { return { a.x / c, a.y / c, a.z / c }; }
Vec<3> operator*(float c, Vec<3> a) { return a * c; }


template<size_t M>
double dot(Vec<M> a, Vec<M> b);

template<>
double dot<2>(Vec<2> a, Vec<2> b) {
  return a.x * b.x + a.y * b.y;
}

template<>
double dot<3>(Vec<3> a, Vec<3> b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

template<size_t M>
double length_sq(Vec<M> a) {
  return dot(a, a);
}

template<size_t M>
double length(Vec<M> a) {
  return sqrt(dot(a, a));
}
template<size_t M>
Vec<M> normalize(Vec<M> a) {
  return a / length(a);
}

template<size_t M>
bool line_and_circle_intersection(Line<M> a, Circle<M> b, double* distance) {
  Vec<M> dirr = a.direction;
  Vec<M> diff = a.origin - b.origin;

  float distance_between_circle_origin_and_line = sqrt(length_sq(diff) - length_sq(dot(diff, dirr)/length_sq(dirr) * dirr));
  if (distance) *distance = distance_between_circle_origin_and_line;
  return distance_between_circle_origin_and_line < b.radius;
}


template<size_t M>
struct Grid_Positionss;

template<>
struct Grid_Positionss<2> {
  uint i, j;
};

template<>
struct Grid_Positionss<3> {
  uint i, j, k;
};

typedef Grid_Positionss<2> Grid_Position;

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

template<size_t M>
struct Cluster_Data;

template<>
struct Cluster_Data<2> {
  bool touches_left  = false;
  bool touches_right = false;
  bool touches_up    = false;
  bool touches_down  = false;

  bool is_percolating_cluster = false;

  array<uint> cluster;
};

template<>
struct Cluster_Data<3> {
  bool touches_left  = false;
  bool touches_right = false;
  bool touches_up    = false;
  bool touches_down  = false;
  bool touches_near  = false;
  bool touches_far   = false;

  bool is_percolating_cluster = false;

  array<uint> cluster;
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

static Shader            basic_shader = {};
static Basic_Shader_Data basic_shader_data = {};

static Cluster_Data<DIM> cluster      = {};

static array<Vec<DIM>> global_positions = {};
static Grid2D          global_grid      = {};
static Graph           global_graph     = {};

static int64  result_largest_cluster_size = 0;


double generate_random_double_in_range(double from, double to) {
  static std::random_device rd;
  static std::mt19937 e(rd());

  std::uniform_real_distribution<double> dist(from, to);
  return dist(e);
}

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

template<size_t M>
Grid_Positionss<M> get_circle_position_on_a_grid(const Grid2D* grid, Vec<M> point);

template<>
Grid_Positionss<2> get_circle_position_on_a_grid(const Grid2D* grid, Vec<2> point) {
  float x = point.x - left_boundary;
  float y = point.y - lower_boundary;

  uint i = x / grid->cell_size;
  uint j = y / grid->cell_size;

  return { i, j };
}

template<>
Grid_Positionss<3> get_circle_position_on_a_grid(const Grid2D* grid, Vec<3> point) {
  float x = point.x - left_boundary;
  float y = point.y - lower_boundary;
  float z = point.z - near_boundary;

  uint i = x / grid->cell_size;
  uint j = y / grid->cell_size;
  uint k = z / grid->cell_size;

  return { i, j, k };
}

template<size_t M>
uint get_circle_id_on_a_grid(const Grid2D* grid, Vec<M> point);

template<>
uint get_circle_id_on_a_grid<2>(const Grid2D* grid, Vec<2> point) {
  Grid_Positionss<2> pos = get_circle_position_on_a_grid(grid, point);

  return pos.i * grid->number_of_cells_per_dimension + pos.j;
}

template<>
uint get_circle_id_on_a_grid<3>(const Grid2D* grid, Vec<3> point) {
  Grid_Positionss<3> pos = get_circle_position_on_a_grid(grid, point);

  return pos.k * square(grid->number_of_cells_per_dimension) + pos.i * grid->number_of_cells_per_dimension + pos.j;
}



template<size_t M>
double distance_squared(Vec<M> a, Vec<M> b) {
  return dot(a-b, a-b);
}

template<size_t M>
bool check_for_connection(Vec<M> a, Vec<M> b, float radius, float L) {
  return distance_squared(a, b) < square(L + 2*radius);
}

template<size_t M>
bool check_for_intersection(Vec<M> a, Vec<M> b, float radius) {
  return distance_squared(a, b) < square(2*radius);
}

template<size_t M>
void check_circle_touches_boundaries(Vec<M> pos, float radius, uint cluster_size, Cluster_Data<M>* result);

template<>
void check_circle_touches_boundaries(Vec<2> pos, float radius, uint cluster_size, Cluster_Data<2>* result) {
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
    result->is_percolating_cluster = true;
  }
}

template<>
void check_circle_touches_boundaries(Vec<3> pos, float radius, uint cluster_size, Cluster_Data<3>* result) {
  float diameter = 2*radius;

  bool touches_left  = pos.x <  left_boundary + diameter;
  bool touches_right = pos.x > right_boundary - diameter;
  bool touches_up    = pos.y < lower_boundary + diameter;
  bool touches_down  = pos.y > upper_boundary - diameter;
  bool touches_near  = pos.z <  near_boundary + diameter;
  bool touches_far   = pos.z >   far_boundary - diameter;

  bool* left  = &result->touches_left;
  bool* right = &result->touches_right;
  bool* up    = &result->touches_up;
  bool* down  = &result->touches_down;
  bool* near_  = &result->touches_near;
  bool* far_   = &result->touches_far;

  *left  = *left  ? *left  : touches_left;
  *right = *right ? *right : touches_right;
  *up    = *up    ? *up    : touches_up;
  *down  = *down  ? *down  : touches_down;
  *near_  = *near_  ? *near_  : touches_near;
  *far_   = *far_   ? *far_   : touches_far;

  if ((*up && *down) || (*left && *right) || (*near_ && *far_)) {
    result->is_percolating_cluster = true;
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

template<size_t M>
void tightest_packing_sampling(array<Vec<M>>* array, float* max_random_walking_distance, float target_radius, float packing_factor);

template<>
void tightest_packing_sampling<2>(array<Vec<2>>* array, float* max_random_walking_distance, float target_radius, float packing_factor) {
  assert(array->size == 0);

  // target_radius^2 -- packing_factor.
  // radius       ^2 -- MAX_POSSIBLE_PACKING_FACTOR.
  float radius = sqrt(square(target_radius) * MAX_POSSIBLE_PACKING_FACTOR_2D / packing_factor);
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

template<>
void tightest_packing_sampling<3>(array<Vec<3>>* array, float* max_random_walking_distance, float target_radius, float packing_factor) {
  assert(array->size == 0);

  // target_radius^2 -- packing_factor.
  // radius       ^2 -- MAX_POSSIBLE_PACKING_FACTOR.
  float radius = sqrt(square(target_radius) * MAX_POSSIBLE_PACKING_FACTOR_3D / packing_factor); // @Incomplete: ?
  float height = sqrt(3) * radius;

  *max_random_walking_distance = radius - target_radius;

  // centers of a circle.
  double x;
  double y;
  double z;

  bool is_even_x = true;
  bool is_even_y = true;

  z = near_boundary + radius;
  while (z < far_boundary - radius) {

    y  = lower_boundary;
    y += (is_even_y) ? radius : 2*radius;

    while (y < upper_boundary - radius) {

      x  = left_boundary;
      x += (is_even_x) ? radius : 2*radius;

      while (x < right_boundary - radius) {
        array_add(array, Vec<3>{ (float) x, (float) y, (float) z});
        x += 2*radius;
      }

      y += 2*radius;
      is_even_x = !is_even_x;
    }

    z += height;
    is_even_y = !is_even_y;
  }
}


template<size_t M>
void create_square_grid(Grid2D* grid, array<Vec<M>>* positions) {
  for (size_t k = 0; k < positions->size; k++) {
    Vec<M>* point = &(*positions)[k];

    uint n = get_circle_id_on_a_grid(grid, *point);

    assert(grid->data[n] == CELL_IS_EMPTY);
    grid->data[n] = k;
  }
}


template<size_t M>
void take_all_neighbours_with_distance(const Grid2D* grid, array<Grid_Cell>* neighbours, Vec<M> point, float radius, float distance);

template<>
void take_all_neighbours_with_distance<2>(const Grid2D* grid, array<Grid_Cell>* neighbours, Vec<2> point, float radius, float distance) {
  Vec<2> new_point = point + Vec<2>{ distance, distance };

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

template<>
void take_all_neighbours_with_distance<3>(const Grid2D* grid, array<Grid_Cell>* neighbours, Vec<3> point, float radius, float distance) {
  Vec<3> new_point = point + Vec<3>{ distance, distance, distance };

  Grid_Positionss<3> p = get_circle_position_on_a_grid(grid,     point);
  Grid_Positionss<3> n = get_circle_position_on_a_grid(grid, new_point);

  size_t width = grid->number_of_cells_per_dimension;
  size_t di = abs((int64)n.i - (int64)p.i);
  size_t dj = abs((int64)n.j - (int64)p.j);
  size_t dk = abs((int64)n.k - (int64)p.k);

  size_t delta = max(dk, max(di, dj)) + 1;


  // 
  // now let's just iterate over all of the: possible neighbours. and find those that contain circles.
  //  

  size_t pks = p.k - delta;
  size_t pke = p.k + delta;

  size_t pis = p.i - delta;
  size_t pie = p.i + delta;

  size_t pjs = p.j - delta;
  size_t pje = p.j + delta;

  for (size_t k = pks; k < pke; k++) {
    for (size_t i = pis; i < pie; i++) {
      for (size_t j = pjs; j < pje; j++) {

        if (i >= width || j >= width || k >= width) { continue; }
        if (i == p.i && j == p.j && k == p.k)       { continue; }

        size_t index = k*square(width) + i*width + j;
        assert(index < grid->number_of_cells);

        Grid_Cell cell_id = grid->data[index];
        if (cell_id != CELL_IS_EMPTY) {
          array_add(neighbours, cell_id);
        }
      }
    }
  }
}


template<size_t M>
Vec<M> make_unit_direction();

template<>
Vec<2> make_unit_direction<2>() {
  double dir_x = generate_random_double_in_range(0, 1);  // @Incomplete: this is not a uniform distribution for an angle.
  double dir_y = generate_random_double_in_range(0, 1);
  return normalize(Vec<2>{ (float) dir_x, (float) dir_y });
}

template<>
Vec<3> make_unit_direction<3>() {
  double dir_x = generate_random_double_in_range(0, 1);  // @Incomplete: this is not a uniform distribution for an angle.
  double dir_y = generate_random_double_in_range(0, 1);
  double dir_z = generate_random_double_in_range(0, 1);
  return normalize(Vec<3>{ (float) dir_x, (float) dir_y, (float) dir_z });
}


template<size_t M>
Vec<M> clamp_boundaries(Vec<M> a, float);

template<>
Vec<2> clamp_boundaries(Vec<2> a, float radius) {
  a.x = clamp(a.x,  left_boundary+radius, right_boundary-radius);
  a.y = clamp(a.y, lower_boundary+radius, upper_boundary-radius);
  return a;
}

template<>
Vec<3> clamp_boundaries(Vec<3> a, float radius) {
  a.x = clamp(a.x,  left_boundary+radius, right_boundary-radius);
  a.y = clamp(a.y, lower_boundary+radius, upper_boundary-radius);
  a.z = clamp(a.z,  near_boundary+radius,   far_boundary-radius);
  return a;
}

template<size_t M>
void process_random_walk(Grid2D* grid, array<Vec<M>>* positions, float radius, float max_random_walking_distance) {
#if 1
  array<Grid_Cell> neighbours;

  defer { array_free(&neighbours); };

  for (size_t i = 0; i < positions->size; i++) {
    for (size_t n = 0; n < 4; n++) {
      Vec<M>* point = &(*positions)[i];

      array_clear(&neighbours);
      take_all_neighbours_with_distance(grid, &neighbours, *point, radius, max_random_walking_distance);


      Vec<M> unit_direction = make_unit_direction<M>();
      double distance       = generate_random_double_in_range(0, max_random_walking_distance);


      Vec<M> jump_to = *point + distance*unit_direction;

      jump_to = clamp_boundaries(jump_to, radius);

      uint p_id = get_circle_id_on_a_grid(grid, *point);
      uint n_id = get_circle_id_on_a_grid(grid, jump_to);

      Grid_Cell* previous = &grid->data[p_id];
      Grid_Cell* next     = &grid->data[n_id];
      bool cell_is_not_occupied = *next == CELL_IS_EMPTY;

      if (*previous == *next) {
        *point = jump_to;
        continue;
      }


      bool do_not_intersect = true;
      if (cell_is_not_occupied) {
        // 
        // just check that we don't intersect anyone.
        // 
      
        for (size_t k = 0; k < neighbours.size; k++) {
          Grid_Cell cell     = neighbours[k];
          Vec<M>   neighbour = (*positions)[cell];

          if (check_for_intersection(*point, neighbour, radius)) {
            do_not_intersect = false;
            break;
          }
        }
      } else {
        if (*previous == *next) {
          // cell is occupied by us.
        } else {
          // cell is ocuupied by someone else.
          do_not_intersect = false;
        }
      }


      if (*previous != *next && do_not_intersect) {
        assert(*next == CELL_IS_EMPTY);

        *point    = jump_to;
        *next     = *previous;
        *previous = CELL_IS_EMPTY;
      } else {
        // Rejected.
      }
    }
  }
#else
  array<Grid_Cell> neighbours;
  array<Grid_Cell> intersection;
  array<double>    intersection_distance;
  array<double>    distances;

  defer { array_free(&neighbours); };
  defer { array_free(&intersection); };
  defer { array_free(&intersection_distance); };
  defer { array_free(&distances); };

  for (size_t i = 0; i < positions->size; i++) {
    for (size_t n = 0; n < 4; n++) {
      array_clear(&neighbours);
      array_clear(&intersection);
      array_clear(&intersection_distance);
      array_clear(&distances);

      Vec<M>* point = &(*positions)[i];

      // find all neighbours.
      take_all_neighbours_with_distance(grid, &neighbours, *point, radius, max_random_walking_distance);

      // 
      // now we should find the one that is in our way.
      // in order to do that, we should know the direction to move in.
      //

      Vec<M> unit_direction = make_unit_direction<M>();

      Line<M> a;
      a.origin    = *point;
      a.direction = unit_direction;

      // find all circles that are intersecting unit_direction.
      for (size_t k = 0; k < neighbours.size; k++) {
        Grid_Cell cell = neighbours[k];
        Vec<M>*   origin = &(*positions)[cell];

        Circle<M> b;
        b.origin = *origin;
        b.radius = 2*radius; // because we want to put a circle in there with another radius, not just simple line & circle intersection check.

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
        Vec<M>*   origin = &(*positions)[cell];

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
      Vec<M>    jump_to = *point + distance*unit_direction;

      jump_to = clamp_boundaries(jump_to, radius);

      uint p_id = get_circle_id_on_a_grid(grid, *point);
      uint n_id = get_circle_id_on_a_grid(grid, jump_to);

      Grid_Cell* previous = &grid->data[p_id];
      Grid_Cell* next     = &grid->data[n_id];
      bool cell_is_not_occupied = *next == CELL_IS_EMPTY;
      
      assert(*previous == *next ? true : cell_is_not_occupied); // if we did a jump, next_id should be empty.

      *point = jump_to;

      if (*previous != *next) {
        assert(*next == CELL_IS_EMPTY);
        *next     = *previous;
        *previous = CELL_IS_EMPTY;
      }
    }
  }
#endif
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

template<size_t M>
void collect_points_to_graph_via_grid(Graph* graph, const Grid2D* grid, const array<Vec<M>>* positions, float radius, float L) {
  assert(positions->size == graph->count);

  array<Grid_Cell> neighbours;
  defer { array_free(&neighbours); };

  for (size_t k = 0; k < positions->size; k++) {
    Vec<M> point = (*positions)[k];

    array_clear(&neighbours);
    take_all_neighbours_with_distance(grid, &neighbours, point, radius, L);

    graph->connected_nodes[k] = graph->graph_data;
    for (size_t c = 0; c < neighbours.size; c++) {
      Grid_Cell cell_id = neighbours[c];

      assert(cell_id != CELL_IS_EMPTY);
      Vec<M> neighbour = (*positions)[cell_id];

      if (check_for_connection(point, neighbour, radius, L)) {
        add_connection_to_graph_node(graph, k, cell_id);
      }
    }
  }
}

template<size_t M>
void breadth_first_search(const Graph* graph, Queue* queue, const array<Vec<M>>* positions, bool* hash_table, uint starting_index, float radius, array<uint>* cluster_sizes, Cluster_Data<M>* result) {

  uint* size = array_add(cluster_sizes);

  // add i-th node.
  hash_table[starting_index] = true;
  add_to_queue(queue, starting_index);
  *size = 1;

  {
    Vec<M> pos = (*positions)[starting_index];

    check_circle_touches_boundaries(pos, radius, *size, result);
    array_add(&result->cluster, starting_index);
  }

  while (!is_queue_empty(queue)) {
    uint node_id = get_from_queue(queue);
    
    const uint*   nodes_to_search = graph->connected_nodes[node_id];
    const uint16  nodes_count     = graph->connected_nodes_count[node_id];

    for (uint i = 0; i < nodes_count; i++) {
      uint new_node_id = nodes_to_search[i];

      if (hash_table[new_node_id]) {
        // 
        // the node is already in a queue, we will process it anyway.
        //
      } else {
        hash_table[new_node_id] = true;
        add_to_queue(queue, new_node_id);
        *size += 1;

        Vec<M> pos = (*positions)[new_node_id];
        check_circle_touches_boundaries(pos, radius, *size, result);
        array_add(&result->cluster, new_node_id);
      }
    }
  }
}

template<size_t M>
void copy_cluster_data(Cluster_Data<M>* a, const Cluster_Data<M>* b) {
  array<uint> reference = a->cluster;

  *a = *b;

  array_copy(&reference, &b->cluster);
  a->cluster = reference;
}

template<size_t M>
void clear_cluster_data(Cluster_Data<M>* a) {
  array<uint> reference = a->cluster;
  array_clear(&reference);

  *a = {};
  a->cluster = reference;
}
























Vertex_And_Fragment_Shader_Sources load_shaders(literal filename) {
  literal s = read_entire_file(filename); // @MemoryLeak: 
  if (!s.count) return {};               // @MemoryLeak: 

  static const uint TAG_NOT_FOUND  = (uint) -1;
  static const literal vertex_tag   = make_literal("#vertex");
  static const literal fragment_tag = make_literal("#fragment");
  static const literal tags[]       = { vertex_tag, fragment_tag };

  bool vertex_found   = false;
  bool fragment_found = false;
  bool vertex_two_or_more_occurrences   = false;
  bool fragment_two_or_more_occurrences = false;

  literal shaders[2] = {};

  uint index  = TAG_NOT_FOUND;
  const char* cursor = s.data;
  literal* current_shader = NULL;

  while (*cursor != '\0') {
    bool vertex   = tags[0] == cursor;
    bool fragment = tags[1] == cursor;

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

unsigned create_shader(literal vertex, literal fragment) {
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


template<size_t M>
void do_the_thing(Memory_Arena* arena, Cluster_Data<M>* cluster, float radius, float L, float packing_factor, size_t* largest_cluster_size) {
  float max_random_walking_distance;

  clear_cluster_data(cluster);
  reset_memory_arena(arena);

  array<Vec<M>> positions = {};
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

  float max_window_area;
  if (M == 2) {
    max_window_area = square(one_dimension_range);
  } else if (M == 3) {
    max_window_area = pow(one_dimension_range, 3);
  }

  float max_circles_area;
  if (M == 2) {
    max_circles_area = positions.size * PI * square(radius);
  } else if (M == 3) {
    max_circles_area = positions.size * 4/3.0f * PI * pow(radius, 3);
  }

  const float experimental_packing_factor = max_circles_area / max_window_area;

  printf("[..]\n");
  printf("[..] Radius of a circle   := %g\n", radius);
  printf("[..] Connection radius(L) := %g\n", L);
  printf("{..] Packing factor       := %g\n", packing_factor);
  printf("[..] Generated packing factor := %g\n", experimental_packing_factor);
  printf("[..] Generated points     := %zu\n", positions.size);

  assert(N < UINT_MAX);                                              // because we are using uints to address graph nodes, N is required to be less than that.
  // assert(packing_factor < MAX_POSSIBLE_PACKING_FACTOR);              // packing factor must be less than 0.9069... ( square grid only!!! )
  // assert(fabs(experimental_packing_factor - packing_factor) < 1e-1); // @Incomplete: 


  Grid2D grid; 
  grid.cell_size                     = sqrt(2)*radius;
  grid.number_of_cells_per_dimension = one_dimension_range / grid.cell_size;
  grid.number_of_cells               = pow(grid.number_of_cells_per_dimension, M);
  grid.data                          = (Grid_Cell*) alloc(arena->allocator, sizeof(*grid.data) * grid.number_of_cells).memory;

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
  // assert(check_circles_are_inside_a_box(&positions, radius));
  // assert(check_circles_do_not_intersect_each_other(&positions, radius));

  Graph graph;
  graph.count = N;
  graph.connected_nodes_count = (uint8*) alloc(arena->allocator, sizeof(uint8) * (arena->allocated - arena->top)).memory; // allocate to max capacity.
  graph.connected_nodes       = (uint**) ((char*)graph.connected_nodes_count + sizeof(*graph.connected_nodes_count) * N);
  graph.graph_data            = (uint*)  ((char*)graph.connected_nodes       + sizeof(*graph.connected_nodes)       * N);

  memset(graph.connected_nodes_count, 0, sizeof(*graph.connected_nodes_count) * N);

  {
    printf("[..]\n");
    printf("[..] Collecting nodes to graph ... \n");
    printf("[..] Finished creating graph in := ");
    measure_scope();
    //naive_collect_points_to_graph(&graph, &positions, radius, L);
    collect_points_to_graph_via_grid(&graph, &grid, &positions, radius, L);
  }


  Memory_Arena temp;
  begin_memory_arena(&temp, MB(200));

  array<uint> cluster_sizes;
  cluster_sizes.allocator = temp.allocator;

  bool* hash_table = (bool*) alloc(temp.allocator, sizeof(bool) * N).memory; // @Incomplete: instead of using 1 byte, we can use 1 bit => 8 times less memory for a hash_table.
  memset(hash_table, false, sizeof(bool) * N);

  Queue queue;
  queue.data     = (uint*) alloc(temp.allocator, sizeof(uint) * N).memory;
  queue.first    = 0;
  queue.last     = 0;
  queue.max_size = N;

  {
    printf("[..]\n");
    printf("[..] Starting BFS ... \n");
    printf("[..] Finished BFS in := ");
    measure_scope();

    for (size_t i = 0; i < graph.count; i++) {
      if (!hash_table[i]) {

        Cluster_Data<M> result;
        result.cluster.allocator = temp.allocator;

        breadth_first_search(&graph, &queue, &positions, hash_table, i, radius, &cluster_sizes, &result);

        bool larger_cluster      = result.cluster.size > cluster->cluster.size;
        bool percolating_cluster = result.is_percolating_cluster;
        if (larger_cluster || percolating_cluster) {
          copy_cluster_data(cluster, &result);
        }
      }
    }
  }
  //assert(check_hash_table_is_filled_up(hash_table, N));

  *largest_cluster_size = cluster->cluster.size;

  global_positions = positions;
  global_grid      = grid;
  global_graph     = graph;

  {
    printf("[..]\n");
    printf("[..] Is percolating cluster := %s\n",    cluster->is_percolating_cluster ? "true" : "false");
    printf("[..] Largest cluster size   := %zu\n",   cluster->cluster.size);
    printf("[..] Number of clusters found := %zu\n", cluster_sizes.size);
    printf("[..] Cluster sizes := [");
    for (size_t i = 0; i < cluster_sizes.size; i++) {
      printf("%lu%s", cluster_sizes[i], (i == cluster_sizes.size-1) ? "]\n" : ", ");
    }
    puts("");
    puts("");
    puts("");
  }

  end_memory_arena(&temp);

}

Memory_Arena arena;
array<Vec<2>> positions;

float particle_radius;
float connection_radius;
float packing_factor;

void init_program(int width, int height) {
  init_filesystem_api();
  init_threads_api();

  check_filesystem_api();
  check_threads_api();

  init_circles_vertices_and_indices_data();

  begin_temporary_storage(&temporary_storage, MB(256));
  begin_memory_arena(&arena, GB(5));

  {
    ImGui::StyleColorsDark();
  }

  void do_the_thing2();
  do_the_thing2();
}

void do_the_thing2() {
  positions.allocator = arena.allocator;

  float max_random_walking_distance;

  particle_radius = 0.1;
  connection_radius = 0.7;
  packing_factor = 0.4;


  {
    print("[..] Generating nodes ... \n");
    print("[..] Finished sampling points in := ");

    measure_scope();
    //naive_random_sampling(&positions, radius);
    //poisson_disk_sampling(&positions, N, radius);
    tightest_packing_sampling(&positions, &max_random_walking_distance, particle_radius, packing_factor);
  }

  size_t N = positions.size;
  float one_dimension_range = right_boundary - left_boundary;
  float max_window_area = square(one_dimension_range);
  float max_circles_area = positions.size * PI * square(particle_radius);;
  float experimental_packing_factor = max_circles_area / max_window_area;

  printf("\n[..]\n");
  printf("[..] Radius of a circle   := %g\n", particle_radius);
  printf("[..] Connection radius(L) := %g\n", connection_radius);
  printf("{..] Packing factor       := %g\n", packing_factor);
  printf("[..] Generated packing factor := %g\n", experimental_packing_factor);
  printf("[..] Generated points     := %zu\n", positions.size);

  assert(N < UINT_MAX);                                              // because we are using uints to address graph nodes, N is required to be less than that.
  // assert(packing_factor < MAX_POSSIBLE_PACKING_FACTOR);              // packing factor must be less than 0.9069... ( square grid only!!! )
  // assert(fabs(experimental_packing_factor - packing_factor) < 1e-1); // @Incomplete: 

}
void deinit_program() {
  end_temporary_storage(&temporary_storage);

  end_memory_arena(&arena);
  array_free(&cluster.cluster);
}

void draw_triangle() {
  glBegin(GL_TRIANGLES);
    glVertex2f(-0.5f, -0.5f);
    glVertex2f(0.0f, 0.5f);
    glVertex2f(0.5f, -0.5f);
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

void draw_circle(glm::vec2 position, glm::vec2 size, glm::vec3 color) {
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
    // glColor3f(color.r, color.g, color.b);

    v.x = circles_vertices_data[i+2];
    v.y = circles_vertices_data[i+3];
    v.z = 0;
    v.w = 1;

    v = matrix * v;

    if (i < N-2) {
      glVertex2f(v.x, v.y);
      // glColor3f(color.r, color.g, color.b);
    }
  }
  glEnd();
}

void draw_circles() {
  uint N = positions.size;

  glm::vec3 red   = glm::vec3(1, 0, 0);
  glm::vec3 green = glm::vec3(0, 1, 0);

  glm::vec3 color = green;

  for (uint i = 0; i < N; i++) {
    Vec2 v = positions[i];

    draw_circle(glm::vec2(v.x, v.y), glm::vec2(particle_radius, particle_radius), color);
  }
}

void update_and_render(GLFWwindow* window) {

  int width, height;
  glfwGetFramebufferSize(window, &width, &height);

#if 0
  ImGuiIO& io = ImGui::GetIO();

  // Start the Dear ImGui frame
  ImGui_ImplOpenGL3_NewFrame();
  ImGui_ImplGlfw_NewFrame();
  ImGui::NewFrame();

  static bool show_demo_window = true;
  if (show_demo_window) {

    ImGui::ShowDemoWindow(&show_demo_window);
  }

  {
    static const float step      = 0.0f;
    static const float step_fast = 0.0f;
    static const char* format    = "%.4f";
    static const ImGuiInputTextFlags flags = ImGuiInputTextFlags_CharsScientific;


    ImGui::Begin("Control Window");
    ImGui::Checkbox("Demo Window", &show_demo_window);

    ImGui::Text("Particle radius               := %.3f", particle_radius);
    ImGui::Text("Jumping conductivity distance := %.3f", connection_radius);
    ImGui::Text("Packing factor                := %.3f", packing_factor);
    ImGui::Text("Largest cluster size          := %lu",  0);
    ImGui::Text("Is percolating cluster found  := %s",   false ? "true" : "false");

    auto framerate = io.Framerate;
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f/framerate, framerate);

    ImGui::End();
  }

  // Render
  ImGui::Render();
  glViewport(0, 0, width, height);
  ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
#endif

  // draw_triangle();

  // //draw_circles<DIM>(&thread_data);
  draw_circles();
}
