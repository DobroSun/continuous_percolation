#vertex
#version 330 core

layout(location = 0) in vec2 position;

uniform mat4 uniform_mvp;

void main()
{
  vec4 result;
  result.xy = position;
  result.z  = 0;
  result.w  = 1;

  gl_Position = uniform_mvp * result;
}


#fragment
#version 330 core

layout(location = 0) out vec4 color;

void main()
{
  color = vec4(1, 0, 0, 0);
}

