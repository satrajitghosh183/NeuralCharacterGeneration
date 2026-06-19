// Single translation unit that compiles the stb header-only libraries. Keeping the
// IMPLEMENTATION defines isolated here avoids multiple-definition link errors.
#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STBI_FAILURE_USERMSG
#include <stb_image.h>
#include <stb_image_write.h>
