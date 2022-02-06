
struct File_Windows {
  HANDLE handle;
};
static_assert(sizeof(File) >= sizeof(File_Windows), "");

bool file_open_windows(File* file, const char* filename) {
  File_Windows* f = (File_Windows*) file;

  f->handle = CreateFile(filename,
                         GENERIC_READ | GENERIC_WRITE,
                         0,
                         NULL,
                         OPEN_EXISTING,
                         FILE_ATTRIBUTE_NORMAL,
                         NULL);

  printf("%s := ", "open");
  if (!f->handle || f->handle == INVALID_HANDLE_VALUE) {
    printf("false := %d\n", GetLastError());
    return false; // @LogError: 
  }

  // @LogError: check if ERROR_FILE_NOT_FOUND.
  printf("true\n");

  return true;
}

bool file_close_windows(File* file) {
  File_Windows* f = (File_Windows*) file;

  printf("%s", "close\n");

  // @LogError: 
  return CloseHandle(f->handle);
}

bool file_read_windows(File* file, void* buffer, size_t to_read, size_t* written) {
  File_Windows* f = (File_Windows*) file;

  DWORD bytes_written;
  BOOL success = ReadFile(f->handle, buffer, (DWORD)to_read, &bytes_written, NULL); // @ErrorProne: when we are going to read file larger than DWORD can address we are gonna crash right here. @PassUIntsNotSizeT: 
  *written = bytes_written;

  printf("%s := ", "read");
  if (!success) {
    printf("false := %d\n", GetLastError());
    return false; // @LogError: 
  }
  printf("true; written := %d\n", bytes_written);
  return true;
}

size_t file_get_size_windows(File* file) {
  File_Windows* f = (File_Windows*) file;
  DWORD hi = 0;
  DWORD lo = GetFileSize(f->handle, &hi);

  size_t hi64 = hi;
  size_t lo64 = lo;

  printf("%s := ", "get size");
  if (lo == INVALID_FILE_SIZE) {
    printf("false := %d\n", GetLastError());
    return 0; // @LogError: 
  }
  printf("true\n");
  return (hi64 << sizeof(DWORD)) | lo64;
}

void init_filesystem_api() {
  file_open     = file_open_windows;
  file_close    = file_close_windows;
  file_read     = file_read_windows;
  file_get_size = file_get_size_windows;
}
