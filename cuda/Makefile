NVCC = nvcc
NVFLAGS = -O3 -arch=sm_60 --compiler-options '-fPIC'
PKG_CFLAGS = $(shell pkg-config --cflags opencv4 2>/dev/null || pkg-config --cflags opencv)
PKG_LIBS = $(shell pkg-config --libs opencv4 2>/dev/null || pkg-config --libs opencv)

TARGET = hough_cuda

all: $(TARGET)

$(TARGET): main.cu
	$(NVCC) $(NVFLAGS) -o $(TARGET) main.cu $(PKG_CFLAGS) $(PKG_LIBS)

clean:
	rm -f $(TARGET)

