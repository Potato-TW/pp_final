NVCC = nvcc
NVFLAGS = -O3 -arch=sm_60 --compiler-options '-fPIC'
PKG_CFLAGS = $(shell pkg-config --cflags opencv4 2>/dev/null || pkg-config --cflags opencv)
PKG_LIBS = $(shell pkg-config --libs opencv4 2>/dev/null || pkg-config --libs opencv)

TARGET = hough_cuda.out
TARGET_STAR = hough_star_cuda.out

all: $(TARGET) $(TARGET_STAR)

$(TARGET): main.cu
	$(NVCC) $(NVFLAGS) -o $(TARGET) main.cu $(PKG_CFLAGS) $(PKG_LIBS)

$(TARGET_STAR): main_star.cu
	$(NVCC) $(NVFLAGS) -o $(TARGET_STAR) main_star.cu $(PKG_CFLAGS) $(PKG_LIBS)

clean:
	rm -f $(TARGET) $(TARGET_STAR)

