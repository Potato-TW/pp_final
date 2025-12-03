CXX = g++
CXXFLAG = -O3 `pkg-config --cflags --libs opencv4`

TARGET = opencv_hough.out

all: $(TARGET)

$(TARGET): main.cpp
	$(CXX) main.cpp -o $(TARGET) $(CXXFLAG)

clean:
	rm -f $(TARGET)

# g++ opencv_hough.cpp -o opencv_ht -O3 `pkg-config --cflags --libs opencv4`
# ./opencv_hough.out line ../lena.png out_line.png
# ./opencv_hough.out circle ../lena.png out_circle.png