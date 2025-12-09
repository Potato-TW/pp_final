all: hough_allopt.out

hough_allopt.out: hough_allopt.cpp
	g++ hough_allopt.cpp -O3 -std=c++17 `pkg-config --cflags --libs opencv4` -pthread -fopenmp -o hough_allopt.out

clean: 
	rm -f hough_allopt.out

# # 序列 2D version (serial)
# ./hough_allopt input.jpg --mode circle --backend serial2d -o out_serial_circle.png -t 1

# # pthread local-acc version (多執行緒)
# ./hough_allopt.out ../star.png --mode line --backend pthread -t 8 -o out_pthread_circle.png
# ./hough_allopt.out ../star.png --mode circle --backend pthread -t 8 -o out_pthread_circle.png

# # openmp local-acc version
# ./hough_allopt.out ../star.png --mode line --backend openmp -t 8 -o out_pthread_circle.png
# ./hough_allopt.out ../star.png --mode circle --backend openmp -t 8 -o out_pthread_circle.png
