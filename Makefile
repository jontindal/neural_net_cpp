CXX=g++
CPPFLAGS=-g -Wall -D_GLIBCXX_DEBUG

SRCS=activation_function.cpp data_reader.cpp neural_network.cpp
OBJS=$(subst .cpp,.o,$(SRCS))

all: main

release: CPPFLAGS=-O2 -DNDEBUG
release: main

main: $(OBJS) main.cpp
	$(CXX) $(CPPFLAGS) -o $@ $^

%.o: %.cpp %.hpp
	$(CXX) $(CPPFLAGS) -c $<

clean:
	rm $(OBJS) main