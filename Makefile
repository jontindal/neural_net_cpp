CXX=g++
CPPFLAGS=-g -Wall -D_GLIBCXX_DEBUG

SRCS=initialization_function.cpp activation_function.cpp data_reader.cpp neural_network.cpp test_funcs.cpp
OBJS=$(subst .cpp,.o,$(SRCS))

all: main

release: CPPFLAGS=-O2 -DNDEBUG
release: main

main: $(OBJS) main.cpp
	$(CXX) $(CPPFLAGS) -o $@ $^

test: unit_tests
	./$<

unit_tests: $(OBJS) unit_tests.cpp
	$(CXX) $(CPPFLAGS) -o $@ $^

%.o: %.cpp %.hpp
	$(CXX) $(CPPFLAGS) -c $<

clean:
	rm -f $(OBJS) main unit_tests