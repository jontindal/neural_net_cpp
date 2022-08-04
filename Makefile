CXX=g++
CPPFLAGS=-g -Wall -D_GLIBCXX_DEBUG
CPPRELEASEFLAGS=-O3 -DNDEBUG

SRCS=initialization_function.cpp activation_function.cpp data_reader.cpp neural_network.cpp test_funcs.cpp
OBJS=$(subst .cpp,.o,$(SRCS))

all: main

release: CPPFLAGS=$(CPPRELEASEFLAGS)
release: main

main: $(OBJS) main.cpp
	$(CXX) $(CPPFLAGS) -o $@ $^

test: unit_tests
	./$<

time: perf_tests
	./$<

unit_tests: $(OBJS) unit_tests.cpp
	$(CXX) $(CPPFLAGS) -o $@ $^

perf_tests: CPPFLAGS=$(CPPRELEASEFLAGS)
perf_tests: $(OBJS) perf_tests.cpp
	$(CXX) $(CPPFLAGS) -o $@ $^

%.o: %.cpp %.hpp
	$(CXX) $(CPPFLAGS) -c $<

clean:
	rm -f $(OBJS) main unit_tests