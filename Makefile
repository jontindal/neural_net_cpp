CXX = g++
CXXFLAGS = -Wall

CXXDBGFLAGS = -g -D_GLIBCXX_DEBUG
CXXRELFLAGS = -O3 -DNDEBUG

SRC_DIR = src
INC_DIRS = inc
DBGBUILDDIR = build/debug
RELBUILDDIR = build/release


SRCS = \
src/initialization_function.cpp \
src/activation_function.cpp \
src/data_reader.cpp \
src/neural_network.cpp \
src/test_funcs.cpp

INC_FLAGS = $(addprefix -I,$(INC_DIRS))

DBGOBJS = $(addprefix $(DBGBUILDDIR)/,$(notdir $(SRCS:.cpp=.o)))
DBGDEPS = $(DBGOBJS:.o=.d)

RELOBJS = $(addprefix $(RELBUILDDIR)/,$(notdir $(SRCS:.cpp=.o)))
RELDEPS = $(RELOBJS:.o=.d)

CXXFLAGS += $(INC_FLAGS) -MMD -MP

.PHONY: all debug test release time clean

all: debug


debug: $(DBGBUILDDIR)/main

test: $(DBGBUILDDIR)/unit_tests
	./$<

$(DBGBUILDDIR)/main: $(DBGOBJS) src/main.cpp
	$(CXX) $(CXXFLAGS) $(CXXDBGFLAGS) -o $@ $^

$(DBGBUILDDIR)/unit_tests: $(DBGOBJS) src/unit_tests.cpp
	$(CXX) $(CXXFLAGS) $(CXXDBGFLAGS) -o $@ $^

$(DBGBUILDDIR)/%.o: $(SRC_DIR)/%.cpp | $(DBGBUILDDIR)
	$(CXX) $(CXXFLAGS) $(CXXDBGFLAGS) -c $< -o $@

$(DBGBUILDDIR):
	mkdir --parents $@



release: $(RELBUILDDIR)/main

time: $(RELBUILDDIR)/perf_tests
	./$<

$(RELBUILDDIR)/main: $(RELOBJS) src/main.cpp
	$(CXX) $(CXXFLAGS) $(CXXRELFLAGS) -o $@ $^

$(RELBUILDDIR)/perf_tests: $(RELOBJS) src/perf_tests.cpp
	$(CXX) $(CXXFLAGS) $(CXXRELFLAGS) -o $@ $^

$(RELBUILDDIR)/%.o: $(SRC_DIR)/%.cpp | $(RELBUILDDIR)
	$(CXX) $(CXXFLAGS) $(CXXRELFLAGS) -c $< -o $@

$(RELBUILDDIR):
	mkdir --parents $@


clean:
	rm -rf $(DBGBUILDDIR) $(RELBUILDDIR)

-include $(DBGDEPS) $(RELDEPS)