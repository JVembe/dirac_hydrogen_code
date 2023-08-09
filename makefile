
CXX = g++

MPIXX = mpic++

CXXFLAGS = -ggdb3 -fopenmp -march=native -O3

LDLIBS = -lwignerSymbols

SOURCEDIR = src
BUILDDIR = build

SOURCES = $(wildcard $(SOURCEDIR)/*.cpp)

HEADERS = $(wildcard $(SOURCEDIR)/*.h)

OBJECTS = $(patsubst $(SOURCEDIR)/%.cpp,$(BUILDDIR)/%.o,$(SOURCES))

coreFiles := splinehandler.cpp dkbbasis.cpp spnrbasis.cpp

coreObjs = $(patsubst %.cpp,$(BUILDDIR)/%.o,$(coreFiles))

mpiObjs = $(patsubst %.cpp,$(BUILDDIR)/%.o,$(coreFiles))


$(OBJECTS): $(BUILDDIR)/%.o: $(SOURCEDIR)/%.cpp $(HEADERS)
	$(MPIXX) -c -o $@ $< $(CXXFLAGS)
	
$(coreObjs): $(BUILDDIR)/%.o: $(SOURCEDIR)/%.cpp $(HEADERS)
	$(CXX) -c -o $@ $< $(CXXFLAGS)

$(mpiObjs): $(BUILDDIR)/%.o: $(SOURCEDIR)/%.cpp $(HEADERS)
	$(MPIXX) -c -o $@ $< $(CXXFLAGS)


mpitest: $(mpiObjs) $(BUILDDIR)/eigenMPI.o
	$(MPIXX) $(CXXFLAGS) $(coreObjs) $(BUILDDIR)/eigenMPI.o $(LDLIBS) -o EMPI.out
	
dipolempitest: $(mpiObjs) $(BUILDDIR)/dipoleMPItest.o
	$(MPIXX) $(CXXFLAGS) $(coreObjs) $(BUILDDIR)/dipoleMPItest.o $(LDLIBS) -o dipoleMPItest.out


clean:
	rm -f $(BUILDDIR)/*.o all
