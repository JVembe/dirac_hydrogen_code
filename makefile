
CXX = g++

MPIXX = mpic++

CXXFLAGS = -ggdb3 -fopenmp -march=native -O3

LDLIBS = -lwignerSymbols

SOURCEDIR = src
BUILDDIR = build

SOURCES = $(wildcard $(SOURCEDIR)/*.cpp)

HEADERS = $(wildcard $(SOURCEDIR)/*.h)

OBJECTS = $(patsubst $(SOURCEDIR)/%.cpp,$(BUILDDIR)/%.o,$(SOURCES))

coreFiles := bsplinebasis.cpp dkbbasis.cpp spharmbasis.cpp spnrbasis.cpp

coreObjs = $(patsubst %.cpp,$(BUILDDIR)/%.o,$(coreFiles))

mpiObjs = $(patsubst %.cpp,$(BUILDDIR)/%.o,$(coreFiles))


$(OBJECTS): $(BUILDDIR)/%.o: $(SOURCEDIR)/%.cpp $(HEADERS)
	$(MPIXX) -c -o $@ $< $(CXXFLAGS)
	
$(coreObjs): $(BUILDDIR)/%.o: $(SOURCEDIR)/%.cpp $(HEADERS)
	$(CXX) -c -o $@ $< $(CXXFLAGS)

$(mpiObjs): $(BUILDDIR)/%.o: $(SOURCEDIR)/%.cpp $(HEADERS)
	$(MPIXX) -c -o $@ $< $(CXXFLAGS)


all: $(coreObjs) $(BUILDDIR)/testwfunc.o
	$(CXX) $(CXXFLAGS) $(coreObjs) $(BUILDDIR)/testwfunc.o $(LDLIBS) -o c.out

schrodinger: $(coreObjs) $(BUILDDIR)/schrWfunc.o
	$(CXX) $(CXXFLAGS) $(coreObjs) $(BUILDDIR)/schrWfunc.o $(LDLIBS) -o a.out

bdpSchr: $(coreObjs) $(BUILDDIR)/bdpSchr.o
	$(CXX) $(CXXFLAGS) $(coreObjs) $(BUILDDIR)/bdpSchr.o $(LDLIBS) -o bdpSchr.out

bdpDirac: $(coreObjs) $(BUILDDIR)/dirBdp.o
	$(CXX) $(CXXFLAGS) $(coreObjs) $(BUILDDIR)/dirBdp.o $(LDLIBS) -o bdpDirac.out

dplDirac: $(coreObjs) $(BUILDDIR)/dplRun.o
	$(CXX) $(CXXFLAGS) $(coreObjs) $(BUILDDIR)/dplRun.o $(LDLIBS) -o d.out
	
unittest: $(coreObjs) $(BUILDDIR)/unittest.o
	$(CXX) $(CXXFLAGS) $(coreObjs) $(BUILDDIR)/unittest.o $(LDLIBS) -o b.out
	
reproject: $(coreObjs) $(BUILDDIR)/reproject.o
	$(CXX) $(CXXFLAGS) $(coreObjs) $(BUILDDIR)/reproject.o $(LDLIBS) -o rprj.out
	
mpitest: $(mpiObjs) $(BUILDDIR)/eigenMPI.o
	$(MPIXX) $(CXXFLAGS) $(coreObjs) $(BUILDDIR)/eigenMPI.o $(LDLIBS) -o EMPI.out
	
dipolempitest: $(mpiObjs) $(BUILDDIR)/dipoleMPItest.o
	$(MPIXX) $(CXXFLAGS) $(coreObjs) $(BUILDDIR)/dipoleMPItest.o $(LDLIBS) -o dipoleMPItest.out


clean:
	rm -f $(BUILDDIR)/*.o all
