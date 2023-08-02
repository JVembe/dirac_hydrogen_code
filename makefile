
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

dpltst: $(coreObjs) $(BUILDDIR)/dplTest.o
	$(CXX) $(CXXFLAGS) $(coreObjs) $(BUILDDIR)/dplTest.o $(LDLIBS) -o dpltst.out

cbdp: $(BUILDDIR)/carrierBDP.o $(coreObjs) 
	$(CXX) $(CXXFLAGS) $(coreObjs) $(BUILDDIR)/carrierBDP.o $(LDLIBS) -o cbdp.out

lcsztst: $(coreObjs) $(BUILDDIR)/lanczsosTest.o
	$(CXX) $(CXXFLAGS) $(coreObjs) $(BUILDDIR)/lanczsosTest.o $(LDLIBS) -o lcsztst.out
	
a2: $(coreObjs) $(BUILDDIR)/testwfunc.o
	$(CXX) $(CXXFLAGS) $(coreObjs) $(BUILDDIR)/testwfunc.o $(LDLIBS) -o a2.out

usemat: $(coreObjs) $(BUILDDIR)/matwfunc.o
	$(CXX) $(CXXFLAGS) $(coreObjs) $(BUILDDIR)/matwfunc.o $(LDLIBS) -o c.out
	
unittest: $(coreObjs) $(BUILDDIR)/unittest.o
	$(CXX) $(CXXFLAGS) $(coreObjs) $(BUILDDIR)/unittest.o $(LDLIBS) -o b.out

TAP: $(coreObjs) $(BUILDDIR)/testAngParts.o
	$(CXX) $(CXXFLAGS) $(coreObjs) $(BUILDDIR)/testAngParts.o $(LDLIBS) -o b.out

E5: $(coreObjs) $(BUILDDIR)/testwfuncE5.o
	$(CXX) $(CXXFLAGS) $(coreObjs) $(BUILDDIR)/testwfuncE5.o $(LDLIBS) -o E5.out
	
reproject: $(coreObjs) $(BUILDDIR)/reproject.o
	$(CXX) $(CXXFLAGS) $(coreObjs) $(BUILDDIR)/reproject.o $(LDLIBS) -o rprj.out
	
blocktest: $(coreObjs) $(BUILDDIR)/schrBlockTest.o
	$(CXX) $(CXXFLAGS) $(coreObjs) $(BUILDDIR)/schrBlockTest.o $(LDLIBS) -o sblt.out
	
mpitest: $(mpiObjs) $(BUILDDIR)/eigenMPI.o
	$(MPIXX) $(CXXFLAGS) $(coreObjs) $(BUILDDIR)/eigenMPI.o $(LDLIBS) -o EMPI.out


clean:
	rm -f $(BUILDDIR)/*.o all
