CXX ?= g++
CFLAGS = -Wall -Wconversion -O3 -fPIC -D _DENSE_REP
OS = $(shell uname)

all: svm-test-runner svm-structs svm

svm: svm.o
	if [ "$(OS)" = "Darwin" ]; then \
		SHARED_LIB_FLAG="-dynamiclib -Wl,-install_name,libsvm.so"; \
	else \
		SHARED_LIB_FLAG="-shared -Wl,-soname,libsvm.so"; \
	fi; \
	$(CXX) $${SHARED_LIB_FLAG} svm.o -o libsvm.so

svm-structs: svm-structs.o
	if [ "$(OS)" = "Darwin" ]; then \
		SHARED_LIB_FLAG="-dynamiclib -Wl,-install_name,libsvm-structs.so"; \
	else \
		SHARED_LIB_FLAG="-shared -Wl,-soname,libsvm-structs.so"; \
	fi; \
	$(CXX) $${SHARED_LIB_FLAG} svm-structs.o -o libsvm-structs.so

svm-test-runner: svm-test-runner.c svm-structs.o
	$(CXX) $(CFLAGS) svm-test-runner.c svm-structs.o -o ../bin/svm-test-runner -lm

svm-structs.o: svm-structs.c svm.h
	$(CXX) $(CFLAGS) -c svm-structs.c

svm.o: svm.cpp svm.h
	$(CXX) $(CFLAGS) -c svm.cpp
clean:
	rm -f *~ svm.o svm-structs.o libsvm*
