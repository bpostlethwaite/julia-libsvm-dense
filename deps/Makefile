CXX ?= g++
CFLAGS = -Wall -Wconversion -O3 -fPIC -D _DENSE_REP
SHVER = 2
OS = $(shell uname)

all: svm-test

lib: svm.o
	if [ "$(OS)" = "Darwin" ]; then \
		SHARED_LIB_FLAG="-dynamiclib -Wl,-install_name,libsvm.so.$(SHVER)"; \
	else \
		SHARED_LIB_FLAG="-shared -Wl,-soname,libsvm.so.$(SHVER)"; \
	fi; \
	$(CXX) $${SHARED_LIB_FLAG} svm.o -o libsvm.so.$(SHVER)

svm-test: svm-test.c svm.o
	$(CXX) $(CFLAGS) svm-test.c svm.o -o svm-test -lm
svm.o: svm.cpp svm.h
	$(CXX) $(CFLAGS) -c svm.cpp
clean:
	rm -f *~ svm.o svm-test.o libsvm.so.$(SHVER)
