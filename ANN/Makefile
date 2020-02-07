##To make file type make
CC = g++
CFLAGS = -g

HEADER = include/matrix/matrix.h include/activations.h include/cost.h include/NeuralNetwork.h

SRCS= activations.cpp cost.cpp NeuralNetwork.cpp main.cpp

OBJS = $(SRCS:.c++=.o)

MAIN = main

.PHONY: depend clean

all:    $(MAIN)
		@echo  Message: executable file ...main... has been created

$(MAIN): $(OBJS) 
		$(CC) $(CFLAGS) $(HEADER) -o $(MAIN) $(OBJS)

.c++.o:
		$(CC) $(CFLAGS) $(HEADER) -cpp $<  -o $@

clean:
		$(RM) *.o *~ $(MAIN)

depend: $(SRCS)
		makedepend $(HEADER) $^