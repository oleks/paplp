CC=nvcc
NAME=paplp
FLAGS=-arch sm_13

paplp: problem.o paplp.cu
	${CC} ${FLAGS} problem.o ${NAME}.cu -o ${NAME}

problem.o: problem.cpp problem.hpp
	${CC} --compile problem.cpp

debug: paplp.cu
	${CC} -g -G ${NAME}.cu -o ${NAME}

run: all
	./${NAME}

clean:
	rm *.o
	rm ./${NAME}

all: paplp
