CC=nvcc
NAME=paplp
FLAGS=-arch sm_13

paplp: problem.o paplp.cu
	${CC} ${FLAGS} problem.o ${NAME}.cu -o ${NAME}

problem.o: problem.cu problem.hpp
	${CC} --compile problem.cu

special.o: special.cpp special.hpp
	${CC} --compile special.cpp

debug: paplp.cu
	${CC} -g -G ${NAME}.cu -o ${NAME}

run: all
	./${NAME}

clean:
	rm *.o
	rm ./${NAME}

all: paplp
