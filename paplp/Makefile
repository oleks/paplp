CC=nvcc
NAME=paplp

paplp: normalize.o paplp.cu
	${CC} normalize.o ${NAME}.cu -o ${NAME}

normalize.o: normalize.cpp
	${CC} --compile normalize.cpp

debug: paplp.cu
	${CC} -g -G ${NAME}.cu -o ${NAME}

run: all
	./${NAME}

clean:
	rm *.o
	rm ./${NAME}

all: paplp
