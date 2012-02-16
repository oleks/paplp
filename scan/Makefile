CC=nvcc
NAME=scan

scan: scan.cu
	${CC} scan.cu -o ${NAME}

debug: scan.cu
	${CC} -g -G scan.cu -o ${NAME}

run: scan
	./${NAME}

all: scan
