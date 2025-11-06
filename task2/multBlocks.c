#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>
#include <math.h>

void fillMatrix(int* matrix, int r, int c) {
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            matrix[i * c + j] = rand() % 10;
        }
    }
}

void fillVector(int* vector, int n) {
    for (int i = 0; i < n; i++) {
        vector[i] = rand() % 10;
    }
}

void multBlocks(int* matrix, int* vector, int* result, int cols, int startRow, int endRow, int startCol, int endCol) {
    for (int i = startRow; i < endRow; i++) {
        for (int j = startCol; j < endCol; j++) {
            result[i] += matrix[i * cols + j] * vector[j];
        }
    }
}

int main() {
    MPI_Init(NULL, NULL);

    int comm_sz;
    int my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    int rows;
    int cols;

    int* matrix = NULL;
    int* vector = NULL;
    int* result = NULL;
    int* localResult = NULL;

    double timeStart;
    double timeEnd;
    double localTime;
    double totalTime;

    if (my_rank == 0) {
        scanf("%d %d", &rows, &cols);
        printf("%d %d\n", rows, cols);

        matrix = (int*)malloc(rows * cols * sizeof(int));
        vector = (int*)malloc(cols * sizeof(int));
        result = (int*)malloc(rows * sizeof(int));

        srand(time(0));
        fillMatrix(matrix, rows, cols);
        fillVector(vector, cols);

        for (int i = 0; i < rows; i++) {
            result[i] = 0;
        }
    }

    MPI_Bcast(&rows, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int block = (int)sqrt(comm_sz);
    int blocksNum = block * block;
    int blockRows = rows / block;
    int blockCols = cols / block;

    if (my_rank != 0) {
        matrix = (int*)malloc(rows * cols * sizeof(int));
        vector = (int*)malloc(cols * sizeof(int));
        result = (int*)malloc(rows * sizeof(int));
    }

    localResult = (int*)malloc(rows * sizeof(int));
    for (int i = 0; i < rows; i++) {
        localResult[i] = 0;
    }

    MPI_Bcast(matrix, rows * cols, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(vector, cols, MPI_INT, 0, MPI_COMM_WORLD);

    int startRow = (my_rank / block) * blockRows;
    int endRow = startRow + blockRows;
    int startCol = (my_rank % block) * blockCols;
    int endCol = startCol + blockCols;

    if (my_rank < blocksNum) {
        if (my_rank == blocksNum - 1) {
            endRow = rows;
            endCol = cols;
        }
        else if ((my_rank + 1) % block == 0) {
            endCol = cols;
        }
        else if ((my_rank + 1) > blocksNum - block) {
            endRow = rows;
        }
    }

    MPI_Barrier(MPI_COMM_WORLD);
    timeStart = MPI_Wtime();

    if (my_rank < blocksNum) {
        multBlocks(matrix, vector, localResult, cols, startRow, endRow, startCol, endCol);
    }

    timeEnd = MPI_Wtime();
    localTime = timeEnd - timeStart;

    MPI_Reduce(localResult, result, rows, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

    MPI_Reduce(&localTime, &totalTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (my_rank == 0) {
        printf("\nMatrix:\n");
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                printf("%d ", matrix[i * cols + j]);
            }
            printf("\n");
        }

        printf("\nVector:\n");
        for (int i = 0; i < cols; i++) {
            printf("%d\n", vector[i]);
        }

        printf("\nResult:\n");
        for (int i = 0; i < rows; i++) {
            printf("%d\n", result[i]);
        }

        printf("\nExecution time: %.3f ms\n", totalTime * 1000.0);
    }

    free(matrix);
    free(vector);
    free(result);
    free(localResult);

    MPI_Finalize();
    return 0;
}