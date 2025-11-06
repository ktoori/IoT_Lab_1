#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

void fillMatrix(int* matrix, int r, int c) {
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c ; j++) {
            matrix[i * c + j] = rand() % 10;
        }
    }
}

void fillVector(int* vector, int n) {
    for (int i = 0; i < n; i++) {
        vector[i] = rand() % 10;
    }
}

void multCols(int* matrix, int* vector, int* result, int r, int c, int startCol, int endCol) {
    for (int i = 0; i < r; i++) {
        result[i] = 0;
    }
    for (int j = startCol; j < endCol; j++) {
        for (int i = 0; i < r; i++) {
            result[i] += matrix[i * c + j] * vector[j];
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

    if (my_rank != 0) {
        matrix = (int*)malloc(rows * cols * sizeof(int));
        vector = (int*)malloc(cols * sizeof(int));
        result = (int*)malloc(rows * sizeof(int));
    }

    MPI_Bcast(matrix, rows * cols, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(vector, cols, MPI_INT, 0, MPI_COMM_WORLD);

    int baseCols = cols / comm_sz;
    int extraCols = cols % comm_sz;

    int startCol;
    int endCol;

    if (my_rank < extraCols) {
        startCol = my_rank * (baseCols + 1);
        endCol = startCol + baseCols + 1;
    }
    else {
        startCol = my_rank * baseCols + extraCols;
        endCol = startCol + baseCols;
    }

    int* localResult = (int*)malloc(rows * sizeof(int));
    for (int i = 0; i < rows; i++) {
        localResult[i] = 0;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    timeStart = MPI_Wtime();

    multCols(matrix, vector, localResult, rows, cols, startCol, endCol);

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